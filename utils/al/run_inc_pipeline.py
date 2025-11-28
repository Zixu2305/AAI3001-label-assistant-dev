#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incremental YOLO pipeline with two-teacher fusion in Phase-B.

- Phase A: new-only warm-up (unchanged)
- Phase B: compile FULL set (new + exemplars, GT-only val), then
           run two frozen teachers on TRAIN images:
             * old teacher (original weights) -> old classes only
             * new teacher (Phase-A best)     -> new class only
           fuse their boxes into labels/train with IoU de-dup
           train B1 (frozen) -> B2 (unfrozen) -> B2 tail

Outputs:
  meta/al/datasets/inc_traffic_light/inc_summary.json
"""
from __future__ import annotations
import os, json, shutil, hashlib
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from PIL import Image
import numpy as np

# ========= USER CONFIG =========
NEW_CLASS    = "traffic_light"
OLD_NAMES = [
    "car","truck","bus","trailer","construction_vehicle",
    "pedestrian","motorcycle","bicycle","traffic_cone","barrier"
]
ALL_NAMES = OLD_NAMES + [NEW_CLASS]
NEW_CLASS_IDX = len(OLD_NAMES)

# base model to expand (teacher for old)
BASE_WEIGHTS = "runs/20251109_174727__yolo11n__exp_yolo_n_v3/weights/best.pt"

# dataloader
IMGSZ   = 1280
BATCH   = 4
WORKERS = 2
SEED    = 1337

# Exemplar validation control (Phase-B val must include old + new GT)
EXEMPLAR_VAL_FRAC = 0.15
EXEMPLAR_FORCE_STRATIFY = True

# -------- Phase A (new-only) --------
PHASE_A_EPOCHS = 100
PHASE_A_LR0    = 1e-3
PHASE_A_FREEZE = 5

# -------- Phase B (full) schedule --------
# B1 (stabilize)  — conservative to protect old classes
PHASE_B1_EPOCHS = 50
PHASE_B1_LR0    = 1e-3
PHASE_B1_FREEZE = 6
OVERSAMPLE_B1   = 2

# B2a (adapt, unfreeze)
PHASE_B2A_EPOCHS = 20
PHASE_B2A_LR0    = 1e-3
OVERSAMPLE_B2A   = 2

# B2b (calibration tail)
PHASE_B2B_EPOCHS = 0
PHASE_B2B_LR0    = 1e-3
OVERSAMPLE_B2B   = 1

# Phase-B loss tilt (small-object friendly, but gentler)
PHASEB_BOX = 12.0
PHASEB_CLS = 0.9
PHASEB_DFL = 2.0
PHASEB_TAIL_BOX = 15.0
PHASEB_TAIL_CLS = 1.0
PHASEB_TAIL_DFL = 3.0

# Teacher pseudo for Phase-A new-only (old classes on train)
TEACHER_CONF_A = 0.70
TEACHER_IOU_A  = 0.60
TEACHER_MAX_OLD_PER_IMAGE = 8
TEACHER_HAS_CONF = True

# === Two-teacher fusion (used for Phase-B train) ===
# Old teacher = BASE_WEIGHTS / old classes only
KD_OLD_CONF = 0.60
KD_OLD_IOU  = 0.60
KD_OLD_MAX_PER_IMG = 10

# New teacher = Phase-A best / new class only
KD_NEW_CONF = 0.50
KD_NEW_IOU  = 0.60
KD_NEW_MAX_PER_IMG = 15

# De-dup thresholds when merging with existing GT labels
KD_DEDUP_IOU_SAMECLS = 0.50      # skip teacher box if a same-class GT/pseudo already overlaps > this
KD_MAX_TOTAL_PER_IMG = 150       # absolute safety cap

# Inputs from your S3 pull + exemplars
PULL_DIR   = sorted((Path("meta/al/run")).glob("pull_*"))[-1] / "compiled" / "detection"
MANIFEST_JSON = PULL_DIR / "manifest.json"
SPLITS_CSV    = PULL_DIR / "splits.csv"
EXEMPLAR_DET_MANIFEST = Path("meta/al/det_manifest_exemplar.csv")

# Destinations
DATASETS_ROOT = Path("meta/al/datasets")
PSEUDO_ROOT_A = Path("meta/al/pseudo/phaseA_teacher_old_on_newonly_train")
SUMMARY_PATH  = DATASETS_ROOT / "inc_traffic_light" / "inc_summary.json"

# ========= UTILS =========
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

def _sha_of_path(p: str) -> str:
    return sha256_file(Path(p))

def im_size(p: Path) -> Tuple[int,int]:
    with Image.open(p) as im:
        return im.size

def xywh_to_yolo(x, y, w, h, W, H):
    cx = (x + w/2.0) / W
    cy = (y + h/2.0) / H
    return cx, cy, w / W, h / H

def xyxy_to_yolo(x0, y0, x1, y1, W, H):
    bw = max(0.0, x1 - x0); bh = max(0.0, y1 - y0)
    cx = (x0 + x1) / 2.0 / W; cy = (y0 + y1) / 2.0 / H
    return cx, cy, bw / W, bh / H

def yolo_norm_to_xyxy(cx, cy, w, h):
    x0 = cx - w/2.0; y0 = cy - h/2.0
    x1 = cx + w/2.0; y1 = cy + h/2.0
    return x0, y0, x1, y1

def iou_xyxy(a, b):
    ax0,ay0,ax1,ay1 = a; bx0,by0,bx1,by1 = b
    iw = max(0.0, min(ax1,bx1)-max(ax0,bx0))
    ih = max(0.0, min(ay1,by1)-max(ay0,by0))
    inter = iw*ih
    if inter<=0: return 0.0
    ua = max(0.0,(ax1-ax0))*max(0.0,(ay1-ay0))
    ub = max(0.0,(bx1-bx0))*max(0.0,(by1-by0))
    union = ua+ub-inter + 1e-9
    return inter/union

def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    shutil.copy2(src, dst)

def ensure_clean_dir(p: Path):
    if p.exists(): shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def print_kv(d: dict, title: str):
    print(f"\n== {title} ==")
    for k,v in d.items():
        print(f"{k}: {v}")

def stratify_exemplars_into_splits(df: pd.DataFrame, seed: int, val_frac: float = 0.15) -> pd.DataFrame:
    dfx = df.copy()
    if "img_sha" not in dfx.columns:
        dfx["img_sha"] = dfx["img_path"].apply(lambda s: _sha_of_path(s))
    rng = np.random.RandomState(seed)
    val_shas = set()
    for cls, g in dfx.groupby("class_10"):
        shas = g["img_sha"].unique()
        if len(shas) == 0:
            continue
        k = max(1, int(round(len(shas) * val_frac)))
        pick = rng.choice(shas, size=min(k, len(shas)), replace=False)
        val_shas.update(pick.tolist())
    dfx["split"] = np.where(dfx["img_sha"].isin(val_shas), "val", "train")
    return dfx

def write_train_txt_with_oversample(root: Path, oversample_factor: int) -> Path:
    img_trn = root / "images" / "train"
    lab_trn = root / "labels" / "train"
    lines = []
    tl_stems, old_stems = [], []
    for lab in lab_trn.glob("*.txt"):
        stem = lab.stem
        txt = lab.read_text()
        if txt:
            if any(L.strip().startswith(f"{NEW_CLASS_IDX} ") for L in txt.splitlines()):
                tl_stems.append(stem)
            else:
                old_stems.append(stem)
    for s in tl_stems:
        p = img_trn / f"{s}.jpg"
        if p.exists():
            for _ in range(oversample_factor): lines.append(str(p))
    for s in old_stems:
        p = img_trn / f"{s}.jpg"
        if p.exists(): lines.append(str(p))
    out = root / "train.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"[oversample] TL imgs={len(tl_stems)} OLD imgs={len(old_stems)} "
          f"factor={oversample_factor} lines={len(lines)}")
    return out

def apply_train_spec(out_dir: Path, train_spec: str):
    data_yaml = out_dir / "data.yaml"
    y = data_yaml.read_text().splitlines()
    data_yaml.write_text("\n".join([f"train: {train_spec}" if ln.strip().startswith("train:")
                                    else ln for ln in y]) + "\n")
    print(f"[data.yaml] train -> {train_spec}")

# ========= COMPILE =========
def compile_dataset(run_tag: str, mode: str, merge_teacher_old: bool, teacher_dir: Path) -> Path:
    """
    mode: "newonly" or "full"
    Returns dataset root path.
    """
    assert MANIFEST_JSON.exists(), f"Missing {MANIFEST_JSON}"
    assert SPLITS_CSV.exists(),    f"Missing {SPLITS_CSV}"
    if mode == "full":
        assert EXEMPLAR_DET_MANIFEST.exists(), f"Missing {EXEMPLAR_DET_MANIFEST}"

    out_dir = (DATASETS_ROOT / run_tag).resolve()
    img_trn = out_dir / "images" / "train"
    img_val = out_dir / "images" / "val"
    lab_trn = out_dir / "labels" / "train"
    lab_val = out_dir / "labels" / "val"
    data_yaml = out_dir / "data.yaml"
    stats_json= out_dir / "stats.json"

    ensure_clean_dir(out_dir); img_trn.mkdir(parents=True); img_val.mkdir(parents=True)
    lab_trn.mkdir(parents=True); lab_val.mkdir(parents=True)

    manifest = json.loads(MANIFEST_JSON.read_text())
    sha_has_new = {it["image_sha256"] for it in manifest if any(a.get("class")==NEW_CLASS for a in it["anns"])}

    sha_to_split: Dict[str,str] = {}
    sha_to_img: Dict[str,Path] = {}
    sha_to_lines: Dict[str,List[str]] = {}

    # new class (always)
    for it in manifest:
        sha, split, local = it["image_sha256"], it["split"], Path(it["local_path"])
        W,H = im_size(local)
        lines=[]
        for ann in it["anns"]:
            if ann.get("class")!=NEW_CLASS: continue
            cx,cy,wn,hn = xywh_to_yolo(*ann["bbox_xywh"], W, H)
            if wn<=0 or hn<=0: continue
            lines.append(f"{NEW_CLASS_IDX} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")
        sha_to_split.setdefault(sha, split)
        sha_to_img.setdefault(sha, local)
        if lines: sha_to_lines.setdefault(sha, []).extend(lines)

    # exemplars (full only): include GT in TRAIN **and** VAL (pure GT; no teacher in val)
    if mode == "full":
        df = pd.read_csv(EXEMPLAR_DET_MANIFEST)
        if EXAMPLAR_FORCE_STRATIFY if 'EXAMPLAR_FORCE_STRATIFY' in globals() else EXEMPLAR_FORCE_STRATIFY:
            df = stratify_exemplars_into_splits(df, seed=SEED, val_frac=EXEMPLAR_VAL_FRAC)
        else:
            if "img_sha" not in df.columns:
                df["img_sha"] = df["img_path"].apply(lambda s: _sha_of_path(s))
            if "split" not in df.columns:
                df = stratify_exemplars_into_splits(df, seed=SEED, val_frac=EXEMPLAR_VAL_FRAC)

        sha_to_ex_split = {sha: grp["split"].iloc[0] for sha, grp in df.groupby("img_sha")}
        for img_sha, grp in df.groupby("img_sha"):
            imgp = Path(grp["img_path"].iloc[0])
            if not imgp.exists(): continue
            W, H = im_size(imgp)
            lines = []
            for _, r in grp.iterrows():
                name = str(r["class_10"])
                if name not in OLD_NAMES: continue
                idx = OLD_NAMES.index(name)
                cx, cy, wn, hn = xyxy_to_yolo(float(r["x0"]), float(r["y0"]),
                                              float(r["x1"]), float(r["y1"]), W, H)
                if wn <= 0 or hn <= 0: continue
                lines.append(f"{idx} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")
            if not lines: continue
            split = sha_to_ex_split.get(img_sha, "train")
            sha_to_split[img_sha] = split
            sha_to_img[img_sha] = imgp
            sha_to_lines.setdefault(img_sha, []).extend(lines)

    # new-only filter: drop non-new-class TRAIN images
    if mode == "newonly":
        for sha in list(sha_to_split.keys()):
            if sha_to_split[sha]=="train" and sha not in sha_has_new:
                sha_to_split.pop(sha,None); sha_to_img.pop(sha,None); sha_to_lines.pop(sha,None)

    # teacher merge for Phase-A (TRAIN only; cap per image)
    if merge_teacher_old and teacher_dir.exists():
        for pseudo in teacher_dir.glob("*.txt"):
            sha = pseudo.stem
            if sha_to_split.get(sha) != "train": continue
            add=[]; cnt=0
            for raw in pseudo.read_text().strip().splitlines():
                toks = raw.split()
                if not toks: continue
                if TEACHER_HAS_CONF and len(toks)>=6: toks = toks[:5]
                if len(toks)!=5: continue
                cls_idx = int(float(toks[0]))
                if not (0 <= cls_idx < len(OLD_NAMES)): continue
                add.append(" ".join([toks[0], toks[1], toks[2], toks[3], toks[4]]))
                cnt+=1
                if cnt>=TEACHER_MAX_OLD_PER_IMAGE: break
            if add: sha_to_lines.setdefault(sha, []).extend(add)
    elif merge_teacher_old:
        print(f"[warn] teacher dir {teacher_dir} missing; skipping pseudo merge.")

    # --- emit labels & images ---
    counts = {"images":{"train":0,"val":0},"boxes":{"train":0,"val":0},
              "boxes_per_class":{"train":{n:0 for n in ALL_NAMES},"val":{n:0 for n in ALL_NAMES}}}
    seen=set()
    for sha, split in sha_to_split.items():
        src = sha_to_img.get(sha)
        if not src or not Path(src).exists(): continue
        img_dst = (img_trn if split=="train" else img_val) / f"{sha}.jpg"
        lab_dst = (lab_trn if split=="train" else lab_val) / f"{sha}.txt"
        safe_copy(src, img_dst)
        lines = sha_to_lines.get(sha, [])
        lab_dst.write_text("\n".join(lines)+("\n" if lines else ""))
        if (sha,split) not in seen:
            counts["images"][split]+=1; seen.add((sha,split))
        for L in lines:
            cls_idx = int(float(L.split()[0])); name = ALL_NAMES[cls_idx]
            counts["boxes"][split]+=1; counts["boxes_per_class"][split][name]+=1

    data_yaml.write_text(
        f"path: {out_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        "names:\n" + "\n".join([f"  {i}: {n}" for i,n in enumerate(ALL_NAMES)]) + "\n"
    )
    stats = {
        "run_tag": run_tag, "names": ALL_NAMES,
        "images_train": counts["images"]["train"], "images_val": counts["images"]["val"],
        "boxes_train": counts["boxes"]["train"], "boxes_val": counts["boxes"]["val"],
        "boxes_per_class": counts["boxes_per_class"],
        "paths":{"root": str(out_dir), "data_yaml": str(data_yaml), "train_spec": "images/train"}
    }
    (out_dir/"stats.json").write_text(json.dumps(stats, indent=2))
    print_kv(stats, f"Compiled {run_tag}")
    return out_dir

# ========= TEACHER PSEUDO (Phase-A) =========
def export_teacher_pseudo_phaseA(dataset_root: Path) -> Path:
    from ultralytics import YOLO
    images_train = dataset_root / "images" / "train"
    out_dir = PSEUDO_ROOT_A
    out_dir.mkdir(parents=True, exist_ok=True)
    assert images_train.exists(), f"Missing {images_train}"
    print(f"[Phase-A teacher] predicting old classes on {images_train} …")
    model = YOLO(BASE_WEIGHTS)
    model.predict(
        source=str(images_train),
        imgsz=IMGSZ,
        conf=TEACHER_CONF_A,
        iou=TEACHER_IOU_A,
        classes=list(range(len(OLD_NAMES))),
        save_txt=True,
        save_conf=True,
        project=str(out_dir),
        name="",
        exist_ok=True
    )
    print(f"[Phase-A teacher] YOLO txt at {out_dir/'labels'}")
    return out_dir / "labels"

# ========= TWO-TEACHER FUSION (Phase-B) =========
def fuse_two_teachers_into_train(ds_full: Path, phaseA_weights: str):
    """
    Run T_old (BASE_WEIGHTS) for old classes and T_new (phaseA_weights) for the new class
    on ds_full/images/train/*.jpg, and merge into labels/train/*.txt with IoU de-dup.
    """
    from ultralytics import YOLO

    img_dir = ds_full / "images" / "train"
    lab_dir = ds_full / "labels" / "train"
    assert img_dir.exists() and lab_dir.exists()

    # Load teachers (no grad, eval)
    T_old = YOLO(BASE_WEIGHTS)
    T_new = YOLO(phaseA_weights)

    # Collect predictions
    print("[KD] Running old-teacher predictions (old classes only)…")
    preds_old = {}
    for r in T_old.predict(source=str(img_dir), imgsz=IMGSZ, conf=KD_OLD_CONF, iou=KD_OLD_IOU,
                           classes=list(range(len(OLD_NAMES))), verbose=False, stream=True):
        stem = Path(r.path).stem
        if not len(r.boxes): continue
        boxes = r.boxes.xyxy.cpu().numpy()
        clses = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        keep = []
        for (x0,y0,x1,y1), c, p in zip(boxes, clses, confs):
            if len(keep) >= KD_OLD_MAX_PER_IMG: break
            keep.append((c, float(x0), float(y0), float(x1), float(y1)))
        preds_old[stem] = keep

    print("[KD] Running new-teacher predictions (new class only)…")
    preds_new = {}
    for r in T_new.predict(source=str(img_dir), imgsz=IMGSZ, conf=KD_NEW_CONF, iou=KD_NEW_IOU,
                           classes=[NEW_CLASS_IDX], verbose=False, stream=True):
        stem = Path(r.path).stem
        if not len(r.boxes): continue
        boxes = r.boxes.xyxy.cpu().numpy()
        keep = []
        for (x0,y0,x1,y1) in boxes[:KD_NEW_MAX_PER_IMG]:
            keep.append((NEW_CLASS_IDX, float(x0), float(y0), float(x1), float(y1)))
        preds_new[stem] = keep

    # Merge into labels with IoU de-dup against existing lines
    total_added = 0
    for img_path in img_dir.glob("*.jpg"):
        stem = img_path.stem
        lab_path = lab_dir / f"{stem}.txt"
        txt = lab_path.read_text() if lab_path.exists() else ""
        lines = [L.strip() for L in txt.splitlines() if L.strip()]
        # Build existing boxes (normalized xyxy in [0,1])
        exist = []
        for L in lines:
            toks = L.split()
            if len(toks)!=5: continue
            c = int(float(toks[0])); cx,cy,w,h = map(float, toks[1:])
            x0,y0,x1,y1 = yolo_norm_to_xyxy(cx,cy,w,h)
            exist.append((c,(x0,y0,x1,y1)))

        W,H = im_size(img_path)
        def ok_to_add(c, x0,y0,x1,y1):
            # compare with same-class existing boxes
            cand = (x0/W, y0/H, x1/W, y1/H)
            for ec,(ex0,ey0,ex1,ey1) in exist:
                if ec!=c: continue
                if iou_xyxy(cand,(ex0,ey0,ex1,ey1)) >= KD_DEDUP_IOU_SAMECLS:
                    return False
            return True

        added_lines=[]
        for store in (preds_old.get(stem, []), preds_new.get(stem, [])):
            for c,x0,y0,x1,y1 in store:
                if len(lines) + len(added_lines) >= KD_MAX_TOTAL_PER_IMG: break
                if ok_to_add(c,x0,y0,x1,y1):
                    cx,cy,wn,hn = xyxy_to_yolo(x0,y0,x1,y1,W,H)
                    added_lines.append(f"{c} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")

        if added_lines:
            lab_path.write_text((txt + ("\n" if txt and not txt.endswith("\n") else "")) +
                                "\n".join(added_lines) + "\n")
            total_added += len(added_lines)

    print(f"[KD] Fusion added {total_added} boxes into labels/train")

# ========= TRAIN / VAL / SUMMARY =========
def _find_col(cols, keys):
    low = [c.lower() for c in cols]
    for i,c in enumerate(low):
        if any(k in c for k in keys):
            return i
    return None

def read_results_csv(run_dir: Path) -> dict:
    csvp = run_dir / "results.csv"
    out = {}
    if not csvp.exists():
        return out
    df = pd.read_csv(csvp)
    i_map = _find_col(df.columns, ["map50-95","map_50_95","map50_95","metrics/mAP50-95"])
    i_p   = _find_col(df.columns, ["precision","metrics/precision"])
    i_r   = _find_col(df.columns, ["recall","metrics/recall"])
    if i_map is not None:
        m = df.iloc[:, i_map].astype(float)
        out["mAP50-95_final"] = float(m.iloc[-1]); out["mAP50-95_best"] = float(m.max())
    if i_p is not None:
        p = df.iloc[:, i_p].astype(float)
        out["precision_final"] = float(p.iloc[-1]); out["precision_best"] = float(p.max())
    if i_r is not None:
        r = df.iloc[:, i_r].astype(float)
        out["recall_final"] = float(r.iloc[-1]); out["recall_best"] = float(r.max())
    return out

def yolo_train(weights: str, data_yaml: Path, project_dir: Path, run_name: str,
               epochs: int, lr0: float, freeze=None, loss_bias=None) -> tuple[str, dict]:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    args = dict(
        data=str(data_yaml), epochs=int(epochs), imgsz=int(IMGSZ), batch=int(BATCH),
        seed=int(SEED), optimizer="AdamW", lr0=float(lr0),
        mosaic=0.0, mixup=0.0, rect=True, workers=int(WORKERS),
        project=str(project_dir), name=run_name, exist_ok=True, verbose=True
    )
    if freeze is not None: args["freeze"] = freeze
    if loss_bias is not None:
        args.update(dict(box=loss_bias["box"], cls=loss_bias["cls"], dfl=loss_bias["dfl"]))
    LOGGER.info(f"== TRAIN {run_name} =="); LOGGER.info(f"weights={weights}")
    model = YOLO(weights)
    model.train(**args)
    run_dir = project_dir / run_name
    best = run_dir / "weights" / "best.pt"
    if not best.exists(): raise FileNotFoundError(f"best.pt not found at {best}")
    metrics = read_results_csv(run_dir); metrics.update({"run_dir": str(run_dir), "best_path": str(best)})
    return str(best), metrics

def yolo_val(model_weights: str, data_yaml: Path, conf=0.001, iou=0.6) -> dict:
    from ultralytics import YOLO
    model = YOLO(model_weights)
    res = model.val(data=str(data_yaml), conf=conf, iou=iou)
    d = {}
    try:
        d = {
            "val_mAP50-95": float(res.results_dict.get("metrics/mAP50-95(B)", 0.0)),
            "val_precision": float(res.results_dict.get("metrics/precision(B)", 0.0)),
            "val_recall": float(res.results_dict.get("metrics/recall(B)", 0.0)),
        }
    except Exception:
        pass
    return d

def conf_sweep(model_weights: str, data_yaml: Path, iou=0.6, grid=None) -> list[dict]:
    from ultralytics import YOLO
    if grid is None:
        grid = [round(x, 2) for x in [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]]
    model = YOLO(model_weights)
    rows = []
    for conf in grid:
        res = model.val(data=str(data_yaml), conf=conf, iou=iou)
        P = float(res.results_dict.get("metrics/precision(B)", 0.0))
        R = float(res.results_dict.get("metrics/recall(B)", 0.0))
        F1 = (2*P*R)/(P+R+1e-9)
        rows.append({"conf": conf, "precision": P, "recall": R, "F1": F1})
    return rows

# ========= PIPELINE =========
def main():
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    summary = {"datasets": {}, "phases": {}, "deploy": {}}

    # --- Compile NEW-ONLY (no teacher) ---
    ds_newonly = compile_dataset(run_tag=f"inc_{NEW_CLASS}_newonly", mode="newonly",
                                 merge_teacher_old=False, teacher_dir=Path("unused"))
    summary["datasets"]["newonly"] = json.loads((ds_newonly/"stats.json").read_text())

    # --- Phase-A: teacher pseudo on new-only TRAIN (old classes) ---
    pseudo_dir_A = export_teacher_pseudo_phaseA(ds_newonly)

    # --- Re-compile NEW-ONLY with teacher merge (capped) ---
    ds_newonly = compile_dataset(run_tag=f"inc_{NEW_CLASS}_newonly", mode="newonly",
                                 merge_teacher_old=True, teacher_dir=pseudo_dir_A)
    summary["datasets"]["newonly_merged"] = json.loads((ds_newonly/"stats.json").read_text())

    # --- Phase A: head-focused on NEW-ONLY ---
    bestA, metA = yolo_train(
        weights=BASE_WEIGHTS,
        data_yaml=ds_newonly / "data.yaml",
        project_dir=ds_newonly,
        run_name="inc_phaseA_head_only",
        epochs=PHASE_A_EPOCHS,
        lr0=PHASE_A_LR0,
        freeze=PHASE_A_FREEZE,
        loss_bias=None
    )
    valA = yolo_val(bestA, ds_newonly / "data.yaml")
    metA.update(valA); summary["phases"]["A"] = metA

    # --- Compile FULL (new + exemplars, NO Phase-A teacher merge here) ---
    ds_full = compile_dataset(run_tag=f"inc_{NEW_CLASS}", mode="full",
                              merge_teacher_old=False, teacher_dir=Path("unused"))
    summary["datasets"]["full_gt"] = json.loads((ds_full/"stats.json").read_text())

    # --- Two-teacher fusion into TRAIN labels (old teacher + Phase-A teacher) ---
    fuse_two_teachers_into_train(ds_full, phaseA_weights=bestA)

    # Apply oversample & write data.yaml train spec
    train_txt = write_train_txt_with_oversample(ds_full, OVERSAMPLE_B1)
    apply_train_spec(ds_full, str(train_txt))

    # --- Phase B1: freeze backbone, gentle loss tilt ---
    bestB1, metB1 = yolo_train(
        weights=BASE_WEIGHTS,
        data_yaml=ds_full / "data.yaml",
        project_dir=ds_full,
        run_name="inc_phaseB1_fused_frozen",
        epochs=PHASE_B1_EPOCHS,
        lr0=PHASE_B1_LR0,
        freeze=PHASE_B1_FREEZE,
        loss_bias=dict(box=PHASEB_BOX, cls=PHASEB_CLS, dfl=PHASEB_DFL)
    )
    valB1 = yolo_val(bestB1, ds_full / "data.yaml")
    metB1.update(valB1); summary["phases"]["B1"] = metB1

    # B2a oversample (keep high for TL)
    train_txt = write_train_txt_with_oversample(ds_full, OVERSAMPLE_B2A)
    apply_train_spec(ds_full, str(train_txt))

    # --- Phase B2a: unfreeze ---
    bestB2a, metB2a = yolo_train(
        weights=bestB1,
        data_yaml=ds_full / "data.yaml",
        project_dir=ds_full,
        run_name="inc_phaseB2a_fused_unfrozen",
        epochs=PHASE_B2A_EPOCHS,
        lr0=PHASE_B2A_LR0,
        freeze=None,
        loss_bias=dict(box=PHASEB_BOX, cls=PHASEB_CLS, dfl=PHASEB_DFL)
    )
    valB2a = yolo_val(bestB2a, ds_full / "data.yaml")
    metB2a.update(valB2a); summary["phases"]["B2a"] = metB2a

    # --- Optional Phase B2b: short calibration tail ---
    # If PHASE_B2B_EPOCHS <= 0, we just skip extra training and reuse B2a.
    bestB2b = bestB2a
    metB2b = dict(metB2a)

    if PHASE_B2B_EPOCHS and PHASE_B2B_EPOCHS > 0:
        # B2b oversample (anneal down)
        train_txt = write_train_txt_with_oversample(ds_full, OVERSAMPLE_B2B)
        apply_train_spec(ds_full, str(train_txt))

        bestB2b, metB2b = yolo_train(
            weights=bestB2a,
            data_yaml=ds_full / "data.yaml",
            project_dir=ds_full,
            run_name="inc_phaseB2b_fused_tail",
            epochs=PHASE_B2B_EPOCHS,
            lr0=PHASE_B2B_LR0,
            freeze=None,
            loss_bias=dict(box=PHASEB_TAIL_BOX, cls=PHASEB_TAIL_CLS, dfl=PHASEB_TAIL_DFL),
        )
        valB2b = yolo_val(bestB2b, ds_full / "data.yaml")
        metB2b.update(valB2b)

    summary["phases"]["B2b"] = metB2b

    # --- Confidence sweep (to pick deploy thresholds) ---
    sweep = conf_sweep(bestB2b, ds_full / "data.yaml", iou=0.6)
    summary["deploy"]["conf_sweep"] = sweep
    summary["final_weights"] = bestB2b

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print("\n[summary] wrote:", SUMMARY_PATH)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
