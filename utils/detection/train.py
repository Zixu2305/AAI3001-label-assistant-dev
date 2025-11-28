#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultralytics 8.3.221 trainer (COCO -> YOLO conversion built-in)

- Reusable converted dataset (skip rebuild between runs)
- Prefer symlink/hardlink to avoid copies
- 'exist_ok=True' to prevent ..._v2 run dirs
- ReduceLROnPlateau on val mAP50-95 (mode='max', with threshold)
- Saves best -> models/detector/v1.0/model.pt (robust even with ...v2 runs)
- Exports metrics.json (mAP, mAP50, AP small/med/large)
- Plots curves: curve_map.png, curve_ap_sml.png
"""
import os
import csv
import json, math
import shutil
import argparse
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO
from ultralytics.utils import LOGGER


# ----------------------------- small utils -----------------------------
def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _expand_vars(s: str, mapping: dict) -> str:
    for _ in range(4):
        try:
            s = s.format(**mapping)
        except KeyError:
            break
    return os.path.abspath(os.path.expanduser(s))


def resolve_paths(cfg: dict) -> dict:
    p = cfg["paths"]
    resolved = {k: v for k, v in p.items()}
    for _ in range(4):
        for k, v in list(resolved.items()):
            if isinstance(v, str):
                resolved[k] = _expand_vars(v, resolved)
    cfg["paths"] = resolved
    return cfg


def must_exist(path: str, label: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


# ------------------------ class-name loading ------------------------
def _load_class_names(cfg) -> list[str]:
    pj = cfg["paths"]
    cj = pj.get("classes_json", None)
    if cj and os.path.exists(cj):
        with open(cj, "r") as f:
            obj = json.load(f)

        if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
            return obj

        if isinstance(obj, dict):
            if isinstance(obj.get("names"), list):
                return obj["names"]
            if isinstance(obj.get("id_to_name"), dict):
                items = [(int(k), v) for k, v in obj["id_to_name"].items() if str(k).isdigit()]
                items.sort(key=lambda x: x[0])
                if items:
                    return [v for _, v in items]
            if isinstance(obj.get("name_to_id"), dict):
                items = [(int(v), k) for k, v in obj["name_to_id"].items() if isinstance(v, int)]
                items.sort(key=lambda x: x[0])
                if items:
                    return [v for _, v in items]
            items = [(int(k), v) for k, v in obj.items() if str(k).isdigit()]
            items.sort(key=lambda x: x[0])
            if items:
                return [v for _, v in items]

    with open(pj["train_json"], "r") as f:
        coco = json.load(f)
    cats = coco.get("categories", [])

    def _to_int(x):
        try:
            return int(x)
        except Exception:
            return x

    cats = sorted(cats, key=lambda cat: _to_int(cat.get("id", 0)))
    return [cat.get("name", f"class_{i}") for i, cat in enumerate(cats)]


# ------------------- COCO -> YOLO converter -------------------
def _xywh_to_yolo(x, y, w, h, iw, ih):
    cx = x + w / 2.0
    cy = y + h / 2.0
    return cx / iw, cy / ih, w / iw, h / ih


def convert_coco_split_to_yolo(cfg, coco_json_path, images_root, out_images_dir, out_labels_dir, cat_id_to_idx):
    ensure_dir(out_images_dir)
    ensure_dir(out_labels_dir)
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    anns_by_img = {}
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    prefer_symlink = cfg.get("conversion", {}).get("prefer_symlink", True)
    n_imgs = 0
    for img_id, info in images.items():
        file_name = info["file_name"]
        iw, ih = info.get("width"), info.get("height")
        src = os.path.join(images_root, file_name)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Image not found: {src}")

        ext = os.path.splitext(file_name)[1].lower()
        dst_img = os.path.join(out_images_dir, f"{img_id}{ext}")
        if not os.path.exists(dst_img):
            linked = False
            if prefer_symlink:
                try:
                    os.symlink(src, dst_img); linked = True
                except Exception:
                    pass
            if not linked:
                try:
                    os.link(src, dst_img); linked = True
                except Exception:
                    pass
            if not linked:
                shutil.copy2(src, dst_img)

        dst_lab = os.path.join(out_labels_dir, f"{img_id}.txt")
        lines = []
        for ann in anns_by_img.get(img_id, []):
            cat = ann["category_id"]
            if cat not in cat_id_to_idx:
                continue
            cls = cat_id_to_idx[cat]
            x, y, w, h = ann["bbox"]
            cx, cy, nw, nh = _xywh_to_yolo(x, y, w, h, iw, ih)
            cx = min(max(cx, 0.0), 1.0); cy = min(max(cy, 0.0), 1.0)
            nw = min(max(nw, 0.0), 1.0); nh = min(max(nh, 0.0), 1.0)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        with open(dst_lab, "w") as f:
            f.write("\n".join(lines))
        n_imgs += 1
    LOGGER.info(f"[convert] {n_imgs} images → {out_images_dir}, labels → {out_labels_dir}")


def build_yolo_dataset_from_coco(cfg, run_dir) -> str:
    pj = cfg["paths"]
    names = _load_class_names(cfg)

    with open(pj["train_json"], "r") as f:
        coco_tr = json.load(f)
    cats = coco_tr.get("categories", [])
    cats = sorted(cats, key=lambda c: int(c["id"]) if str(c.get("id", "")).isdigit() else c.get("id", 0))
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}

    yolo_root = ensure_dir(os.path.join(run_dir, "yolo_ds"))
    im_tr = ensure_dir(os.path.join(yolo_root, "images", "train"))
    lb_tr = ensure_dir(os.path.join(yolo_root, "labels", "train"))
    im_va = ensure_dir(os.path.join(yolo_root, "images", "val"))
    lb_va = ensure_dir(os.path.join(yolo_root, "labels", "val"))

    convert_coco_split_to_yolo(cfg, pj["train_json"], pj["dataset_root"], im_tr, lb_tr, cat_id_to_idx)
    convert_coco_split_to_yolo(cfg, pj["val_json"],   pj["dataset_root"], im_va, lb_va, cat_id_to_idx)

    data_yaml = os.path.join(run_dir, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"path: {yolo_root}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(names)}\n")
        f.write("names:\n")
        for i, n in enumerate(names):
            n_str = str(n).replace('"', '\\"')
            f.write(f"  {i}: \"{n_str}\"\n")
    return data_yaml


def try_reuse_yolo_ds(cfg, run_dir) -> str | None:
    reuse = cfg.get("conversion", {}).get("reuse_yolo_ds", "")
    if not reuse:
        return None
    reuse_abs = os.path.abspath(reuse)
    im_tr = os.path.join(reuse_abs, "images", "train")
    im_va = os.path.join(reuse_abs, "images", "val")
    lb_tr = os.path.join(reuse_abs, "labels", "train")
    lb_va = os.path.join(reuse_abs, "labels", "val")
    if all(os.path.exists(p) for p in (im_tr, im_va, lb_tr, lb_va)):
        LOGGER.info(f"[reuse] Using existing YOLO dataset at {reuse_abs}")
        names = _load_class_names(cfg)
        data_yaml = os.path.join(run_dir, "data.yaml")
        with open(data_yaml, "w") as f:
            f.write(f"path: {reuse_abs}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write(f"nc: {len(names)}\n")
            f.write("names:\n")
            for i, n in enumerate(names):
                n_str = str(n).replace('"', '\\"')
                f.write(f"  {i}: \"{n_str}\"\n")
        return data_yaml
    return None


# --------------------------- scheduler cb ---------------------------
class PlateauSchedulerCB:
    """ReduceLROnPlateau stepped on val mAP50-95 each epoch."""
    def __init__(self, trainer, factor, patience, threshold, min_lr):
        import torch
        self.opt = trainer.optimizer
        self.min_lr = float(min_lr)
        self.rlrp = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode="max",                # mAP: higher is better
            factor=factor,
            patience=patience,
            threshold=threshold,       # relative (default) improvement threshold
            min_lr=min_lr,
        )
        LOGGER.info(f"[RLRP] factor={factor} patience={patience} min_lr={min_lr} threshold={threshold}")

    def step(self, metric: float):
        self.rlrp.step(metric)
        for g in self.opt.param_groups:
            if g["lr"] < self.min_lr:
                g["lr"] = self.min_lr


# ------------------------------- plots -------------------------------
def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def plot_training_curves(save_dir: Path):
    import matplotlib.pyplot as plt

    csv_path = Path(save_dir) / "results.csv"
    if not csv_path.exists():
        LOGGER.warning(f"[plots] results.csv not found at {csv_path}")
        return
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    xs = list(range(1, len(rows) + 1))
    k_map = ("metrics/mAP50-95(B)", "metrics/mAP50(B)")
    k_aps = ("metrics/APsmall(B)", "metrics/APmedium(B)", "metrics/APlarge(B)")

    y_map = [[_safe_float(r.get(k)) for r in rows] for k in k_map]
    y_aps = [[_safe_float(r.get(k)) for r in rows] for k in k_aps]

    # mAP curves
    plt.figure()
    for y, label in zip(y_map, ("mAP50-95", "mAP50")):
        plt.plot(xs, y, label=label)
    plt.xlabel("epoch"); plt.ylabel("AP"); plt.title("mAP over epochs"); plt.legend()
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "curve_map.png", dpi=160)
    plt.close()

    # AP small/medium/large
    if any(any(v is not None for v in y) for y in y_aps):
        import matplotlib.pyplot as plt  # keep isolated state
        plt.figure()
        for y, label in zip(y_aps, ("AP small", "AP medium", "AP large")):
            plt.plot(xs, y, label=label)
        plt.xlabel("epoch"); plt.ylabel("AP"); plt.title("AP by object size"); plt.legend()
        plt.tight_layout()
        plt.savefig(Path(save_dir) / "curve_ap_sml.png", dpi=160)
        plt.close()


# ------------------------------- main -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="meta/detection/train_config.json")
    ap.add_argument("--run_name", default="", help="optional run suffix")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)
    cfg = resolve_paths(cfg)
    pj = cfg["paths"]

    # preflight
    for k in ("train_json", "val_json", "runs_dir", "models_out"):
        if k in ("runs_dir", "models_out"):
            ensure_dir(pj[k])
        else:
            must_exist(pj[k], k)
    must_exist(pj["dataset_root"], "dataset_root")

    run_id = f"{ts()}__{cfg['model']['arch']}"
    if args.run_name:
        run_id += f"__{args.run_name}"
    run_dir = ensure_dir(os.path.join(pj["runs_dir"], run_id))

    # persist configs
    shutil.copy2(args.config, os.path.join(run_dir, "config.input.json"))
    with open(os.path.join(run_dir, "config.resolved.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # reuse converted dataset if available, else build and optionally move to reuse path
    data_yaml = try_reuse_yolo_ds(cfg, run_dir)
    if data_yaml is None:
        data_yaml = build_yolo_dataset_from_coco(cfg, run_dir)
        reuse = cfg.get("conversion", {}).get("reuse_yolo_ds", "")
        if reuse:
            src = os.path.join(run_dir, "yolo_ds")
            dst = os.path.abspath(reuse)
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)
                LOGGER.info(f"[reuse] Moved converted dataset to {dst}")
            # rewrite data.yaml to point to reused path
            names = _load_class_names(cfg)
            with open(os.path.join(run_dir, "data.yaml"), "w") as f:
                f.write(f"path: {dst}\ntrain: images/train\nval: images/val\n")
                f.write(f"nc: {len(names)}\nnames:\n")
                for i, n in enumerate(names):
                    n_str = str(n).replace('"', '\\"')
                    f.write(f"  {i}: \"{n_str}\"\n")

    # Load model
    model = YOLO(cfg["model"]["weights"])

    # Build training args (imgsz must be INT for train/val in 8.3.221)
    imgsz_int = int(cfg["model"]["img_size"])
    raw_args = {
        "data": data_yaml,
        "epochs": int(cfg["train"]["epochs_max"]),
        "imgsz": imgsz_int,
        "batch": int(cfg["model"]["batch"]),
        "workers": int(cfg["model"]["workers"]),
        "device": cfg["model"]["device"],
        "project": pj["runs_dir"],
        "name": run_id,
        "exist_ok": True,  # do not make ..._v2
        "save_period": int(cfg["train"]["save_period"]),
        "patience": int(cfg["train"]["patience_early_stop"]),
        "optimizer": cfg["train"]["optimizer"],
        "lr0": float(cfg["train"]["lr0"]),
        "weight_decay": float(cfg["train"]["weight_decay"]),
        # speed/loader knobs
        "cache": cfg["train"].get("cache", False),
        "pin_memory": bool(cfg["train"].get("pin_memory", True)),
        "persistent_workers": bool(cfg["train"].get("persistent_workers", True)),
        "prefetch_factor": int(cfg["train"].get("prefetch_factor", 4)),
        "plots": bool(cfg["train"].get("plots", True)),
        "accumulate": int(cfg["train"].get("accumulate", 1)),
        "deterministic": bool(cfg["train"].get("deterministic", False)),
        # light aug only
        "auto_augment": cfg.get("augment", {}).get("auto_augment", "none"),
        "hsv_h": float(cfg["augment"]["hsv_h"]),
        "hsv_s": float(cfg["augment"]["hsv_s"]),
        "hsv_v": float(cfg["augment"]["hsv_v"]),
        "fliplr": float(cfg["augment"]["flip_p"]),
        "flipud": 0.0,
        "scale": float(cfg["augment"]["scale"]),
        "translate": float(cfg["augment"]["translate"]),
        "mosaic": float(cfg["augment"]["mosaic"]),
        "mixup": float(cfg["augment"]["mixup"]),
        "perspective": float(cfg["augment"]["perspective"]),
        "rect": True,
        "single_cls": False,
        "channels_last": bool(cfg["model"].get("channels_last", False)),
        "box": float(cfg["loss"]["box"]),
        "cls": float(cfg["loss"]["cls"]),
        "dfl": float(cfg["loss"]["dfl"]),
    }

    # Filter args to what this Ultralytics build accepts
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        allowed = set(DEFAULT_CFG_DICT.keys())
    except Exception:
        allowed = set(raw_args.keys())
    train_args = {k: v for k, v in raw_args.items() if k in allowed}

    # RLRP callback on mAP50-95
    rlrp_holder = {"obj": None}

    def on_fit_epoch_end(trainer):
        metric = float(getattr(trainer.validator.metrics.box, "map", 0.0))  # mAP50-95
        if rlrp_holder["obj"] is None:
            rlrp_holder["obj"] = PlateauSchedulerCB(
                trainer,
                factor=cfg["scheduler"]["factor"],
                patience=cfg["scheduler"]["patience"],
                threshold=cfg["scheduler"]["threshold"],
                min_lr=cfg["scheduler"]["min_lr"],
            )
        rlrp_holder["obj"].step(metric)
        LOGGER.info(
            f"[RLRP] epoch {trainer.epoch} mAP50-95={metric:.4f} "
            f"lr={[round(g['lr'],7) for g in trainer.optimizer.param_groups]}"
        )

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    # Train
    results = model.train(**train_args)

    # The actual save_dir Ultralytics used (handles ..._v2)
    try:
        used_save_dir = Path(model.trainer.save_dir)
    except Exception:
        globs = sorted(Path(pj["runs_dir"]).glob(f"{run_id}*/weights/best.pt"))
        used_save_dir = globs[-1].parents[1] if globs else Path(pj["runs_dir"]) / run_id

    # Optional extra plots (from results.csv, if available)
    try:
        plot_training_curves(used_save_dir)
    except Exception as e:
        LOGGER.warning(f"[plots] failed to generate curves: {e}")

    # One-shot COCO-style validation to get AP_S/M/L (and write COCO JSON)
    val_coco = model.val(
		data=data_yaml,
		split="val",
		save_json=True,             
		imgsz=imgsz_int,
		device=cfg["model"]["device"],
		project=str(used_save_dir),     
		name="val_coco",             
		exist_ok=True
	)

    # Save best -> canonical path
    best_src = used_save_dir / "weights" / "best.pt"
    best_dst = Path(pj["models_out"]) / "model.pt"
    ensure_dir(pj["models_out"])
    if not best_src.exists():
        raise FileNotFoundError(f"best.pt not found at {best_src}")
    shutil.copy2(best_src, best_dst)
    print(f"✓ Saved best model to: {best_dst}")
    print(f"✓ Run dir: {used_save_dir}")

    # -------- Write metrics.json robustly (DetMetrics or dict) --------
    def _as_metrics_dict(m):
        if m is None:
            return {}
        if isinstance(m, dict):
            return m
        rd = getattr(m, "results_dict", None)
        if callable(rd):
            try:
                return rd()
            except Exception:
                pass
        if isinstance(rd, dict):
            return rd
        try:
            return dict(getattr(m, "__dict__", {}))
        except Exception:
            return {}

    def _safe_float(x):
        try:
            v = float(x)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        except Exception:
            return None

    def _pick(d, keys):
        if not isinstance(d, dict):
            return None
        for k in keys:
            if k in d and d[k] not in (None, ""):
                return d[k]
        return None

    train_metrics = _as_metrics_dict(results)
    coco_metrics  = _as_metrics_dict(val_coco)

    # Try common key variants for size-bucketed APs from COCO eval
    ap_small_val  = _pick(coco_metrics,  ["ap_small", "APs", "AP_s", "ap_s", "aps", "metrics/APsmall(B)"])
    ap_medium_val = _pick(coco_metrics,  ["ap_medium", "APm", "AP_m", "ap_m", "apmedium", "metrics/APmedium(B)"])
    ap_large_val  = _pick(coco_metrics,  ["ap_large", "APl", "AP_l", "ap_l", "aplarge", "metrics/APlarge(B)"])

    # Fallback to whatever the trainer might have exposed on train_metrics
    if ap_small_val is None:
        ap_small_val = _pick(train_metrics, ["ap_small", "APs", "AP_s", "metrics/APsmall(B)"])
    if ap_medium_val is None:
        ap_medium_val = _pick(train_metrics, ["ap_medium", "APm", "AP_m", "metrics/APmedium(B)"])
    if ap_large_val is None:
        ap_large_val = _pick(train_metrics, ["ap_large", "APl", "AP_l", "metrics/APlarge(B)"])

    # mAP@.5:.95 and mAP@.5 (with B variants if present)
    mAP5095 = _safe_float(
        train_metrics.get("metrics/mAP50-95(B)", train_metrics.get("metrics/mAP50-95", 0.0))
    )
    mAP50   = _safe_float(
        train_metrics.get("metrics/mAP50(B)",    train_metrics.get("metrics/mAP50",    0.0))
    )

    # epochs_ran: derive from results.csv if not present
    epochs_csv = None
    csv_path = used_save_dir / "results.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            epochs_csv = sum(1 for _ in f) - 1  # minus header

    payload = {
        "best_weights": str(best_dst),
        "imgsz": imgsz_int,
        "epochs_ran": int(train_metrics.get("epoch", epochs_csv or 0)),
        "val": {
            "map_50_95": mAP5095,
            "map_50":    mAP50,
            "ap_small":  _safe_float(ap_small_val),
            "ap_medium": _safe_float(ap_medium_val),
            "ap_large":  _safe_float(ap_large_val),
        },
    }
    with open(cfg["logging"]["metrics_json_path"].format(**pj), "w") as f:
        json.dump(payload, f, indent=2)
    # -------------------------------------------------------------------

    # Optional cleanup
    if cfg.get("conversion", {}).get("cleanup_after_train", False):
        yolo_root = used_save_dir / "yolo_ds"
        if yolo_root.exists():
            try:
                shutil.rmtree(yolo_root)
                LOGGER.info(f"[cleanup] removed {yolo_root}")
            except Exception as e:
                LOGGER.warning(f"[cleanup] could not remove {yolo_root}: {e}")


if __name__ == "__main__":
    main()
