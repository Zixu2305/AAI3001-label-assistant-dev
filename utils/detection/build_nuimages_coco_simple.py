# build_nuimages_coco_from_cropper.py  — patched & faster
# ====== EDIT THESE ONLY ======
DATAROOT = "data/nuimages"                 # contains 'samples/' and 'v1.0-{train,val,test}/'
VERSIONS = ["v1.0-train", "v1.0-val", "v1.0-test"]
OUTDIR   = "meta/dataset"                   # output root
COCO_STORE_ABS_PATHS = False               # True => absolute file paths in COCO
USE_MASK_TIGHTEN = False                   # True => decode masks to tighten boxes (slower)
# ======================================

import json, csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from pycocotools import mask as maskUtils
from nuimages import NuImages

# ---------- Canonical classes (IDs = index) ----------
CANONICAL = [
    "car","truck","bus","trailer","construction_vehicle",
    "pedestrian","motorcycle","bicycle","traffic_cone","barrier"
]
CANON2ID = {n:i for i,n in enumerate(CANONICAL)}

# ---------- Mapping 23 → 10 (same rules as cropper) ----------
def to_canonical(name: str):
    if name.startswith("vehicle.car"): return "car"
    if name.startswith("vehicle.truck"): return "truck"
    if name.startswith("vehicle.bus"): return "bus"
    if name.startswith("vehicle.trailer"): return "trailer"
    if name.startswith("vehicle.construction"): return "construction_vehicle"
    if name.startswith("human.pedestrian"): return "pedestrian"
    if name.startswith("vehicle.motorcycle"): return "motorcycle"
    if name.startswith("vehicle.bicycle"): return "bicycle"
    if name.startswith("movable_object.trafficcone"): return "traffic_cone"
    if name.startswith("movable_object.barrier"): return "barrier"
    return None

# ---------- Helpers ----------
def _camera_name_from_filename(fn: str) -> str:
    seg = [p for p in fn.split('/') if p.startswith('CAM_')]
    return seg[0] if seg else ""

def clamp_box(x0,y0,x1,y1,W,H):
    x0 = max(0, min(int(x0), W-1))
    y0 = max(0, min(int(y0), H-1))
    x1 = max(0, min(int(x1), W-1))
    y1 = max(0, min(int(y1), H-1))
    if x1 <= x0: x1 = min(W-1, x0+1)
    if y1 <= y0: y1 = min(H-1, y0+1)
    return x0, y0, x1, y1

def tight_box_from_mask_json(mask_json_str):
    if not mask_json_str: return None
    try:
        rle = json.loads(mask_json_str)
        m = maskUtils.decode(rle)
        if m.ndim == 3: m = m[:, :, 0]
        ys, xs = np.where(m > 0)
        if xs.size == 0: return None
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        return (x0, y0, x1, y1)
    except Exception:
        return None

# ---------- IO ----------
OUT = Path(OUTDIR)
(OUT / "annotations").mkdir(parents=True, exist_ok=True)

# manifests
det_fields  = ["split","version","camera","img_path","W","H","image_id","sd_token",
               "class_10","class_id","x0","y0","x1","y1","area","tightened"]
img_fields  = ["split","version","image_id","sd_token","img_path","camera","W","H",
               "n_objects","classes_present"]
det_rows = []
img_rows = []
image_row_index = {}   # sd_token -> index into img_rows (fixes n_objects=0 issue)

UNMAPPED_COUNTS = defaultdict(int)

for VERSION in VERSIONS:
    split = "train" if "train" in VERSION else ("val" if "val" in VERSION else "test")
    print(f"[{VERSION}] indexing…")
    nuim = NuImages(dataroot=DATAROOT, version=VERSION, verbose=False)

    # category token -> name
    cat_by_token = {c["token"]: c["name"] for c in nuim.category}

    # camera keyframes only
    cam_sd = [sd for sd in nuim.sample_data
              if sd.get("is_key_frame", False)
              and sd["filename"].startswith("samples/")
              and "/CAM_" in sd["filename"]]
    token2sd = {sd["token"]: sd for sd in cam_sd}

    # COCO skeleton
    coco = {
        "info": {"description": f"NuImages {VERSION} (canonical={len(CANONICAL)} classes)"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": CANON2ID[n], "name": n} for n in CANONICAL],
    }

    # register images (use sizes directly from sample_data.json; no disk I/O)
    token2imgid = {}
    imgid = 1
    for sd in cam_sd:
        rel = sd["filename"]
        cam_name = _camera_name_from_filename(rel)
        W = int(sd["width"]); H = int(sd["height"])
        img_path = (Path(DATAROOT) / rel).as_posix()

        coco["images"].append({
            "id": imgid,
            "file_name": (img_path if COCO_STORE_ABS_PATHS else rel),
            "width": W, "height": H
        })
        token2imgid[sd["token"]] = imgid

        # image-level manifest stub
        img_rows.append({
            "split": split, "version": VERSION, "image_id": imgid,
            "sd_token": sd["token"],
            "img_path": img_path,
            "camera": cam_name, "W": W, "H": H,
            "n_objects": 0, "classes_present": ""
        })
        image_row_index[sd["token"]] = len(img_rows) - 1  # <-- robust index by sd_token
        imgid += 1

    # index object anns by sample_data_token (if present)
    sd_to_anns = defaultdict(list)
    if hasattr(nuim, "object_ann") and nuim.object_ann:
        for a in nuim.object_ann:
            tok = a.get("sample_data_token")
            if tok in token2imgid:
                sd_to_anns[tok].append(a)

    # per-image pass: map exactly as cropper does
    annid = 1
    for sd in cam_sd:
        tok = sd["token"]
        if tok not in token2imgid:
            continue
        img_id = token2imgid[tok]

        # reuse known props; no image read
        W = int(sd["width"]); H = int(sd["height"])
        img_path = (Path(DATAROOT) / sd["filename"]).as_posix()
        cam_name = _camera_name_from_filename(sd["filename"])

        anns = sd_to_anns.get(tok, [])
        classes_in_image = []
        n_for_image = 0

        for a in anns:
            raw_name = cat_by_token.get(a["category_token"])
            canon = to_canonical(raw_name) if raw_name else None
            if not canon:
                UNMAPPED_COUNTS[raw_name or "<missing>"] += 1
                continue

            # bbox as xyxy (cropper does this)
            x0, y0, x1, y1 = map(float, a["bbox"])

            # optional: tighten to mask
            if USE_MASK_TIGHTEN:
                tb = tight_box_from_mask_json(a.get("mask"))
                if tb is not None:
                    x0, y0, x1, y1 = tb
                    tightened = 1
                else:
                    tightened = 0
            else:
                tightened = 0

            x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, W, H)
            if x1 <= x0 or y1 <= y0:
                continue

            w_box = x1 - x0
            h_box = y1 - y0
            area = float(w_box * h_box)

            # COCO requires xywh; manifest keeps corners
            coco["annotations"].append({
                "id": annid,
                "image_id": img_id,
                "category_id": CANON2ID[canon],
                "bbox": [float(x0), float(y0), float(w_box), float(h_box)],
                "area": area,
                "iscrowd": 0
            })
            annid += 1

            det_rows.append({
                "split": split, "version": VERSION, "camera": cam_name,
                "img_path": img_path, "W": W, "H": H, "image_id": img_id,
                "sd_token": tok, "class_10": canon, "class_id": CANON2ID[canon],
                "x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1),
                "area": area, "tightened": tightened
            })

            classes_in_image.append(canon)
            n_for_image += 1

        # finalize image row — use the index by sd_token (so split restarts don’t break us)
        row_idx = image_row_index[tok]
        img_rows[row_idx]["n_objects"] = n_for_image
        img_rows[row_idx]["classes_present"] = ";".join(sorted(set(classes_in_image)))

    # write COCO for this split
    out_json = OUT / "annotations" / f"instances_{split}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(coco, f)
    print(f"[{VERSION}] images={len(coco['images'])} anns={len(coco['annotations'])}")

# write manifests
with (OUT / "manifest_det.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=det_fields)
    w.writeheader()
    for r in det_rows:
        w.writerow(r)

with (OUT / "manifest_images.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=img_fields)
    w.writeheader()
    for r in img_rows:
        w.writerow(r)

# unmapped raw categories (just to know what was skipped)
rep = OUT / "unmapped_categories.csv"
with rep.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["raw_category","count"])
    for k, v in sorted(UNMAPPED_COUNTS.items(), key=lambda kv: -kv[1]):
        w.writerow([k, v])

print(f"[OK] wrote:\n - {OUT/'annotations'}\n - {OUT/'manifest_det.csv'}\n - {OUT/'manifest_images.csv'}\n - {rep}")
