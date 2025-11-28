#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a larger exemplar bank (~500-600 images) with GLOBAL UNIQUENESS enforced,
optionally prioritizing images that already have classification crops.

Inputs (expected):
  meta/detection/manifest_images.csv
  meta/detection/manifest_det.csv

Optional input:
  meta/classification/manifest.csv    # to bias selection toward existing crops

Outputs:
  meta/al/exemplars_manifest.csv      # images selected for the bank
  meta/al/det_manifest_exemplar.csv   # reduced det manifest (objects within exemplar images)
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
ROOT = Path(".")  # run from repo root; change if needed

# Detection manifests (unchanged)
IMAGES_CSV = ROOT / "meta" / "detection" / "manifest_images.csv"
DETS_CSV   = ROOT / "meta" / "detection" / "manifest_det.csv"

# Optional classification crops manifest
CLASS_CROPS_CSV = ROOT / "meta" / "classification" / "manifest.csv"
USE_CLASSIFICATION_CROPS = True  # set False to ignore classification crops even if file exists

# Output
OUT_DIR    = ROOT / "meta" / "al"
EXEMPLAR_MANIFEST = OUT_DIR / "exemplars_manifest.csv"
DET_REDUCED       = OUT_DIR / "det_manifest_exemplar.csv"

# Target size knobs (bigger bank)
B_BASE_PER_CLASS = 20            # base images per class (was 12)
TAIL_GLOBAL = 300                # global tail add (was 180); with ~10 classes -> ≈500-600 unique images

# Size buckets for S/M/L balancing
SIZE_BUCKET_THRESH = (32*32, 96*96)  # S <= 32^2, M <= 96^2, else L
TAIL_BUCKET_RATIOS = {"S": 0.35, "M": 0.40, "L": 0.25}

# Viewpoint spread
VIEWPOINTS = [
    "CAM_FRONT", "CAM_BACK",
    "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]

# Crop-aware selection knobs
CROP_SCORE_BONUS = 2.0     # additive bonus to (n_objects) when an image has ANY crop
CROP_COUNT_WEIGHT = 0.05   # mild tie-breaker weight per crop (e.g., +0.05 per crop)
MIN_CROP_IMG_HINT = 8      # soft target: try to include at least ~8 crop-backed images per class (not a hard cap)
RANDOM_SEED = 1337         # deterministic tie-breaks

# ------------- helpers -------------
def pick(colnames: List[str], options: List[str]) -> str | None:
    for c in options:
        if c in colnames: return c
    return None

def ensure_img_path_column(df: pd.DataFrame, preferred_keys: List[str]) -> pd.DataFrame:
    """Return df with an 'img_path' column, renaming from the first existing key in preferred_keys."""
    if "img_path" in df.columns: return df
    for k in preferred_keys:
        if k in df.columns:
            return df.rename(columns={k: "img_path"})
    raise KeyError(f"None of these columns were found to serve as img_path: {preferred_keys}")

def size_bucket_from_area(a: float) -> str:
    s_thr, m_thr = SIZE_BUCKET_THRESH
    if a <= s_thr:  return "S"
    if a <= m_thr:  return "M"
    return "L"

def load_class_crops_map(csv_path: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load classification manifest (if present) and return:
      - DataFrame with ['img_path', 'crop_count'] (unique per image)
      - dict img_path -> crop_count
    Accepts flexible column names for image and crop path; class column optional.
    """
    dfc = pd.read_csv(csv_path)
    img_col = pick(dfc.columns, ["img_path","image_path","img","image","path","filepath","file"])
    crop_col = pick(dfc.columns, ["crop_path","crop","crop_filepath","crop_file","rel_crop_path","path","filepath","file"])
    if img_col is None or crop_col is None:
        # try common naming from your prior classification manifests
        raise KeyError(f"Could not locate image/crop columns in classification manifest: {csv_path}")
    small = dfc[[img_col, crop_col]].rename(columns={img_col:"img_path", crop_col:"crop_path"})
    # Count crops per image
    cc = small.groupby("img_path").size().rename("crop_count").reset_index()
    crop_map = {r["img_path"]: int(r["crop_count"]) for _, r in cc.iterrows()}
    return cc, crop_map

# ------------- load -------------
np.random.seed(RANDOM_SEED)

if not IMAGES_CSV.exists() or not DETS_CSV.exists():
    raise FileNotFoundError(f"Missing inputs. Expect:\n  {IMAGES_CSV}\n  {DETS_CSV}")

df_img = pd.read_csv(IMAGES_CSV)
df_det = pd.read_csv(DETS_CSV)

# Identify canonical columns (detection)
split_col   = pick(df_det.columns, ["split","set","subset"]) or "split"
class_col   = pick(df_det.columns, ["class_10","class","category","name","cls_name"]) or "class_10"
img_det_key = pick(df_det.columns, ["img_path","image_path","path","filepath","file"]) or "img_path"
img_img_key = pick(df_img.columns, ["img_path","image_path","path","filepath","file"]) or "img_path"
cam_key     = pick(df_img.columns, ["camera","sensor","cam","cam_name"]) or "camera"

# bbox presence
for k in ["x0","y0","x1","y1"]:
    if k not in df_det.columns:
        raise RuntimeError(f"Required bbox column missing in manifest_det: {k}")

# ------------- prep (train split, size) -------------
df_det = df_det.copy()
df_det["_w"] = (df_det["x1"] - df_det["x0"]).clip(lower=0)
df_det["_h"] = (df_det["y1"] - df_det["y0"]).clip(lower=0)
df_det["_area"] = (df_det["_w"] * df_det["_h"]).clip(lower=0)
df_det["_size_bucket"] = df_det["_area"].map(size_bucket_from_area)

df_img_tr = df_img[df_img[split_col] == "train"].copy()
df_det_tr = df_det[df_det[split_col] == "train"].copy()
if df_img_tr.empty or df_det_tr.empty:
    raise RuntimeError("Train split is empty in one of the manifests; please check your CSVs.")

# Standardize keys and ensure 'img_path'
df_img_tr = df_img_tr.rename(columns={cam_key: "camera"})
df_img_tr = ensure_img_path_column(df_img_tr, [img_img_key])

# Count classes
classes = sorted(df_det_tr[class_col].unique().tolist())
print(f"[info] Found {len(classes)} classes in train split.")

# ------------- optional classification crops map -------------
crop_map: Dict[str, int] = {}
crop_df = pd.DataFrame(columns=["img_path","crop_count"])
if USE_CLASSIFICATION_CROPS and CLASS_CROPS_CSV.exists():
    try:
        crop_df, crop_map = load_class_crops_map(CLASS_CROPS_CSV)
        print(f"[info] Loaded classification crops manifest: {CLASS_CROPS_CSV} "
              f"({len(crop_df)} images with crops)")
    except Exception as e:
        print(f"[warn] Could not use classification crops manifest ({e}). Proceeding without it.")
else:
    print("[info] No classification crops biasing (file missing or disabled).")

# ------------- per-image per-class stats -------------
det_small = df_det_tr[[class_col, img_det_key, "_size_bucket"]].copy()
det_small = ensure_img_path_column(det_small, [img_det_key])  # gives 'img_path'

img_small = df_img_tr[["img_path", "camera", split_col]].drop_duplicates()

cnt = (det_small
       .groupby([class_col, "img_path", "_size_bucket"]).size()
       .rename("n").reset_index())

cnt = cnt.merge(img_small[["img_path","camera"]], on="img_path", how="left")
cnt["camera"] = cnt["camera"].fillna("UNKNOWN")

# per (class, img): total objs & dominant size bucket
tot = (cnt.groupby([class_col, "img_path"])
       .agg(n_objects=("n","sum"))
       .reset_index())

dom = (cnt.groupby([class_col, "img_path", "_size_bucket"])["n"].sum()
       .reset_index()
       .sort_values([class_col, "img_path", "n"], ascending=[True, True, False]))
dom = dom.loc[dom.groupby([class_col, "img_path"])["n"].idxmax()][[class_col, "img_path", "_size_bucket"]]
dom = dom.rename(columns={"_size_bucket":"dom_size_bucket"})

tot = tot.merge(dom, on=[class_col, "img_path"], how="left")
tot = tot.merge(img_small[["img_path","camera"]], on="img_path", how="left")
tot["camera"] = tot["camera"].fillna("UNKNOWN")

# merge crop-count info (0 if none)
if not crop_df.empty:
    tot = tot.merge(crop_df, on="img_path", how="left")
    tot["crop_count"] = tot["crop_count"].fillna(0).astype(int)
else:
    tot["crop_count"] = 0

# A simple crop-aware score:
#   score = n_objects + CROP_SCORE_BONUS*(has_any_crop) + CROP_COUNT_WEIGHT*crop_count
tot["has_crop"] = (tot["crop_count"] > 0).astype(int)
tot["score"] = (tot["n_objects"].astype(float)
                + CROP_SCORE_BONUS * tot["has_crop"].astype(float)
                + CROP_COUNT_WEIGHT * tot["crop_count"].astype(float))

# Split per class
per_class: Dict[str, pd.DataFrame] = {c: tot[tot[class_col]==c].copy() for c in classes}

# Frequency per class (object counts)
freq = df_det_tr.groupby(class_col).size().rename("obj_count")

# ------------- selection with GLOBAL uniqueness -------------
rows = []
selected_global = set()
selected_per_class = {c: set() for c in classes}
bucket_rank = {"L":3,"M":2,"S":1}

# Stage A: Base picks — try 2 per viewpoint, then fill to B, crop-aware
for c in classes:
    dfc = per_class[c].copy()
    if dfc.empty:
        continue
    dfc["bucket_rank"] = dfc["dom_size_bucket"].map(bucket_rank).fillna(0)
    # Sort primarily by (score, bucket_rank), fallback to n_objects
    dfc = dfc.sort_values(["score","bucket_rank","n_objects"], ascending=[False, False, False])

    picked = 0
    picked_crop_imgs = 0

    # try per viewpoint (2 each), crop-aware order
    for vp in VIEWPOINTS:
        if picked >= B_BASE_PER_CLASS:
            break
        sub = dfc[(dfc["camera"]==vp) &
                  (~dfc["img_path"].isin(selected_per_class[c])) &
                  (~dfc["img_path"].isin(selected_global))]
        take = sub.head(2)
        for _, r in take.iterrows():
            rows.append({
                "class": c,
                "img_path": r["img_path"],
                "camera": r["camera"] if pd.notna(r["camera"]) else "UNKNOWN",
                "split": "train",
                "size_bucket": r["dom_size_bucket"],
                "n_objects_for_class": int(r["n_objects"]),
                "has_crop": int(r["has_crop"]),
                "crop_count": int(r["crop_count"]),
                "reason": f"base_viewpoint_{vp}"
            })
            selected_global.add(r["img_path"])
            selected_per_class[c].add(r["img_path"])
            picked += 1
            picked_crop_imgs += int(r["has_crop"])
            if picked >= B_BASE_PER_CLASS:
                break

    # fill to B (prefer crop-backed first if we're under MIN_CROP_IMG_HINT)
    if picked < B_BASE_PER_CLASS:
        remaining = dfc[(~dfc["img_path"].isin(selected_per_class[c])) &
                        (~dfc["img_path"].isin(selected_global))].copy()
        # Bias toward crop-backed if needed
        if picked_crop_imgs < MIN_CROP_IMG_HINT:
            remaining = remaining.sort_values(
                ["has_crop","score","n_objects"], ascending=[False, False, False]
            )
        else:
            remaining = remaining.sort_values(["score","n_objects"], ascending=[False, False])
        rem = remaining.head(B_BASE_PER_CLASS - picked)
        for _, r in rem.iterrows():
            rows.append({
                "class": c,
                "img_path": r["img_path"],
                "camera": r["camera"] if pd.notna(r["camera"]) else "UNKNOWN",
                "split": "train",
                "size_bucket": r["dom_size_bucket"],
                "n_objects_for_class": int(r["n_objects"]),
                "has_crop": int(r["has_crop"]),
                "crop_count": int(r["crop_count"]),
                "reason": "base_fill"
            })
            selected_global.add(r["img_path"])
            selected_per_class[c].add(r["img_path"])

# Stage B: Tail boost (+TAIL_GLOBAL globally by inverse sqrt freq)
fi = freq.reindex(classes).fillna(0).astype(float)
wi = 1.0 / np.sqrt(fi + 1.0)
wi = wi / wi.sum()
alloc = np.floor(TAIL_GLOBAL * wi).astype(int)
remainder = int(TAIL_GLOBAL - alloc.sum())
if remainder > 0:
    order = wi.sort_values(ascending=False).index.tolist()
    for k in range(remainder):
        alloc[order[k % len(order)]] += 1

for c in classes:
    q = int(alloc.get(c, 0))
    if q <= 0: continue

    dfc_remain = per_class[c][(~per_class[c]["img_path"].isin(selected_per_class[c])) &
                               (~per_class[c]["img_path"].isin(selected_global))].copy()
    if dfc_remain.empty: continue

    target_S = max(0, int(round(TAIL_BUCKET_RATIOS["S"] * q)))
    target_M = max(0, int(round(TAIL_BUCKET_RATIOS["M"] * q)))
    target_L = q - target_S - target_M
    desired = [("S", target_S), ("M", target_M), ("L", target_L)]

    for bucket, tgt in desired:
        if tgt <= 0: continue
        cand = dfc_remain[dfc_remain["dom_size_bucket"] == bucket].copy()
        if cand.empty: continue

        # prefer higher score; lightly prefer underused cameras for this class
        used_cams = pd.Series([r["camera"] for r in rows if r["class"]==c])
        counts_by_cam = (used_cams.value_counts() if not used_cams.empty else pd.Series(dtype=int))
        cand["cam_pen"] = cand["camera"].map(counts_by_cam).fillna(0).astype(int)

        # If we are below the crop hint, push crop-backed first
        already_crop = sum(int(r.get("has_crop",0)) for r in rows if r["class"]==c)
        if already_crop < MIN_CROP_IMG_HINT:
            cand = cand.sort_values(["has_crop","score","cam_pen"], ascending=[False, False, True])
        else:
            cand = cand.sort_values(["score","cam_pen"], ascending=[False, True])

        take = cand.head(tgt)
        for _, r in take.iterrows():
            ip = r["img_path"]
            if ip in selected_global: continue
            rows.append({
                "class": c,
                "img_path": ip,
                "camera": r["camera"] if pd.notna(r["camera"]) else "UNKNOWN",
                "split": "train",
                "size_bucket": r["dom_size_bucket"],
                "n_objects_for_class": int(r["n_objects"]),
                "has_crop": int(r["has_crop"]),
                "crop_count": int(r["crop_count"]),
                "reason": f"tail_size_{bucket}"
            })
            selected_global.add(ip)
            selected_per_class[c].add(ip)

        dfc_remain = dfc_remain[(~dfc_remain["img_path"].isin(selected_per_class[c])) &
                                (~dfc_remain["img_path"].isin(selected_global))]

    # fill any shortfall (score-first)
    current_q = sum(1 for r in rows if r["class"]==c and str(r["reason"]).startswith("tail_"))
    if current_q < q and not dfc_remain.empty:
        fill = dfc_remain.sort_values(["score","n_objects"], ascending=[False, False]).head(q - current_q)
        for _, r in fill.iterrows():
            ip = r["img_path"]
            if ip in selected_global: continue
            rows.append({
                "class": c,
                "img_path": ip,
                "camera": r["camera"] if pd.notna(r["camera"]) else "UNKNOWN",
                "split": "train",
                "size_bucket": r["dom_size_bucket"],
                "n_objects_for_class": int(r["n_objects"]),
                "has_crop": int(r["has_crop"]),
                "crop_count": int(r["crop_count"]),
                "reason": "tail_fill"
            })
            selected_global.add(ip)
            selected_per_class[c].add(ip)

manifest = pd.DataFrame(rows)

# Monitoring only: whether selected image has >= 2 classes in train dets
img_to_nclasses = df_det_tr.groupby(img_det_key)[class_col].nunique()
manifest["multi_class_image"] = manifest["img_path"].map(lambda p: int(img_to_nclasses.get(p, 0) >= 2))

# ------------- outputs -------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
manifest.to_csv(EXEMPLAR_MANIFEST, index=False)

# Reduced detection manifest: all objects whose image is in exemplar set
exemplar_imgs = set(manifest["img_path"].unique().tolist())
df_det_exemplar = df_det[df_det[img_det_key].isin(exemplar_imgs)].copy()
df_det_exemplar.to_csv(DET_REDUCED, index=False)

# ------------- report -------------
total_rows = len(manifest)
within_class_dup = total_rows - len(manifest.drop_duplicates(subset=["class","img_path"]))
global_unique = manifest["img_path"].nunique()
multi_class_reuse = (manifest.groupby("img_path")["class"].nunique() > 1).sum()
with_crop = int(manifest["has_crop"].sum()) if "has_crop" in manifest.columns else 0

print("\n=== Exemplar Build Summary ===")
print(f"Classes (train):             {len(classes)}")
print(f"Base per class (B):          {B_BASE_PER_CLASS}")
print(f"Tail boost (global):         {TAIL_GLOBAL}")
print(f"Selected rows (images):      {total_rows}")
print(f"Unique images (global):      {global_unique}")
print(f"Images with crops:           {with_crop}")
print(f"Duplicates within class:     {within_class_dup} (should be 0)")
print(f"Images used by >1 class:     {multi_class_reuse} (should be 0)")
print(f"Saved exemplar manifest:     {EXEMPLAR_MANIFEST}")
print(f"Saved reduced det manifest:  {DET_REDUCED}")
