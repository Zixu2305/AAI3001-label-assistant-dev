	#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export all CROPS that belong to images selected for the exemplar bank.

It cross-references:
  1) exemplars_manifest.csv  -> tells which IMAGES are in the bank
  2) crops manifest          -> tells where each CROP file lives and which image it came from

Usage:
  python export_exemplar_crops.py \
      --exemplar meta/al/exemplars_manifest.csv \
      --crops meta/detection/manifest.csv \
      --out meta/al/exemplar_crops

If you omit args, it tries sensible defaults.

Outputs:
  - Copies crops under --out, mirroring relative paths (absolute paths go under __abs__/)
  - Writes an index CSV (src, dest, status) next to --out
"""

from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import pandas as pd

# ---------- helpers ----------

IMG_SHA_CANDS = [
    "img_sha", "image_sha", "image_sha256", "sha256", "sha", "image_id", "img_key", "image_key"
]
IMG_PATH_CANDS = [
    "img_path", "image_path", "image", "img", "path", "filepath", "file"
]
CROP_PATH_CANDS = [
    "crop_path", "crop", "crop_file", "crop_filepath", "rel_crop_path", "path", "filepath", "file"
]

def pick_one(df: pd.DataFrame, candidates, must=False, who=""):
    for c in candidates:
        if c in df.columns:
            return c
    if must:
        raise KeyError(f"Could not find any of {candidates} in {who} columns: {list(df.columns)}")
    return None

def resolve_src(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else (root / p)

def dest_for(out_dir: Path, p: Path) -> Path:
    if p.is_absolute():
        # put under __abs__/path/without/leading/slash
        rel = Path(*p.parts[1:]) if len(p.parts) > 1 else Path(p.name)
        return out_dir / "__abs__" / rel
    else:
        return out_dir / p

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exemplar", type=Path, default=Path("meta/al/exemplars_manifest.csv"))
    ap.add_argument("--crops",    type=Path, default=None,
                    help="CSV with crop rows. If omitted, tries meta/detection/manifest.csv then meta/detection/manifest_det.csv")
    ap.add_argument("--out",      type=Path, default=Path("meta/al/exemplar_crops"))
    ap.add_argument("--repo_root",type=Path, default=Path("."),
                    help="Base folder to resolve relative paths from")
    args = ap.parse_args()

    root: Path = args.repo_root.resolve()
    exemplar_csv: Path = (root / args.exemplar).resolve()
    if args.crops is None:
        # try common locations
        for cand in ["meta/classification/manifest.csv", "meta/classification/manifest_det.csv", "manifest.csv"]:
            p = (root / cand)
            if p.exists():
                crops_csv = p.resolve()
                break
        else:
            raise FileNotFoundError("Could not auto-find a crops manifest (tried meta/classification/manifest.csv, "
                                    "meta/classification/manifest_det.csv, manifest.csv). Pass --crops explicitly.")
    else:
        crops_csv = (root / args.crops).resolve()

    out_dir: Path = (root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    index_csv: Path = out_dir.with_suffix("") / "index.csv"
    index_csv.parent.mkdir(parents=True, exist_ok=True)

    # --- load exemplar images
    if not exemplar_csv.exists():
        raise FileNotFoundError(f"Exemplar manifest not found: {exemplar_csv}")
    ex = pd.read_csv(exemplar_csv)

    ex_sha_col  = pick_one(ex, IMG_SHA_CANDS, must=False, who="exemplars_manifest")
    ex_path_col = pick_one(ex, IMG_PATH_CANDS, must=False, who="exemplars_manifest")
    if ex_sha_col is None and ex_path_col is None:
        raise KeyError("Need at least one of image SHA or image path column in exemplars manifest.")

    exemplar_sha_set  = set(ex[ex_sha_col].dropna().astype(str)) if ex_sha_col else set()
    exemplar_path_set = set(ex[ex_path_col].dropna().astype(str)) if ex_path_col else set()

    # --- load crops manifest
    if not crops_csv.exists():
        raise FileNotFoundError(f"Crops manifest not found: {crops_csv}")
    cm = pd.read_csv(crops_csv)

    cm_sha_col   = pick_one(cm, IMG_SHA_CANDS, must=False, who="crops manifest")
    cm_img_col   = pick_one(cm, IMG_PATH_CANDS, must=False, who="crops manifest")
    cm_crop_col  = pick_one(cm, CROP_PATH_CANDS, must=True,  who="crops manifest")

    # build mask: crop rows where the parent image is in exemplars
    mask = pd.Series([False] * len(cm))
    if cm_sha_col and exemplar_sha_set:
        mask = mask | cm[cm_sha_col].astype(str).isin(exemplar_sha_set)
    if cm_img_col and exemplar_path_set:
        mask = mask | cm[cm_img_col].astype(str).isin(exemplar_path_set)

    rows = []
    copied = skipped = missing = 0
    selected = cm[mask].copy()

    for _, r in selected.iterrows():
        crop_str = str(r[cm_crop_col]).strip()
        if not crop_str:
            continue
        src_p = resolve_src(root, Path(crop_str))
        dest_p = dest_for(out_dir, Path(crop_str))
        dest_p.parent.mkdir(parents=True, exist_ok=True)

        if not src_p.exists() or not src_p.is_file():
            rows.append({"src": str(src_p), "dest": str(dest_p), "status": "MISSING"})
            missing += 1
            continue

        # if dest exists with same size, skip copy
        if dest_p.exists() and dest_p.stat().st_size == src_p.stat().st_size:
            rows.append({"src": str(src_p), "dest": str(dest_p), "status": "EXISTS"})
            skipped += 1
            continue

        shutil.copy2(src_p, dest_p)
        rows.append({"src": str(src_p), "dest": str(dest_p), "status": "COPIED"})
        copied += 1

    pd.DataFrame(rows).to_csv(index_csv, index=False)

    print("\n=== Exemplar CROPS export ===")
    print(f"Repo root:          {root}")
    print(f"Exemplar manifest:  {exemplar_csv}")
    print(f"Crops manifest:     {crops_csv}")
    print(f"Output folder:      {out_dir}")
    print(f"Crop rows selected: {len(selected)}")
    print(f"Copied:             {copied}")
    print(f"Already existed:    {skipped}")
    print(f"Missing/Errors:     {missing}")
    print(f"Index CSV:          {index_csv}")

if __name__ == "__main__":
    main()
