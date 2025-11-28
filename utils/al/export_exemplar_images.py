#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copy all images referenced in meta/al/exemplars_manifest.csv into meta/al/exemplar/.

Rules
- If img_path is relative: copy from repo root into meta/al/exemplar/<relative/path>.
- If img_path is absolute: copy into meta/al/exemplar/__abs__/<absolute/path/without-leading-slash>.
- Creates meta/al/exemplar_index.csv listing src -> dest, status.
- Prints a compact summary at the end.

No CLI args; edit ROOT below if needed.
"""

from __future__ import annotations
import shutil
from pathlib import Path
import pandas as pd

# ---- Fixed locations (adjust if your repo layout differs) ----
ROOT = Path(".").resolve()
EXEMPLAR_MANIFEST = ROOT / "meta" / "al" / "exemplars_manifest.csv"
OUT_DIR = ROOT / "meta" / "al" / "exemplar"
INDEX_CSV = ROOT / "meta" / "al" / "exemplar_index.csv"

# Column in the manifest that has the image file path
IMG_COL_CANDIDATES = ["img_path", "image_path", "path", "filepath", "file"]

def pick(colnames, options):
    for c in options:
        if c in colnames:
            return c
    raise KeyError(f"Could not find any of {options} in exemplar manifest columns: {list(colnames)}")

def main():
    if not EXEMPLAR_MANIFEST.exists():
        raise FileNotFoundError(f"Missing manifest: {EXEMPLAR_MANIFEST}")

    df = pd.read_csv(EXEMPLAR_MANIFEST)
    img_col = pick(df.columns, IMG_COL_CANDIDATES)

    # unique image list
    imgs = df[img_col].dropna().astype(str).unique().tolist()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    copied = 0
    missing = 0
    skipped = 0

    for p_str in imgs:
        try:
            p = Path(p_str)
            # Resolve source path (relative to repo if not absolute)
            src = p if p.is_absolute() else (ROOT / p)
            # Build destination path
            if p.is_absolute():
                # strip leading slash and put under __abs__
                rel = Path(*p.parts[1:]) if len(p.parts) > 1 else Path(p.name)
                dest = OUT_DIR / "__abs__" / rel
            else:
                dest = OUT_DIR / p

            dest.parent.mkdir(parents=True, exist_ok=True)

            if not src.exists() or not src.is_file():
                rows.append({"src": str(src), "dest": str(dest), "status": "MISSING"})
                missing += 1
                continue

            # If already exists with same size, skip copying to save time
            if dest.exists() and dest.stat().st_size == src.stat().st_size:
                rows.append({"src": str(src), "dest": str(dest), "status": "EXISTS"})
                skipped += 1
                continue

            shutil.copy2(src, dest)
            rows.append({"src": str(src), "dest": str(dest), "status": "COPIED"})
            copied += 1

        except Exception as e:
            rows.append({"src": p_str, "dest": "", "status": f"ERROR: {e}"})
            missing += 1

    # Write index
    pd.DataFrame(rows).to_csv(INDEX_CSV, index=False)

    total = len(imgs)
    print("\n=== Exemplar Image Export ===")
    print(f"Manifest:           {EXEMPLAR_MANIFEST}")
    print(f"Output folder:      {OUT_DIR}")
    print(f"Images referenced:  {total}")
    print(f"Copied:             {copied}")
    print(f"Already existed:    {skipped}")
    print(f"Missing/Errors:     {missing}")
    print(f"Index CSV:          {INDEX_CSV}")

if __name__ == "__main__":
    main()
