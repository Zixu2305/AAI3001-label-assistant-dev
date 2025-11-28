#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract crops for a specific class from a pull_from_s3 detection manifest.

Each manifest row should contain:
  - "local_path": path to the source image (relative or absolute)
  - "anns": list of annotations, each with "class" and "bbox_xywh"

Usage example:
  python utils/al/extract_s3_crop.py \
    --manifest meta/al/run/pull_traffic_light_20251121_142152/compiled/detection/manifest.json \
    --class-name traffic_light

By default, crops are saved under <run_folder>/crops/<class_name>.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image


def clamp_bbox(x: float, y: float, w: float, h: float, img_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Clamp bbox (XYWH) to image bounds and convert to XYXY tuple."""
    width, height = img_size
    x1 = max(0, min(width, x))
    y1 = max(0, min(height, y))
    x2 = max(x1, min(width, x + w))
    y2 = max(y1, min(height, y + h))
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def iter_annotations(manifest_data: Iterable[dict], target_class: str) -> Iterable[Tuple[dict, dict]]:
    """Yield (image_entry, annotation) pairs for the specified class."""
    for entry in manifest_data:
        anns = entry.get("anns") or []
        for ann in anns:
            if target_class and ann.get("class") != target_class:
                continue
            bbox = ann.get("bbox_xywh")
            if not bbox or len(bbox) != 4:
                continue
            yield entry, ann


def resolve_path(root: Path, maybe_path: str) -> Path:
    """Resolve manifest paths relative to repo root."""
    p = Path(maybe_path)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def safe_dest_path(class_dir: Path, base_name: str) -> Path:
    """Generate a destination path for the crop image that avoids collisions."""
    dest = class_dir / f"{base_name}.jpg"
    idx = 1
    while dest.exists():
        dest = class_dir / f"{base_name}_{idx}.jpg"
        idx += 1
    return dest


def main():
    ap = argparse.ArgumentParser(description="Crop objects from detection manifest annotations.")
    ap.add_argument("--manifest", type=Path, required=True, help="Path to detection manifest JSON")
    ap.add_argument("--class-name", type=str, default="traffic_light", help="Annotation class to crop")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="Folder that will contain class-specific subfolders. "
                         "Defaults to <manifest run folder>/crops")
    ap.add_argument("--repo-root", type=Path, default=Path("."), help="Root used to resolve relative manifest paths")
    ap.add_argument("--overwrite", action="store_true", help="Recreate crops even if the file already exists")
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    manifest_path = (repo_root / args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if args.out_root is None:
        parents = list(manifest_path.parents)
        run_dir = manifest_path.parent
        if len(parents) >= 3:
            run_dir = manifest_path.parents[2]
        out_root = (run_dir / "crops").resolve()
    else:
        out_root = args.out_root
        if not out_root.is_absolute():
            out_root = (repo_root / out_root).resolve()
        else:
            out_root = out_root.resolve()

    class_dir = out_root / args.class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r") as fp:
        data = json.load(fp)

    processed = skipped = missing = invalid = 0

    for entry, ann in iter_annotations(data, args.class_name):
        local_path = entry.get("local_path")
        if not local_path:
            invalid += 1
            continue
        src_path = resolve_path(repo_root, local_path)
        if not src_path.exists():
            missing += 1
            continue

        bbox = ann["bbox_xywh"]
        x, y, w, h = [float(v) for v in bbox]
        if w <= 0 or h <= 0:
            invalid += 1
            continue

        uid = ann.get("uid")
        image_sha = entry.get("image_sha256") or entry.get("image_sha") or entry.get("img_sha")
        base_name_parts = []
        if image_sha:
            base_name_parts.append(str(image_sha))
        else:
            base_name_parts.append(src_path.stem)
        if uid:
            base_name_parts.append(str(uid))
        else:
            ann_idx = ann.get("idx")
            base_name_parts.append(str(ann_idx) if ann_idx is not None else f"{int(x)}_{int(y)}")
        base_name = "_".join(base_name_parts)

        dest_path = class_dir / f"{base_name}.jpg"
        if dest_path.exists() and not args.overwrite:
            skipped += 1
            continue
        if args.overwrite and dest_path.exists():
            dest_path.unlink()
        if not args.overwrite and dest_path.exists():
            # Should not happen because of earlier skip, but guard anyway.
            dest_path = safe_dest_path(class_dir, base_name)

        with Image.open(src_path) as im:
            bounds = clamp_bbox(x, y, w, h, im.size)
            if bounds[0] == bounds[2] or bounds[1] == bounds[3]:
                invalid += 1
                continue
            crop = im.crop(bounds)
            crop.save(dest_path, format="JPEG")
            processed += 1

    print("\n=== Detection Crop Export ===")
    print(f"Manifest:   {manifest_path}")
    print(f"Output dir: {class_dir}")
    print(f"Class:      {args.class_name}")
    print(f"Crops made: {processed}")
    print(f"Skipped:    {skipped}")
    print(f"Missing:    {missing}")
    print(f"Invalid:    {invalid}")


if __name__ == "__main__":
    main()
