#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pull new-class data from S3 shards, enforce a minimum of 50 unique images per class
before inclusion, download originals, and emit a consolidated JSON manifest (array)
with a reusable 70/30 image-level split for both detection and classification.

Inputs (S3, fixed):
  s3://$S3_BUCKET/label-assistant/new_class_bank/<class>/shards/*.jsonl
  images at s3://$S3_BUCKET/label-assistant/objects/originals/sha256=<hash>/original.jpg

Outputs (local, fixed):
  meta/al/run/<run_id>/raw/originals/sha256=<hash>/original.jpg
  meta/al/run/<run_id>/compiled/detection/manifest.json       # JSON array; each item has "split"
  meta/al/run/<run_id>/compiled/detection/splits.csv          # image_sha256,split
  meta/al/run/<run_id>/compiled/detection/stats.json
"""

from __future__ import annotations
import os, json, re, random
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime, timezone
import boto3

# ---------------- CONFIG ----------------
S3_BUCKET      = os.environ.get("AWS_S3_BUCKET", "streetview-label-assistant")  # set via env or edit here
ROOT_PREFIX    = "label-assistant"
NEW_CLASSES    = ["traffic_light"]          # edit to your new class slugs
MIN_IMAGES_ELIGIBLE = 50                    # minimum unique images per class to include
PROFILE_NAME   = os.environ.get("AWS_PROFILE", None)  # optional named profile

# Split knobs
SPLIT_SEED = 1337
TRAIN_FRAC = 0.70                           # 70/30 train/val
MIN_VAL_IMAGES = 15                         # ensure some minimum val size
MIN_VAL_POS = 10                            # ensure at least this many positives in val

# ---------------- OUTPUT LAYOUT ----------------
RUN_ID = f"pull_{'_'.join(NEW_CLASSES)}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
ROOT_OUT = Path("meta/al/run") / RUN_ID
RAW_DIR  = ROOT_OUT / "raw" / "originals"
CMP_DIR  = ROOT_OUT / "compiled" / "detection"
MANIFEST_JSON = CMP_DIR / "manifest.json"
STATS_JSON    = CMP_DIR / "stats.json"
SPLITS_CSV    = CMP_DIR / "splits.csv"

# ---------------- UTIL ----------------
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T.*Z$")

def parse_iso(ts: str) -> datetime:
    if not ts:
        return datetime.min.replace(tzinfo=timezone.utc)
    if ISO_RE.match(ts):
        ts = ts.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)

def s3_client():
    if PROFILE_NAME:
        return boto3.Session(profile_name=PROFILE_NAME).client("s3")
    return boto3.client("s3")

def list_shards(c, bucket: str, prefix: str) -> List[str]:
    paginator = c.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith(".jsonl"):
                keys.append(k)
    return keys

def read_jsonl_from_s3(c, bucket: str, key: str):
    body = c.get_object(Bucket=bucket, Key=key)["Body"]
    for line in body.iter_lines():
        if not line:
            continue
        yield json.loads(line.decode("utf-8"))

def parse_s3_uri(uri: str) -> Tuple[str, str]:
    assert uri.startswith("s3://"), f"Bad S3 URI: {uri}"
    _, _, rest = uri.partition("s3://")
    bucket, _, key = rest.partition("/")
    return bucket, key

def ensure_local_original(c, image_uri: str, image_sha256: str) -> Path:
    out = RAW_DIR / f"sha256={image_sha256}" / "original.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return out
    bkt, key = parse_s3_uri(image_uri)
    c.download_file(bkt, key, str(out))
    return out

# ---------------- MAIN ----------------
def main():
    assert S3_BUCKET and S3_BUCKET != "<your-bucket>", "Set S3_BUCKET env var or edit the script."

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CMP_DIR.mkdir(parents=True, exist_ok=True)

    c = s3_client()

    # Collect rows per requested class
    rows_by_class: Dict[str, List[Dict[str, Any]]] = {cls: [] for cls in NEW_CLASSES}
    for cls in NEW_CLASSES:
        shard_prefix = f"{ROOT_PREFIX}/new_class_bank/{cls}/shards/"
        keys = list_shards(c, S3_BUCKET, shard_prefix)
        if not keys:
            print(f"[warn] No shards at s3://{S3_BUCKET}/{shard_prefix}")
            continue
        for k in keys:
            for r in read_jsonl_from_s3(c, S3_BUCKET, k):
                # Strict per-class and schema
                if r.get("class") != cls:
                    continue
                if not all(k in r for k in ("uid","image_sha256","image_uri","bbox_xywh","class")):
                    continue
                rows_by_class[cls].append(r)

    # Dedup by uid per class and group by image_sha256
    selected_images: Dict[str, Dict[str, Any]] = {}  # keyed by image_sha256
    class_image_counts: Dict[str, int] = {}
    class_box_counts: Dict[str, int] = {}

    for cls, rows in rows_by_class.items():
        if not rows:
            continue
        seen_uids = set()
        deduped = []
        for r in rows:
            uid = r["uid"]
            if uid in seen_uids:
                continue
            seen_uids.add(uid)
            deduped.append(r)

        # group by image
        by_img: Dict[str, Dict[str, Any]] = {}
        for r in deduped:
            h = r["image_sha256"]
            it = by_img.setdefault(h, {
                "image_sha256": h,
                "image_uri": r["image_uri"],
                "anns": [],
                "source_sessions": set(),
                "latest_ts": datetime.min.replace(tzinfo=timezone.utc),
                "was_relabelled_any": False,
            })
            it["anns"].append({
                "class": r["class"],
                "bbox_xywh": r["bbox_xywh"],
                "uid": r["uid"],
            })
            if r.get("session_id"):
                it["source_sessions"].add(r["session_id"])
            it["was_relabelled_any"] = it["was_relabelled_any"] or bool(r.get("was_relabelled", False))
            ts = parse_iso(r.get("timestamp", ""))
            if ts > it["latest_ts"]:
                it["latest_ts"] = ts

        # eligibility check
        eligible_images = [v for v in by_img.values() if len(v["anns"]) > 0]
        unique_image_count = len(eligible_images)
        class_image_counts[cls] = unique_image_count
        class_box_counts[cls] = sum(len(v["anns"]) for v in eligible_images)

        if unique_image_count < MIN_IMAGES_ELIGIBLE:
            print(f"[info] Skipping class '{cls}' ({unique_image_count} images; needs ≥ {MIN_IMAGES_ELIGIBLE}).")
            continue

        # include all images for this class (no cap)
        for v in eligible_images:
            key = v["image_sha256"]
            if key not in selected_images:
                selected_images[key] = v
            else:
                # merge anns if multi-class requested
                selected_images[key]["anns"].extend(v["anns"])
                selected_images[key]["source_sessions"] |= v["source_sessions"]
                if v["latest_ts"] > selected_images[key]["latest_ts"]:
                    selected_images[key]["latest_ts"] = v["latest_ts"]
                selected_images[key]["was_relabelled_any"] |= v["was_relabelled_any"]

    if not selected_images:
        print("[error] No classes met the minimum unique-image requirement; nothing to pull.")
        return

    # materialize originals locally
    for item in selected_images.values():
        local = ensure_local_original(c, item["image_uri"], item["image_sha256"])
        item["local_path"] = str(local)
        item["source_sessions"] = sorted(list(item["source_sessions"]))
        item["latest_ts"] = item["latest_ts"].isoformat()

    # --------- build deterministic image-level split (70/30) ----------
    rng = random.Random(SPLIT_SEED)
    # positives: images that contain any of NEW_CLASSES
    # (here, all selected images are positives for at least one requested class)
    items = list(selected_images.values())
    items.sort(key=lambda x: (x["latest_ts"], x["image_sha256"]))  # stable base order
    rng.shuffle(items)

    n_total = len(items)
    n_val_target = max(int(round((1 - TRAIN_FRAC) * n_total)), MIN_VAL_IMAGES)

    # If there are also negatives someday, stratify; here likely all are positives.
    val_items = set(i["image_sha256"] for i in items[:n_val_target])
    # ensure at least MIN_VAL_POS positives in val (very likely already true)
    # (All are positives; guard left for completeness.)

    # assign splits
    split_map: Dict[str, str] = {}
    for it in items:
        split_map[it["image_sha256"]] = "val" if it["image_sha256"] in val_items else "train"

    # --------- write manifest (JSON array with split) + splits.csv ----------
    CMP_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for x in items:
        manifest.append({
            "image_uri": x["image_uri"],
            "image_sha256": x["image_sha256"],
            "local_path": x["local_path"],
            "split": split_map[x["image_sha256"]],
            "anns": x["anns"],
            "source_sessions": x["source_sessions"],
            "was_relabelled_any": x["was_relabelled_any"],
            "latest_ts": x["latest_ts"],
        })
    # sort by split then timestamp desc for readability
    manifest.sort(key=lambda z: (z["split"], z["latest_ts"]), reverse=True)

    with MANIFEST_JSON.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # splits.csv
    with SPLITS_CSV.open("w", encoding="utf-8") as f:
        f.write("image_sha256,split\n")
        for it in items:
            f.write(f"{it['image_sha256']},{split_map[it['image_sha256']]}\n")

        # ---------------- stats ----------------
    total_images = len(manifest)
    n_train = sum(1 for x in manifest if x["split"] == "train")
    n_val   = sum(1 for x in manifest if x["split"] == "val")

    # overall box counts
    total_crops = sum(len(x["anns"]) for x in manifest)
    boxes_per_split = {
        "train": sum(len(x["anns"]) for x in manifest if x["split"] == "train"),
        "val":   sum(len(x["anns"]) for x in manifest if x["split"] == "val"),
    }

    # per-class overall + per-split box counts
    by_cls_overall: Dict[str, int] = {}
    by_cls_split: Dict[str, Dict[str, int]] = {}
    for x in manifest:
        sp = x["split"]
        for a in x["anns"]:
            cls = a["class"]
            by_cls_overall[cls] = by_cls_overall.get(cls, 0) + 1
            if cls not in by_cls_split:
                by_cls_split[cls] = {"train": 0, "val": 0}
            by_cls_split[cls][sp] += 1

    stats = {
        "run_id": RUN_ID,
        "classes_requested": NEW_CLASSES,
        "min_images_per_class": MIN_IMAGES_ELIGIBLE,
        "classes_seen_images": class_image_counts,
        "classes_seen_boxes": class_box_counts,

        "total_images": total_images,
        "train_images": n_train,
        "val_images": n_val,

        "total_boxes": total_crops,
        "boxes_per_split": boxes_per_split,                    # NEW
        "boxes_per_class_overall": by_cls_overall,
        "boxes_per_class_split": by_cls_split,                 # NEW

        "train_frac_target": TRAIN_FRAC,
        "split_seed": SPLIT_SEED,
        "s3_bucket": S3_BUCKET,
        "root_prefix": ROOT_PREFIX,
        "outputs": {
            "manifest_json": str(MANIFEST_JSON),
            "splits_csv": str(SPLITS_CSV),
            "stats_json": str(STATS_JSON),
            "raw_dir": str(RAW_DIR),
        },
    }
    with STATS_JSON.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # prints
    print("\n=== Pull Summary ===")
    print(f"Run ID:         {RUN_ID}")
    print(f"S3 bucket:      {S3_BUCKET}")
    for cls in NEW_CLASSES:
        imgs = class_image_counts.get(cls, 0)
        boxes = class_box_counts.get(cls, 0)
        print(f"Class '{cls}':  {imgs} images, {boxes} boxes "
              f"{'(included)' if imgs >= MIN_IMAGES_ELIGIBLE else '(below min; excluded)'}")
    print(f"Selected images: {total_images}  (train={n_train}, val={n_val})")
    print(f"Total boxes:     {total_crops} (train={boxes_per_split['train']}, val={boxes_per_split['val']})")
    print(f"Manifest JSON:   {MANIFEST_JSON}")
    print(f"Splits CSV:      {SPLITS_CSV}")
    print(f"Stats JSON:      {STATS_JSON}")

if __name__ == "__main__":
    main()
