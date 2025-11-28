# utils/al/build_inc_manifest.py
import csv, os, glob
from pathlib import Path

PROJ = Path(".")
OUT = PROJ/"meta/classification/manifest_inc.csv"

SRC_DIRS = [
    PROJ/"meta/al/exemplar_crops/crops",                                  # old classes (exemplar)
    PROJ/"meta/al/run/pull_traffic_light_20251121_142152/crops",          # TL batch (edit if new run)
]

def guess_img_path(crop_path: Path) -> str:
    # Try a few filename patterns to recover source image; else fall back to crop path
    stem = crop_path.stem
    # common patterns: <imgsha>__x_y_w_h.jpg  or  <imgname>__…  or hold 'img=' token
    for token in ["__img=", "__source="]:
        if token in stem:
            return stem.split(token,1)[1].split("__",1)[0]
    if "__" in stem:
        return stem.split("__",1)[0]
    return str(crop_path)  # fallback: group by crop itself

rows = []
for root in SRC_DIRS:
    for cls_dir in sorted((root).glob("*")):
        if not cls_dir.is_dir(): 
            continue
        cls = cls_dir.name
        for p in cls_dir.rglob("*"):
            if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".webp"]:
                crop_rel = p.as_posix().split(PROJ.as_posix()+"/",1)[-1]  # keep it project-relative
                rows.append({
                    "img_path": guess_img_path(p),
                    "crop_path": crop_rel,
                    "class": cls
                })

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["img_path","crop_path","class"])
    w.writeheader(); w.writerows(rows)

print(f"Written {OUT} with {len(rows)} rows.")
