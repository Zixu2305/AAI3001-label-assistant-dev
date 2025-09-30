import csv, uuid, json
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from pycocotools import mask as maskUtils
from nuimages import NuImages

# ========= CONFIG =========
DATAROOT = "data/nuimages"
VERSIONS = ["v1.0-train", "v1.0-val"]
OUTDIR   = Path("crops")
MANIFEST = Path("manifest.csv")

# --- Size & policy knobs ---
MIN_SHORT_SIDE = 64
MIN_SHORT_SIDE_PER_CLASS = {}
CLASS_MAX_IMG_FRAC = {
    "barrier": 0.40, "bus": 0.80, "trailer": 0.80,
    "car": 0.60, "truck": 0.60, "construction_vehicle": 0.60,
    "pedestrian": 0.50, "motorcycle": 0.50, "bicycle": 0.50, "traffic_cone": 0.40
}
PER_IMAGE_CLASS_CAP = {"barrier": 3}

# --- Grouping knobs ---
GROUPABLE_CLASSES = {"traffic_cone", "barrier"}
ENABLE_GROUPING = True
GROUP_MIN_SIZE = 2
GROUP_IOU_THR = 0.05
GROUP_CENTER_DIST_PX = 60

# ========= CLASS MAPPING (23→10) =========
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

# ========= Geom utils =========
def iou_abs(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0)
    inter = iw*ih
    area_a = max(0, ax1-ax0)*max(0, ay1-ay0)
    area_b = max(0, bx1-bx0)*max(0, by1-by0)
    union = area_a + area_b - inter + 1e-6
    return inter/union

def ioa_abs(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0)
    inter = iw*ih
    area_a = max(0, ax1-ax0)*max(0, ay1-ay0)
    return inter / (area_a + 1e-6)

def nms_abs(boxes, iou_thr=0.7):
    if not boxes: return []
    areas = [max(0,(x1-x0))*max(0,(y1-y0)) for x0,y0,x1,y1 in boxes]
    order = np.argsort(areas)[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0]); keep.append(i)
        if order.size == 1: break
        rest = order[1:]
        ious = np.array([iou_abs(boxes[i], boxes[j]) for j in rest])
        order = rest[ious < iou_thr]
    return keep

# ========= Mask → tight box =========
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

def adaptive_margin(short_side):
    if short_side < 64:   return 0.00
    if short_side < 128:  return 0.04
    if short_side < 256:  return 0.08
    return 0.12

def clamp_box(x0,y0,x1,y1,W,H):
    return max(0,x0), max(0,y0), min(W,x1), min(H,y1)

def min_short_side_for(cls: str) -> int:
    return int(MIN_SHORT_SIDE_PER_CLASS.get(cls, MIN_SHORT_SIDE))

# ========= CropMeta =========
@dataclass
class CropMeta:
    cls: str
    img_path: str
    sd_token: str
    x0: int; y0: int; x1: int; y1: int
    W: int; H: int
    tightened: int
    is_group: int = 0
    group_id: str = ""
    group_size: int = 1
    group_x0: int = -1; group_y0: int = -1; group_x1: int = -1; group_y1: int = -1
    max_cross_iou: float = 0.0
    max_cross_iou_class: str = ""
    max_cross_ioa: float = 0.0

# ========= Crop policy =========
def make_crop(img, box_abs, cls, margin_scale=1.0):
    H, W = img.shape[:2]
    x0,y0,x1,y1 = map(int, box_abs)
    w, h = x1-x0, y1-y0
    mss = min_short_side_for(cls)
    if min(w,h) < mss: return None, None
    if (w*h) / float(W*H) > CLASS_MAX_IMG_FRAC.get(cls, 0.60): return None, None
    m = adaptive_margin(min(w,h)) * margin_scale
    if cls in {"pedestrian","bicycle","motorcycle"}: m = min(m, 0.05)
    dx, dy = int(m*w), int(m*h)
    X0, Y0, X1, Y1 = clamp_box(x0-dx, y0-dy, x1+dx, y1+dy, W, H)
    if min(X1-X0, Y1-Y0) < mss: return None, None
    crop = img[Y0:Y1, X0:X1]
    coverage = (w*h) / float(max(1,(X1-X0)*(Y1-Y0)))
    if coverage < 0.30:
        X0,Y0,X1,Y1 = clamp_box(x0, y0, x1, y1, W, H)
        if min(X1-X0, Y1-Y0) < mss: return None, None
        crop = img[Y0:Y1, X0:X1]
        coverage = (w*h) / float(max(1,(X1-X0)*(Y1-Y0)))
        if coverage < 0.30: return None, None
    ar = max((X1-X0)/(Y1-Y0+1e-6), (Y1-Y0)/(X1-X0+1e-6))
    if ar > 6.0: return None, None
    short_side = min(X1-X0, Y1-Y0)
    long_side  = max(X1-X0, Y1-Y0)
    img_frac   = ((X1-X0)*(Y1-Y0)) / float(W*H)
    return crop, {"crop_x0": X0, "crop_y0": Y0, "crop_x1": X1, "crop_y1": Y1,
                  "obj_w": w, "obj_h": h, "short_side": short_side, "long_side": long_side,
                  "coverage": coverage, "crop_img_frac": img_frac, "aspect_ratio": ar, "margin_used": float(m)}

# ========= Grouping =========
def _build_same_class_clusters(boxes, iou_thr=GROUP_IOU_THR, dist_px=GROUP_CENTER_DIST_PX):
    n = len(boxes)
    if n <= 1: return [[i] for i in range(n)]
    centers = np.array([[(b[0]+b[2])/2.0, (b[1]+b[3])/2.0] for b in boxes])
    adj = [[False]*n for _ in range(n)]
    for i in range(n): adj[i][i] = True
    for i in range(n):
        for j in range(i+1, n):
            iou = iou_abs(boxes[i], boxes[j])
            d = np.linalg.norm(centers[i] - centers[j])
            if iou >= iou_thr or d <= dist_px: adj[i][j] = adj[j][i] = True
    seen = [False]*n; clusters = []
    for i in range(n):
        if seen[i]: continue
        stack = [i]; seen[i] = True; comp = [i]
        while stack:
            u = stack.pop()
            for v in range(n):
                if not seen[v] and adj[u][v]: seen[v] = True; stack.append(v); comp.append(v)
        clusters.append(sorted(comp))
    return clusters

def _camera_name_from_filename(fn: str) -> str:
    seg = [p for p in fn.split('/') if p.startswith('CAM_')]
    return seg[0] if seg else ""

# ========= Main =========
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["version","camera","crop_path","class","img_path","sample_data_token",
                    "x0","y0","x1","y1","W","H","tightened",
                    "crop_x0","crop_y0","crop_x1","crop_y1","obj_w","obj_h",
                    "short_side","long_side","coverage","crop_img_frac","aspect_ratio","margin_used",
                    "is_group","group_id","group_size","group_x0","group_y0","group_x1","group_y1",
                    "max_cross_iou","max_cross_iou_class","max_cross_ioa"])
        for VERSION in VERSIONS:
            nuim = NuImages(dataroot=DATAROOT, version=VERSION, verbose=True)
            cat_by_token = {c["token"]: c for c in nuim.category}
            for sd in nuim.sample_data:
                if not sd.get("is_key_frame", False): continue
                if not sd["filename"].startswith("samples/"): continue
                if "/CAM_" not in sd["filename"]: continue
                img_rel = sd["filename"]; cam_name = _camera_name_from_filename(img_rel)
                img_path = Path(DATAROOT) / img_rel
                img = cv2.imread(str(img_path))
                if img is None: continue
                H, W = img.shape[:2]
                raw_anns = [a for a in nuim.object_ann if a["sample_data_token"] == sd["token"]]
                by_class = {}; tightened_map = {}
                for a in raw_anns:
                    cls = to_canonical(cat_by_token[a["category_token"]]["name"])
                    if not cls: continue
                    x0,y0,x1,y1 = map(float, a["bbox"]); tightened = 0
                    tb = tight_box_from_mask_json(a.get("mask"))
                    if tb is not None: x0,y0,x1,y1 = tb; tightened = 1
                    x0,y0,x1,y1 = clamp_box(int(x0),int(y0),int(x1),int(y1), W,H)
                    if x1<=x0 or y1<=y0: continue
                    w_box, h_box = x1-x0, y1-y0
                    if min(w_box,h_box) < min_short_side_for(cls): continue
                    if (w_box*h_box) / float(W*H) > CLASS_MAX_IMG_FRAC.get(cls, .60): continue
                    by_class.setdefault(cls, []).append((x0,y0,x1,y1))
                    tightened_map[(cls,(x0,y0,x1,y1))] = tightened
                kept_by_class = {}
                for cls, boxes in by_class.items():
                    if not boxes: continue
                    iou_thr = 0.6 if cls=="barrier" else 0.7
                    keep_idx = nms_abs(boxes, iou_thr=iou_thr)
                    kept = [boxes[i] for i in keep_idx]
                    cap = PER_IMAGE_CLASS_CAP.get(cls)
                    if cap is not None and len(kept) > cap:
                        kept = sorted(kept, key=lambda b:(b[2]-b[0])*(b[3]-b[1]), reverse=True)[:cap]
                    kept_by_class[cls] = kept
                flat = [(c,b) for c,bs in kept_by_class.items() for b in bs]
                max_iou_map = {}
                for i,(ci,bi) in enumerate(flat):
                    mx_iou, mx_cls, mx_ioa = 0.0, "", 0.0
                    for j,(cj,bj) in enumerate(flat):
                        if i==j or ci==cj: continue
                        iou = iou_abs(bi,bj)
                        if iou > mx_iou: mx_iou, mx_cls = iou, cj
                        mx_ioa = max(mx_ioa, ioa_abs(bi,bj))
                    max_iou_map[(ci,bi)] = (mx_iou, mx_cls, mx_ioa)
                # save singletons
                for cls, boxes in kept_by_class.items():
                    out_dir = OUTDIR / cls; out_dir.mkdir(parents=True, exist_ok=True)
                    for (x0,y0,x1,y1) in boxes:
                        crop, anal = make_crop(img, (x0,y0,x1,y1), cls)
                        if crop is None: continue
                        crop_path = out_dir / f"{uuid.uuid4().hex}.jpg"
                        cv2.imwrite(str(crop_path), crop)
                        mx_iou, mx_cls, mx_ioa = max_iou_map.get((cls,(x0,y0,x1,y1)), (0.0,"",0.0))
                        tight = tightened_map.get((cls,(x0,y0,x1,y1)), 0)
                        meta = CropMeta(
                            cls=cls, img_path=str(img_path), sd_token=sd["token"],
                            x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1),
                            W=H and W or W, H=H, tightened=int(tight),
                            is_group=0, group_id="", group_size=1,
                            group_x0=-1, group_y0=-1, group_x1=-1, group_y1=-1,
                            max_cross_iou=float(mx_iou), max_cross_iou_class=mx_cls, max_cross_ioa=float(mx_ioa)
                        )
                        w.writerow([VERSION, cam_name, str(crop_path), meta.cls, meta.img_path, meta.sd_token,
                                    meta.x0, meta.y0, meta.x1, meta.y1, meta.W, meta.H, meta.tightened,
                                    anal["crop_x0"],anal["crop_y0"],anal["crop_x1"],anal["crop_y1"],anal["obj_w"],anal["obj_h"],
                                    anal["short_side"],anal["long_side"], f"{anal['coverage']:.4f}", f"{anal['crop_img_frac']:.6f}", f"{anal['aspect_ratio']:.3f}", f"{anal['margin_used']:.3f}",
                                    meta.is_group, meta.group_id, meta.group_size, meta.group_x0, meta.group_y0, meta.group_x1, meta.group_y1,
                                    f"{meta.max_cross_iou:.4f}", meta.max_cross_iou_class, f"{meta.max_cross_ioa:.4f}"])
                # save groups
                if ENABLE_GROUPING:
                    for cls, boxes in kept_by_class.items():
                        if cls not in GROUPABLE_CLASSES or len(boxes) < GROUP_MIN_SIZE: continue
                        clusters = _build_same_class_clusters(boxes, GROUP_IOU_THR, GROUP_CENTER_DIST_PX)
                        for comp in clusters:
                            if len(comp) < GROUP_MIN_SIZE: continue
                            xs0 = [boxes[i][0] for i in comp]; ys0 = [boxes[i][1] for i in comp]
                            xs1 = [boxes[i][2] for i in comp]; ys1 = [boxes[i][3] for i in comp]
                            gx0,gy0,gx1,gy1 = int(min(xs0)), int(min(ys0)), int(max(xs1)), int(max(ys1))
                            crop, anal = make_crop(img, (gx0,gy0,gx1,gy1), cls)
                            if crop is None: continue
                            mx_iou, mx_cls, mx_ioa = 0.0, "", 0.0
                            for cj, bs in kept_by_class.items():
                                if cj == cls: continue
                                for bj in bs:
                                    iou = iou_abs((gx0,gy0,gx1,gy1), bj)
                                    if iou > mx_iou: mx_iou, mx_cls = iou, cj
                                    mx_ioa = max(mx_ioa, ioa_abs((gx0,gy0,gx1,gy1), bj))
                            out_dir = OUTDIR / cls; out_dir.mkdir(parents=True, exist_ok=True)
                            crop_path = out_dir / f"{uuid.uuid4().hex}.jpg"
                            cv2.imwrite(str(crop_path), crop)
                            group_id = uuid.uuid4().hex[:12]
                            meta = CropMeta(
                                cls=cls, img_path=str(img_path), sd_token=sd["token"],
                                x0=gx0, y0=gy0, x1=gx1, y1=gy1,
                                W=W, H=H, tightened=0,
                                is_group=1, group_id=group_id, group_size=len(comp),
                                group_x0=gx0, group_y0=gy0, group_x1=gx1, group_y1=gy1,
                                max_cross_iou=float(mx_iou), max_cross_iou_class=mx_cls, max_cross_ioa=float(mx_ioa)
                            )
                            w.writerow([VERSION, cam_name, str(crop_path), meta.cls, meta.img_path, meta.sd_token,
                                        meta.x0, meta.y0, meta.x1, meta.y1, meta.W, meta.H, meta.tightened,
                                        anal["crop_x0"],anal["crop_y0"],anal["crop_x1"],anal["crop_y1"],anal["obj_w"],anal["obj_h"],
                                        anal["short_side"],anal["long_side"], f"{anal['coverage']:.4f}", f"{anal['crop_img_frac']:.6f}", f"{anal['aspect_ratio']:.3f}", f"{anal['margin_used']:.3f}",
                                        meta.is_group, meta.group_id, meta.group_size, meta.group_x0, meta.group_y0, meta.group_x1, meta.group_y1,
                                        f"{meta.max_cross_iou:.4f}", meta.max_cross_iou_class, f"{meta.max_cross_ioa:.4f}"])
    print(f"Done. Manifest: {MANIFEST}")

if __name__ == "__main__":
    main()
