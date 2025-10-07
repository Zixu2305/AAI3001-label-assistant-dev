import os, json, argparse, csv, re, random
from datetime import datetime
import shutil, platform
from typing import List, Dict, Tuple, Optional, Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Globals & tiny utils
# ------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for closer reproducibility; can reduce throughput a bit:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _seed_worker(worker_id):
    # each worker gets a different seed derived from the base generator
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def timestamp(): return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(d: str):
    assert isinstance(d, str) and d.strip(), f"Invalid directory path: {d!r}"
    os.makedirs(d, exist_ok=True)

def resolve_paths(cfg: dict) -> dict:
    p = cfg["paths"].copy()
    for _ in range(3):
        for k, v in list(p.items()):
            if isinstance(v, str):
                try:
                    p[k] = v.format(**p)
                except KeyError:
                    pass
    for k, v in list(p.items()):
        if isinstance(v, str):
            v = os.path.expanduser(v)
            if k != "proj_root" and not os.path.isabs(v) and "proj_root" in p:
                v = os.path.join(p["proj_root"], v)
            p[k] = os.path.abspath(v)
    for key in ["proj_root","runs_dir","meta_dir","crops_dir","manifest","classes"]:
        assert p.get(key), f"paths.{key} missing after resolution"
    cfg["paths"] = p
    return cfg

def csv_logger_init(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow([
                "stage","epoch","cum_epoch",
                "train_loss","train_acc","train_macro_f1",
                "val_loss","val_acc","val_macro_f1","lr"
            ])
    return path

def csv_logger_write(path, row: dict):
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([row.get(k,"") for k in [
            "stage","epoch","cum_epoch",
            "train_loss","train_acc","train_macro_f1",
            "val_loss","val_acc","val_macro_f1","lr"
        ]])

# ------------------------------------------------------------
# classes.json loader (robust)
# ------------------------------------------------------------
def _parse_classes(raw):
    """
    Supports:
      A) {"version":"...", "classes":[{"id":0,"name":"car"}, ...], "index_order":[...]}
      B) [{"id":0,"name":"car"}, ...]
      C) {"car":0, "truck":1, ...}
      D) ["car","truck", ...]
    Returns: name_to_id, id_to_name, normalized list[{"id","name"}] (ids 0..C-1)
    """
    # A) wrapped
    if isinstance(raw, dict) and "classes" in raw and isinstance(raw["classes"], list):
        items = []
        for i, c in enumerate(raw["classes"]):
            cid = int(c["id"]) if "id" in c else i
            items.append({"id": cid, "name": str(c["name"])})
        if "index_order" in raw and isinstance(raw["index_order"], list):
            order = list(map(str, raw["index_order"]))
            name2new = {n:i for i,n in enumerate(order)}
            items = [{"id": name2new[it["name"]], "name": it["name"]}
                     for it in items if it["name"] in name2new]
        items = sorted(items, key=lambda x: x["id"])
    # B) list of dicts
    elif isinstance(raw, list) and raw and isinstance(raw[0], dict) and "name" in raw[0]:
        items = [{"id": int(c["id"]) if "id" in c else i, "name": str(c["name"])}
                 for i,c in enumerate(raw)]
        items = sorted(items, key=lambda x: x["id"])
    # C) dict name->id
    elif isinstance(raw, dict):
        items = sorted([{"id": int(v), "name": str(k)} for k,v in raw.items()],
                       key=lambda x: x["id"])
    # D) list of names
    elif isinstance(raw, list):
        items = [{"id": i, "name": str(n)} for i,n in enumerate(raw)]
    else:
        raise ValueError(f"Unsupported classes.json structure: {type(raw)}")

    ids = [it["id"] for it in items]
    if ids != list(range(len(items))):
        remap = {old:i for i,old in enumerate(sorted(set(ids)))}
        items = sorted([{"id": remap[it["id"]], "name": it["name"]} for it in items],
                       key=lambda x: x["id"])

    name_to_id = {it["name"]: it["id"] for it in items}
    id_to_name = {it["id"]: it["name"] for it in items}
    return name_to_id, id_to_name, items

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------
class Letterbox224:
    def __init__(self, size=224, fill=0):
        self.size=size; self.fill=fill
    def __call__(self, img: Image.Image):
        w,h = img.size
        if w==0 or h==0:
            return Image.new("RGB",(self.size,self.size),self.fill)
        s = min(self.size/w, self.size/h)
        nw, nh = max(1,int(round(w*s))), max(1,int(round(h*s)))
        img = img.resize((nw,nh), Image.BICUBIC)
        canvas = Image.new("RGB",(self.size,self.size),self.fill)
        left = (self.size-nw)//2; top=(self.size-nh)//2
        canvas.paste(img,(left,top))
        return canvas

def attach_abs_paths(df: pd.DataFrame, crops_root: str) -> pd.DataFrame:
    """Notebook-consistent: resolve everything inside <crops_root> unless absolute."""
    def resolve(p):
        p = str(p).strip().lstrip("./")
        if os.path.isabs(p): return p
        if p.startswith("crops/"):
            return os.path.join(crops_root, p.split("crops/",1)[1])
        return os.path.join(crops_root, p)
    out = df.copy()
    out["__abs_path__"] = out["crop_path"].apply(resolve)
    mask = out["__abs_path__"].apply(os.path.exists)
    missing = int((~mask).sum())
    if missing:
        print(f"[warn] {missing} / {len(out)} crop files are missing; skipped.")
        print(out.loc[~mask, ["crop_path","__abs_path__"]].head(5).to_string(index=False))
    kept = out[mask].reset_index(drop=True)
    if len(kept)==0:
        raise RuntimeError("After resolving paths, no files exist. Check paths.crops_dir and 'crop_path'.")
    return kept

def stratified_group_split(df: pd.DataFrame, img_col="img_path", class_col="class",
                           train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert abs(train_ratio+val_ratio+test_ratio-1.0) < 1e-6
    img_tab = (df.groupby([img_col,class_col]).size()
                 .reset_index(name="n")
                 .sort_values([img_col,"n"], ascending=[True,False])
                 .drop_duplicates(img_col))
    imgs = img_tab[img_col].values
    yimg = img_tab[class_col].values
    from sklearn.model_selection import train_test_split
    def can_strat(y): return (pd.Series(y).value_counts()>=2).all()
    strat = yimg if can_strat(yimg) else None
    imgs_tr, imgs_tmp, _, y_tmp = train_test_split(
        imgs, yimg, train_size=train_ratio, random_state=seed, stratify=strat)
    strat2 = y_tmp if can_strat(y_tmp) else None
    val_within = val_ratio/(val_ratio+test_ratio)
    imgs_va, imgs_te = train_test_split(
        imgs_tmp, train_size=val_within, random_state=seed, stratify=strat2)[0:2]
    s_tr, s_va, s_te = set(imgs_tr), set(imgs_va), set(imgs_te)
    tr = df[df[img_col].isin(s_tr)].reset_index(drop=True)
    va = df[df[img_col].isin(s_va)].reset_index(drop=True)
    te = df[df[img_col].isin(s_te)].reset_index(drop=True)
    a,b,c = set(tr[img_col]), set(va[img_col]), set(te[img_col])
    assert len(a&b)==0 and len(a&c)==0 and len(b&c)==0
    return tr, va, te

class CropsDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, name_to_id: Dict[str,int], transform):
        self.df = frame.reset_index(drop=True)
        self.map = name_to_id
        self.t = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        y = self.map[r["class"]]
        try:
            with Image.open(r["__abs_path__"]) as im:
                im = im.convert("RGB")
                x = self.t(im)
        except Exception:
            return None
        return x, y

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    x,y = zip(*batch)
    return torch.stack(x,0), torch.tensor(y)

def build_transforms(cfg):
    s = cfg["preprocess"]["image_size"]
    letter = Letterbox224(s) if cfg["preprocess"].get("letterbox",True) else transforms.Resize((s,s))
    train_tfms = [
        letter,
        transforms.RandomHorizontalFlip(p=cfg["augment"].get("hflip_p",0.5)),
        transforms.RandomVerticalFlip(p=cfg["augment"].get("vflip_p",0.5)),
        transforms.RandomRotation(degrees=cfg["augment"].get("rotation_deg",15)),
        transforms.RandomAffine(degrees=0, scale=(cfg["augment"].get("scale_min",0.9),
                                                  cfg["augment"].get("scale_max",1.1))),
        transforms.ToTensor(),
    ]
    eval_tfms = [letter, transforms.ToTensor()]
    if cfg["preprocess"].get("normalize","imagenet") == "imagenet":
        norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        train_tfms.append(norm); eval_tfms.append(norm)
    return transforms.Compose(train_tfms), transforms.Compose(eval_tfms)

def build_loaders(cfg, train_df, val_df, test_df, name_to_id, generator=None):
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        assert len(df)>0, f"{name} split empty after filtering/path resolution."
    train_tfms, eval_tfms = build_transforms(cfg)
    train_ds = CropsDataset(train_df, name_to_id, train_tfms)
    val_ds   = CropsDataset(val_df,   name_to_id, eval_tfms)
    test_ds  = CropsDataset(test_df,  name_to_id, eval_tfms)
    dlc = cfg["dataloader"]
    common = dict(
        num_workers=dlc["num_workers"],
        pin_memory=dlc["pin_memory"],
        persistent_workers=dlc.get("persistent_workers", True),
        prefetch_factor=dlc.get("prefetch_factor", 2),
        collate_fn=collate_skip_none,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    train_loader = DataLoader(train_ds, batch_size=dlc["batch_size"], shuffle=True,
                              drop_last=dlc.get("drop_last", True), **common)
    val_loader   = DataLoader(val_ds,   batch_size=dlc["batch_size"], shuffle=False,
                              drop_last=False, **common)
    test_loader  = DataLoader(test_ds,  batch_size=dlc["batch_size"], shuffle=False,
                              drop_last=False, **common)
    return train_loader, val_loader, test_loader

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
def infer_last_block_patterns(model, arch: str, last_k: int = 1):
    """
    Returns (patterns:list[str], head_name:str) for Stage-B:
      - Always includes the classifier head
      - Plus ONLY the last 'k' backbone block(s) for the given arch
    Torchvision backbones supported in this repo:
      resnet50            -> layer4.<last_idx> (+ fc)
      convnext_tiny       -> stages.3 (if present) else features.<last_idx> (+ classifier)
      efficientnet_v2_s   -> features.<last_idx> (+ classifier)
      mobilenet_v3_large  -> features.<last_idx> (+ classifier)
    """
    head_name = "fc" if arch.startswith("resnet") else "classifier"
    names = [n for (n, _) in model.named_parameters()]

    def last_indices(prefix_regex):
        rx = re.compile(prefix_regex)
        idx = []
        for n in names:
            m = rx.match(n)
            if m:
                try:
                    idx.append(int(m.group(1)))
                except Exception:
                    pass
        idx = sorted(set(idx))
        return idx[-last_k:] if idx else []

    pats = []
    if arch.startswith("resnet"):
        # e.g., layer4.2 is the last bottleneck in resnet50
        idxs = last_indices(r"^layer4\.(\d+)\.")
        pats.extend([fr"^layer4\.{i}\b" for i in idxs] or [r"^layer4\b"])
    elif arch.startswith("convnext"):
        # Prefer explicit last stage if surfaced as stages.3; fallback to last features.N
        if any(n.startswith("stages.3") for n in names):
            pats.append(r"^stages\.3\b")
        else:
            idxs = last_indices(r"^features\.(\d+)\.")
            pats.extend([fr"^features\.{i}\b" for i in idxs] or [r"^features\b"])
    elif arch.startswith("efficientnet_v2") or arch.startswith("mobilenet_v3"):
        idxs = last_indices(r"^features\.(\d+)\.")
        pats.extend([fr"^features\.{i}\b" for i in idxs] or [r"^features\b"])
    else:
        raise ValueError(f"Unsupported arch for auto last-block inference: {arch}")

    pats.append(head_name)  # always include the classifier head
    return pats, head_name

def get_feature_dim(model, arch: str) -> int:
    if arch.startswith("resnet"):          return model.fc.in_features
    if arch.startswith("convnext"):        return model.classifier[2].in_features
    if arch.startswith("efficientnet_v2"): return model.classifier[1].in_features
    if arch.startswith("mobilenet_v3"):    return model.classifier[0].in_features
    raise ValueError(f"Unsupported arch: {arch}")

def attach_head(model, arch: str, head: nn.Module) -> nn.Module:
    if arch.startswith("resnet"):
        model.fc = head; return model
    if arch.startswith("convnext"):
        ln   = model.classifier[0]   # LayerNorm
        flat = model.classifier[1]   # Flatten(1)
        model.classifier = nn.Sequential(ln, flat, head)
        return model
    if arch.startswith("efficientnet_v2"):
        model.classifier = nn.Sequential(nn.Dropout(p=0.0, inplace=False), head); return model
    if arch.startswith("mobilenet_v3"):
        model.classifier = head; return model
    raise ValueError(f"Unsupported arch: {arch}")

def make_head(in_dim: int, num_classes: int, hidden: List[int], dropout: float, norm: str):
    layers = []
    last = in_dim
    for h in hidden:
        layers.append(nn.Linear(last, h))
        if norm == "batchnorm": layers.append(nn.BatchNorm1d(h))
        elif norm == "layernorm": layers.append(nn.LayerNorm(h))
        layers += [nn.ReLU(inplace=True), nn.Dropout(dropout)]
        last = h
    layers.append(nn.Linear(last, num_classes))
    return nn.Sequential(*layers)

def build_model(cfg, num_classes):
    arch = cfg["model"]["arch"]; pretrained = cfg["model"].get("pretrained", True)
    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    elif arch == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
    elif arch == "convnext_small":
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None)
    elif arch == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    in_dim = get_feature_dim(model, arch)
    head = make_head(in_dim, num_classes,
                     cfg["model"]["head_hidden"],
                     cfg["model"]["head_dropout"],
                     cfg["model"]["head_norm"])
    return attach_head(model, arch, head)

def set_trainable(module, flag: bool):
    for p in module.parameters(): p.requires_grad = flag

def freeze_all_but_head(model, arch: str):
    set_trainable(model, False)
    if arch.startswith("resnet"): set_trainable(model.fc, True)
    elif arch.startswith("convnext"): set_trainable(model.classifier, True)
    elif arch.startswith("efficientnet_v2"): set_trainable(model.classifier, True)
    elif arch.startswith("mobilenet_v3"): set_trainable(model.classifier, True)

def set_backbone_eval_head_train(model, arch: str):
    """Stage-A BN fix: backbone .eval() (freeze BN stats), head .train() so its BN learns."""
    model.train()
    if arch.startswith("resnet"):
        model.conv1.eval(); model.bn1.eval()
        model.layer1.eval(); model.layer2.eval(); model.layer3.eval(); model.layer4.eval()
        model.fc.train()
    else:
        if hasattr(model, "features"):   model.features.eval()
        if hasattr(model, "classifier"): model.classifier.train()

def param_groups_by_pattern(model, patterns: List[Dict[str, float]]):
    named = list(model.named_parameters())
    groups, used = [], set()
    for g in patterns:
        pat = re.compile(g["pattern"])
        params = [p for n,p in named if pat.search(n) and p.requires_grad]
        if params:
            groups.append({"params": params, "lr": g["lr"], "weight_decay": g.get("weight_decay",0.01)})
            used.update(params)
    rest = [p for _,p in named if p.requires_grad and p not in used]
    if rest:
        base = patterns[-1] if patterns else {"lr":1e-4,"weight_decay":0.01}
        groups.append({"params": rest, "lr": base["lr"], "weight_decay": base.get("weight_decay",0.01)})
    return groups

# ------------------------------------------------------------
# Losses & class weights
# ------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float32)  # store as fp32
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        # match dtype & device of logits (fp16/bf16 under autocast)
        alpha = None
        if self.alpha is not None:
            alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)

        ce = F.cross_entropy(logits, target, reduction="none", weight=alpha)
        pt = torch.exp(-ce)
        fl = (1 - pt) ** self.gamma * ce

        if self.reduction == "mean":
            return fl.mean()
        if self.reduction == "sum":
            return fl.sum()
        return fl


def inv_freq_weights(train_ids: np.ndarray, num_classes: int, clip=(0.25,4.0)):
    cnt = np.bincount(train_ids, minlength=num_classes).astype(float)
    w = 1.0 / np.clip(cnt, 1, None)
    w = w * (len(w)/w.sum())
    if clip: w = np.clip(w, clip[0], clip[1])
    return w, cnt

def effective_number_alpha(train_ids: np.ndarray, num_classes: int, beta=0.999, normalize_mean_to_1=True):
    cnt = np.bincount(train_ids, minlength=num_classes).astype(float)
    eff = (1.0 - np.power(beta, cnt)) / (1.0 - beta)
    w = 1.0 / np.clip(eff, 1e-8, None)
    if normalize_mean_to_1 and w.sum()>0: w = w*(len(w)/w.sum())
    return w, cnt

# ------------------------------------------------------------
# Train / eval loops (with AMP; optional per-batch scheduler)
# ------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None, device="cuda", desc="train",
              scaler: Optional[torch.cuda.amp.GradScaler]=None,
              scheduler_step_per_batch: Optional[Callable[[], None]]=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    losses=[]; y_true=[]; y_pred=[]
    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        if batch is None: continue
        xb, yb = batch
        xb = xb.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        yb = yb.to(device, non_blocking=True)

        if train_mode:
            with torch.amp.autocast(device_type="cuda", enabled=(device=="cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler_step_per_batch is not None:
                scheduler_step_per_batch()
        else:
            with torch.no_grad():
                logits = model(xb)
                loss = criterion(logits, yb) if criterion is not None else F.cross_entropy(logits, yb)

        with torch.no_grad():
            pred = logits.argmax(1)
        losses.append(loss.item()); y_true.append(yb.cpu()); y_pred.append(pred.cpu())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    if not losses: return float("nan"), 0.0, 0.0
    y_true = torch.cat(y_true).numpy(); y_pred = torch.cat(y_pred).numpy()
    return float(np.mean(losses)), accuracy_score(y_true,y_pred), f1_score(y_true,y_pred,average="macro")

def train_stage(stage_name, model, train_loader, val_loader, criterion, optimizer, scheduler_epoch,
                epochs, log_csv, best_ckpt_path, epoch_base=0, device="cuda",
                early_patience=10,
                scheduler_step_per_batch: Optional[Callable[[], None]]=None):
    best_f1 = -1.0
    best_epoch = 0
    no_improve = 0
    scaler = torch.amp.GradScaler(enabled=(device=="cuda"))

    for local_epoch in range(1, epochs+1):
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, criterion, optimizer, device,
                                           f"{stage_name} | train {local_epoch}/{epochs}",
                                           scaler=scaler,
                                           scheduler_step_per_batch=scheduler_step_per_batch)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader,   criterion, None,      device,
                                           f"{stage_name} | val   {local_epoch}/{epochs}",
                                           scaler=None, scheduler_step_per_batch=None)

        if scheduler_epoch is not None:
            # epoch-level schedulers (e.g., ReduceLROnPlateau on val loss)
            scheduler_epoch.step(va_loss)

        lr0 = next(iter(optimizer.param_groups))["lr"]
        csv_logger_write(log_csv, {
            "stage": stage_name, "epoch": local_epoch,
            "cum_epoch": epoch_base + local_epoch,
            "train_loss": tr_loss, "train_acc": tr_acc, "train_macro_f1": tr_f1,
            "val_loss": va_loss,   "val_acc": va_acc,   "val_macro_f1": va_f1,
            "lr": lr0
        })

        if va_f1 > best_f1:
            best_f1 = va_f1; best_epoch = local_epoch; no_improve = 0
            torch.save({"model": model.state_dict(), "stage": stage_name,
                        "epoch": local_epoch, "val_macro_f1": va_f1}, best_ckpt_path)
            print(f"[{stage_name}] epoch {local_epoch}: new BEST macro-F1={va_f1:.4f} → saved")
        else:
            no_improve += 1
            if no_improve >= early_patience:
                print(f"[{stage_name}] early stop at epoch {local_epoch} (best {best_epoch} F1={best_f1:.4f})")
                return epoch_base + local_epoch, best_f1

    return epoch_base + epochs, best_f1

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, y_true, y_pred = [], [], []
    for batch in loader:
        if batch is None: continue
        xb, yb = batch
        xb = xb.to(device).to(memory_format=torch.channels_last); yb = yb.to(device)
        with torch.amp.autocast(device_type="cuda", enabled=(device=="cuda")):
            logits = model(xb)
        loss = criterion(logits, yb) if criterion is not None else F.cross_entropy(logits, yb)
        losses.append(loss.item())
        y_true.append(yb.cpu()); y_pred.append(logits.argmax(1).cpu())
    y_true = torch.cat(y_true).numpy(); y_pred = torch.cat(y_pred).numpy()
    return float(np.mean(losses)), accuracy_score(y_true,y_pred), f1_score(y_true,y_pred,average="macro")

def plot_curves(csv_path, out_png):
    df = pd.read_csv(csv_path)
    x = df["cum_epoch"] if "cum_epoch" in df.columns else df["epoch"]
    fig, axes = plt.subplots(1,2, figsize=(11,4))
    axes[0].plot(x, df["train_loss"], label="train"); axes[0].plot(x, df["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend(); axes[0].grid(ls="--", alpha=0.4)
    axes[1].plot(x, df["train_acc"], label="train acc"); axes[1].plot(x, df["val_acc"], label="val acc")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].grid(ls="--", alpha=0.4)
    plt.suptitle("Train/Val Loss & Accuracy"); plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close(fig)

def confusion_png(model, loader, class_names, device, out_prefix):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            if batch is None: continue
            xb, yb = batch
            xb = xb.to(device).to(memory_format=torch.channels_last); yb = yb.to(device)
            ps.append(model(xb).argmax(1).cpu().numpy())
            ys.append(yb.cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    cm = confusion_matrix(y, p, labels=list(range(len(class_names))))
    np.savetxt(out_prefix + "_counts.csv", cm, fmt="%d", delimiter=",")

    def _plot(mat, title, path, normalize=False):
        if normalize:
            with np.errstate(divide='ignore', invalid='ignore'):
                mat = mat / mat.sum(axis=1, keepdims=True)
                mat = np.nan_to_num(mat)
        fig, ax = plt.subplots(figsize=(9,7))
        im = ax.imshow(mat, cmap="Blues")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right"); ax.set_yticklabels(class_names)
        ax.set_xlabel("Pred"); ax.set_ylabel("True"); ax.set_title(title)
        H, W = mat.shape; thr = mat.max() * 0.6
        for i in range(H):
            for j in range(W):
                txt = f"{mat[i,j]:.2f}" if normalize else f"{int(mat[i,j])}"
                ax.text(j, i, txt, ha="center", va="center",
                        color=("white" if mat[i,j] > thr else "black"), fontsize=8)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)

    _plot(cm, "Confusion (counts)", out_prefix + "_counts.png", False)
    _plot(cm, "Confusion (row-normalized)", out_prefix + "_rownorm.png", True)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_name", default="")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)
    cfg = resolve_paths(cfg)

    # run folder
    run_id = f"{timestamp()}__{cfg['model']['arch']}"
    if args.run_name: run_id += f"__{args.run_name}"
    run_dir = os.path.join(cfg["paths"]["runs_dir"], run_id)
    print("Run directory →", run_dir)
    ensure_dir(run_dir)
    log_csv = csv_logger_init(os.path.join(run_dir, "train_log.csv"))
    best_ckpt = os.path.join(run_dir, "best.pth")
    last_ckpt = os.path.join(run_dir, "last.pth")

    # 1) Keep the exact input config you passed on the CLI
    shutil.copy2(args.config, os.path.join(run_dir, "config.input.json"))

    # 2) Save the fully-resolved config we actually used
    with open(os.path.join(run_dir, "config.resolved.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # classes
    with open(cfg["paths"]["classes"], "r") as f:
        raw = json.load(f)
    name_to_id, id_to_name, classes_norm = _parse_classes(raw)
    with open(os.path.join(run_dir, "class_map.json"), "w") as f:
        json.dump(classes_norm, f, indent=2)

    seed = int(cfg["split"].get("seed", 42))
    seed_everything(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)

    # manifest + filter
    df = pd.read_csv(cfg["paths"]["manifest"])
    IOU_THR = cfg["filters"]["min_iou"]
    IOA_THR = cfg["filters"]["min_ioa"]
    df = df.query("max_cross_iou < @IOU_THR and max_cross_ioa < @IOA_THR").reset_index(drop=True)
    
    df = df[df["class"].isin(name_to_id.keys())].reset_index(drop=True)
    print("Manifest size after filtering:", len(df))

    # split
    train_df, val_df, test_df = stratified_group_split(
        df, img_col="img_path", class_col="class",
        train_ratio=cfg["split"]["train"], val_ratio=cfg["split"]["val"],
        test_ratio=cfg["split"]["test"], seed=cfg["split"]["seed"])

    # paths
    train_df = attach_abs_paths(train_df, cfg["paths"]["crops_dir"])
    val_df   = attach_abs_paths(val_df,   cfg["paths"]["crops_dir"])
    test_df  = attach_abs_paths(test_df,  cfg["paths"]["crops_dir"])
    print("Split sizes (after path resolution):",
          f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # loaders
    train_loader, val_loader, test_loader = build_loaders(cfg, train_df, val_df, test_df, name_to_id, generator=g)

    # model & device setup
    num_classes = len(name_to_id)
    model = build_model(cfg, num_classes=num_classes)
    print(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # speed hints
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    model.to(device)
    model.to(memory_format=torch.channels_last)

    # ---------- Stage A ----------
    if cfg["stage_A"]["freeze"] == "all_but_head":
        freeze_all_but_head(model, cfg["model"]["arch"])
        set_backbone_eval_head_train(model, cfg["model"]["arch"])  # BN fix

    # CE + class weights + optional label smoothing
    train_ids = np.array([name_to_id[c] for c in train_df["class"].values], dtype=np.int64)
    cw_cfg = cfg["stage_A"]["loss"]["class_weights"]
    wA, _ = inv_freq_weights(train_ids, num_classes,
                             clip=(cw_cfg["clip_min"], cw_cfg["clip_max"])) if cw_cfg["mode"]=="inv_freq" else (None,None)
    ls = float(cfg["stage_A"]["loss"].get("label_smoothing", 0.0))
    critA = nn.CrossEntropyLoss(weight=(torch.tensor(wA, dtype=torch.float32).to(device) if wA is not None else None),
                                label_smoothing=ls)

    optA = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=cfg["stage_A"]["optimizer"]["lr"],
                             weight_decay=cfg["stage_A"]["optimizer"]["weight_decay"])
    schA = torch.optim.lr_scheduler.ReduceLROnPlateau(optA, mode="min",
             patience=cfg["stage_A"]["scheduler"]["patience"],
             factor=cfg["stage_A"]["scheduler"]["factor"],
             min_lr=cfg["stage_A"]["scheduler"]["min_lr"])

    pat = cfg["metrics"].get("early_stop_patience", 8)
    epoch_base = 0
    epoch_base, _ = train_stage("StageA", model, train_loader, val_loader,
                                critA, optA, schA, cfg["stage_A"]["epochs"],
                                log_csv, best_ckpt, epoch_base, device,
                                early_patience=pat,
                                scheduler_step_per_batch=None)  # Stage A uses epoch-level RoP

    # ---------- Stage B ----------
    # Freeze everything
    for _, p in model.named_parameters():
        p.requires_grad = False

    arch = cfg["model"]["arch"]
    last_k = int(cfg["stage_B"].get("last_k_blocks", 1))

    # Use config-provided patterns if present; otherwise auto-infer last block(s) + head
    cfg_pats = cfg["stage_B"].get("unfreeze_patterns", None)
    if cfg_pats is None:
        unfreeze_pats, head_name = infer_last_block_patterns(model, arch, last_k=last_k)
    else:
        unfreeze_pats = cfg_pats
        head_name = "fc" if arch.startswith("resnet") else "classifier"

    # Unfreeze only those layers
    for patn in unfreeze_pats:
        rgx = re.compile(patn)
        for n, p in model.named_parameters():
            if rgx.search(n):
                p.requires_grad = True

    # (Optional: one-time debug)
    if cfg["logging"].get("print_stageB_trainable", False):
        print("== Stage-B trainable params ==")
        for n, p in model.named_parameters():
            if p.requires_grad:
                print("TRAIN:", n)

    # Optimizer param groups — build from the SAME patterns we just unfroze (unless user provided explicit groups)
    pg_cfg = cfg["stage_B"]["optimizer"].get("param_groups", [])

    if not pg_cfg:
        # Split head vs backbone tail blocks with discriminative LRs
        # Reuse computed patterns; ensure head is its own group
        if cfg_pats is None:
            # we used auto-inference above
            pats, head_name_local = unfreeze_pats, head_name
        else:
            # build from the given patterns
            pats = unfreeze_pats
            head_name_local = head_name

        # Identify the single head pattern (fc or classifier) and the rest (tail blocks)
        head_pat = head_name_local
        tail_pats = [p for p in pats if p != head_pat]

        backbone_lr = cfg["stage_B"]["optimizer"].get("backbone_lr", 1e-4)
        head_lr     = cfg["stage_B"]["optimizer"].get("head_lr",     5e-4)
        wd          = cfg["stage_B"]["optimizer"].get("weight_decay", 0.01)

        # Build regex-based groups
        auto_groups = []
        for ptn in tail_pats:
            auto_groups.append({"pattern": ptn, "lr": backbone_lr, "weight_decay": wd})
        auto_groups.append({"pattern": head_pat, "lr": head_lr, "weight_decay": wd})

        param_groups = param_groups_by_pattern(model, auto_groups)
    else:
        param_groups = param_groups_by_pattern(model, pg_cfg)

    # Fallback safety (shouldn’t trigger if patterns matched)
    if not param_groups:
        param_groups = [{
            "params": [p for p in model.parameters() if p.requires_grad],
            "lr": cfg["stage_B"]["optimizer"].get("lr", 1e-4),
            "weight_decay": cfg["stage_B"]["optimizer"].get("weight_decay", 0.01),
        }]

    optB = torch.optim.AdamW(param_groups)

    # Scheduler selection
    sched_cfg = cfg["stage_B"].get("scheduler", {"type": "reduce_on_plateau"})
    schB_epoch = None           # epoch-level scheduler (e.g., ReduceLROnPlateau)
    step_per_batch = None       # batch-level scheduler step (e.g., cosine)

    if sched_cfg.get("type", "reduce_on_plateau") == "reduce_on_plateau":
        # Notebook behavior: epoch-level RoP on val loss
        schB_epoch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optB, mode="min",
            patience=sched_cfg.get("patience", 5),
            factor=sched_cfg.get("factor", 0.5),
            min_lr=sched_cfg.get("min_lr", 1e-6),
        )
    else:
        # Cosine with warmup (and optional flat phase), per-batch stepping
        total_steps  = len(train_loader) * cfg["stage_B"]["epochs"]
        warmup_steps = len(train_loader) * int(sched_cfg.get("warmup_epochs", 1))
        flat_ratio   = float(sched_cfg.get("flat_ratio", 0.0))  # e.g., 0.15 holds peak LR for 15% of Stage B
        flat_steps   = int(flat_ratio * total_steps)
        min_lr       = float(sched_cfg.get("min_lr", 1e-6))
        base_lrs     = [g["lr"] for g in optB.param_groups]

        from torch.optim.lr_scheduler import LambdaLR
        def lr_lambda(step):
            if total_steps <= 0:
                return 1.0
            # linear warmup → 1.0
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            # optional flat plateau at 1.0
            if step < warmup_steps + flat_steps:
                return 1.0
            # cosine 1.0 → 0.0 over the remainder
            rem = max(1, total_steps - warmup_steps - flat_steps)
            progress = (step - warmup_steps - flat_steps) / rem
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        schB_batch = LambdaLR(optB, lr_lambda=lr_lambda)

        def step_per_batch():
            schB_batch.step()
            # Enforce LR floor
            for g, _base in zip(optB.param_groups, base_lrs):
                g["lr"] = max(g["lr"], min_lr)

    # Loss: Class-Balanced Focal (alpha from effective number).  AMP-safe via dtype cast in FocalLoss.
    a_cfg = cfg["stage_B"]["loss"]["alpha"]
    alphaB, _ = effective_number_alpha(
        train_ids, num_classes,
        beta=a_cfg.get("beta", 0.999),
        normalize_mean_to_1=a_cfg.get("normalize_mean_to_1", True),
    )
    critB = FocalLoss(
        alpha=torch.tensor(alphaB, dtype=torch.float32).to(device),
        gamma=cfg["stage_B"]["loss"].get("gamma", 2.0),
        reduction="mean"
    )

    # Train Stage B (early stopping kept; AMP handled inside run_epoch)
    epoch_base, _ = train_stage(
        "StageB",
        model, train_loader, val_loader,
        critB, optB,
        scheduler_epoch=schB_epoch,                  # RoP (epoch-level) if chosen
        epochs=cfg["stage_B"]["epochs"],
        log_csv=log_csv, best_ckpt_path=best_ckpt,
        epoch_base=epoch_base, device=device,
        early_patience=pat,                           # keep early stopping
        scheduler_step_per_batch=step_per_batch,      # cosine path only; None for RoP
    )

    # Save last state
    torch.save({"model": model.state_dict(), "stage": "last"}, last_ckpt)

    # Evaluate BEST checkpoint on val & test and log
    best = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(best["model"])
    val_loss,  val_acc,  val_f1  = evaluate(model, val_loader,  critB, device)
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, critB, device)
    csv_logger_write(log_csv, {"stage":"BEST_VAL","epoch":0,"cum_epoch":epoch_base,
        "train_loss":"","train_acc":"","train_macro_f1":"",
        "val_loss":val_loss,"val_acc":val_acc,"val_macro_f1":val_f1,"lr":""})
    csv_logger_write(log_csv, {"stage":"TEST","epoch":0,"cum_epoch":epoch_base,
        "train_loss":"","train_acc":"","train_macro_f1":"",
        "val_loss":test_loss,"val_acc":test_acc,"val_macro_f1":test_f1,"lr":""})
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump({"val":{"loss":val_loss,"acc":val_acc,"macro_f1":val_f1},
                   "test":{"loss":test_loss,"acc":test_acc,"macro_f1":test_f1}}, f, indent=2)
    print(f"[BEST] val acc={val_acc:.4f} F1={val_f1:.4f} | [TEST] acc={test_acc:.4f} F1={test_f1:.4f}")

    # Curves & confusion
    if cfg["logging"].get("save_curves", True):
        plot_curves(log_csv, os.path.join(run_dir, "curves.png"))
    if cfg["logging"].get("save_confusion", True):
        names = [id_to_name[i] for i in range(len(id_to_name))]
        confusion_png(model, val_loader,  names, device, os.path.join(run_dir,"val_confusion"))
        confusion_png(model, test_loader, names, device, os.path.join(run_dir,"test_confusion"))

    print(f"✓ Done. Run directory: {run_dir}")

if __name__ == "__main__":
    main()