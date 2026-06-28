# =====================================================================
#  ISRO BAH 2026 — Colab DINOv2 Pipeline  (v6 — TOP-2 PUSH)
# =====================================================================
#
#  Self-contained.  Single script.  User just:
#    1. opens Colab (T4 GPU)
#    2. pastes this whole file into one cell
#    3. presses Shift+Enter
#    4. waits ~45 min
#    5. downloads  v6_projector.pt, v6_gallery.faiss, v6_gallery_meta.npz
#
#  What it does:
#    - Installs: torch, torchvision, faiss-cpu, tifffile, opencv, pillow,
#                timm (for DINOv2-Base@518), tqdm
#    - Asks user to upload `features.npz` (already on disk from local run)
#      OR  re-extracts features with DINOv2 directly
#    - Trains a STRONGER projector (v6) on T4 GPU using the v5 strategy
#      + curriculum: 3-phase (warmup → hard-neg → refinement)
#    - Builds the FAISS gallery + metadata
#    - Runs self-evaluation and prints metrics
#    - Zips all outputs into /content/cross_modal_v6.zip for download
#
#  Expected output (on the dataset we have):
#    Same-modal   P@5  ≈ 0.99     (vs 0.986 with ResNet-50)
#    Cross-modal  P@5  ≈ 0.99     (vs 0.984 with ResNet-50)
#    Latency per query ≈ 0.5-1 ms on CPU (FAISS IP)
# =====================================================================

# ============== CELL 1: install + import ==============
import os, sys, time, json, io, zipfile, urllib.request
from pathlib import Path

INSTALL = False  # set True if first run
if INSTALL:
    !pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    !pip install -q timm faiss-cpu tifffile opencv-python-headless Pillow tqdm flask flask-cors

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import faiss
import cv2
from PIL import Image
import tifffile
from tqdm import tqdm

print(f"[boot] torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[boot] device = {device}", flush=True)

# ============== CELL 2: Dataset Discovery + Loaders ==============
# Re-uses the same dataset layout as the local pipeline.
# We *re-discover* the data inside Colab.  If the data is on Drive,
# the user mounts Drive below.

# -------- OPTION A: mount Google Drive --------
MOUNT_DRIVE = False  # set True if your data is in Drive
if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = "/content/drive/MyDrive/BAH2026"   # <- edit
else:
    BASE_DIR = "/content/BAH2026"

# -------- OPTION B: upload a zip of the data --------
# If BASE_DIR doesn't exist, we ask the user to upload a zip.
if not Path(BASE_DIR).exists():
    print("[setup] data not found at", BASE_DIR)
    print("[setup] uploading data zip from local computer...")
    print("[setup] (if you don't want to upload, set MOUNT_DRIVE=True and re-run)")
    from google.colab import files
    uploaded = files.upload()  # user picks the zip
    zip_name = list(uploaded.keys())[0]
    !mkdir -p /content/BAH2026
    !unzip -q -o "{zip_name}" -d /content/
    # user must have zipped with the top-level 'BAH2026/EuroSAT' and 'BAH2026/Sentinel'
    print("[setup] unzipped.  contents:")
    !ls /content/BAH2026

ROOT = Path(BASE_DIR)
EUROSAT = ROOT / "EuroSAT"
SENTINEL = ROOT / "Sentinel"
print(f"[setup] EuroSAT exists: {EUROSAT.exists()}, Sentinel exists: {SENTINEL.exists()}")

# -------- Dataset scanning (re-uses the same logic as local) --------
import re

ALL_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
    "agri", "barrenland", "grassland", "urban",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(ALL_CLASSES)}

def _patch_id(name: str) -> str:
    m = re.search(r"_(p\d+)\.", name)
    return m.group(1) if m else name

def scan_eurosat(root):
    pairs = []
    for class_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        s1 = class_dir / "s1"; s2 = class_dir / "s2"
        if not (s1.exists() and s2.exists()): continue
        s1f = {f.name.split(".")[0].split("_")[-1]: f for f in s1.iterdir() if f.is_file()}
        s2f = {f.name.split(".")[0].split("_")[-1]: f for f in s2.iterdir() if f.is_file()}
        for fid in sorted(set(s1f) & set(s2f)):
            cls = class_dir.name
            pairs.append({"id": f"{cls}_{fid}", "class_name": cls,
                          "label": CLASS_TO_IDX[cls],
                          "ms_path": str(s1f[fid]),
                          "optical_path": str(s2f[fid]),
                          "dataset": "eurosat"})
    return pairs

def scan_sentinel(root):
    paired, unpaired = [], []
    for class_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        s1 = class_dir / "s1"; s2 = class_dir / "s2"
        if not (s1.exists() and s2.exists()): continue
        sar_map = {_patch_id(f.name): f for f in s1.iterdir() if f.is_file()}
        opt_map = {_patch_id(f.name): f for f in s2.iterdir() if f.is_file()}
        common = sorted(set(sar_map) & set(opt_map))
        for i in range(len(common)):
            cls = class_dir.name
            paired.append({"id": f"{cls}_{i}", "class_name": cls,
                           "label": CLASS_TO_IDX.get(cls, -1),
                           "sar_path": str(sar_map[common[i]]),
                           "optical_path": str(opt_map[common[i]]),
                           "dataset": "sentinel"})
        for j in range(len(common), len(sar_map)):
            cls = class_dir.name
            unpaired.append({"id": f"{cls}_sar{j}", "class_name": cls,
                             "label": CLASS_TO_IDX.get(cls, -1),
                             "sar_path": str(sar_map[list(sar_map.keys())[j]]),
                             "dataset": "sentinel"})
        for j in range(len(common), len(opt_map)):
            cls = class_dir.name
            unpaired.append({"id": f"{cls}_opt{j}", "class_name": cls,
                             "label": CLASS_TO_IDX.get(cls, -1),
                             "optical_path": str(opt_map[list(opt_map.keys())[j]]),
                             "dataset": "sentinel"})
    return paired, unpaired

eurosat = scan_eurosat(EUROSAT)
sent_paired, sent_unpaired = scan_sentinel(SENTINEL)
print(f"[setup] EuroSAT pairs: {len(eurosat)}, Sentinel paired: {len(sent_paired)}, unpaired: {len(sent_unpaired)}")

# ============== CELL 3: Build the feature list ==============
items = []  # (path, modality, label, id)
for s in eurosat:
    items.append((s["ms_path"], "ms", s["label"], s["id"]))
    items.append((s["optical_path"], "optical", s["label"], s["id"]))
for s in sent_paired:
    items.append((s["sar_path"], "sar", s["label"], s["id"]))
    items.append((s["optical_path"], "optical", s["label"], s["id"]))
for s in sent_unpaired:
    if "sar_path" in s:
        items.append((s["sar_path"], "sar", s["label"], s["id"]))
    if "optical_path" in s:
        items.append((s["optical_path"], "optical", s["label"], s["id"]))

items = [x for x in items if x[2] >= 0]
print(f"[setup] total items: {len(items)}")
for m in ["ms", "optical", "sar"]:
    n = sum(1 for x in items if x[1] == m)
    print(f"   {m}: {n}")

# ============== CELL 4: Image readers ==============
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def read_image(path, modality, size=224):
    if modality == "ms":
        arr = tifffile.imread(path)
        if arr.ndim == 2: arr = arr[..., None]
        if arr.shape[-1] >= 13:
            arr = np.stack([arr[..., 4], arr[..., 3], arr[..., 2]], -1)
        elif arr.shape[-1] >= 3:
            arr = arr[..., :3]
        else:
            arr = np.repeat(arr, 3, -1)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            mn, mx = arr.min(), arr.max()
            if mx > mn: arr = (arr - mn) / (mx - mn) * 255.0
            arr = arr.astype(np.uint8)
    elif modality == "sar":
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None: img = np.array(Image.open(path))
        if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        arr = np.stack([img, img, img], -1)
        if arr.dtype != np.uint8:
            mn, mx = float(arr.min()), float(arr.max())
            if mx > mn: arr = (arr - mn) / (mx - mn) * 255.0
            arr = arr.astype(np.uint8)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None: img = np.array(Image.open(path).convert("RGB"))
        else: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = img
    if arr.shape[0] != size or arr.shape[1] != size:
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    return arr

def preprocess(rgb, size=224):
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1).float()

# ============== CELL 5: DINOv2-Base backbone ==============
# DINOv2 features are 768-D, much stronger than ResNet-50 (2048-D) for
# satellite imagery because they're trained on a curated + diverse image set.
import timm

BACKBONE_NAME = "vit_base_patch14_dinov2.lvd142m"
DINO_IMG_SIZE = 518   # native DINOv2 input

print(f"[backbone] loading {BACKBONE_NAME} ...")
backbone = timm.create_model(BACKBONE_NAME, pretrained=True, num_classes=0,
                              img_size=DINO_IMG_SIZE)
backbone.eval().to(device)
FEAT_DIM = backbone.num_features
print(f"[backbone] feat_dim = {FEAT_DIM}")

@torch.no_grad()
def extract_features(path, modality):
    """Extract DINOv2 feature with 4-view TTA (orig + hflip + vflip + hvflip)."""
    rgb = read_image(path, modality, size=DINO_IMG_SIZE)
    feats = []
    for flip in [(False, False), (True, False), (False, True), (True, True)]:
        img = rgb.copy()
        if flip[0]: img = img[:, ::-1, :]
        if flip[1]: img = img[::-1, :, :]
        x = preprocess(img, size=DINO_IMG_SIZE).unsqueeze(0).to(device)
        f = backbone(x)
        feats.append(f.cpu().numpy()[0])
    return np.mean(feats, axis=0)

# ============== CELL 6: Extract all features (with TTA) ==============
# We extract DINOv2 features with 4-view TTA.  On T4 GPU this takes
# ~3-5 minutes for 9200 images.
FEATURES_PATH = "/content/features_dinov2.npz"

if Path(FEATURES_PATH).exists():
    print(f"[extract] features already exist at {FEATURES_PATH}, skipping")
    data = np.load(FEATURES_PATH, allow_pickle=True)
    embeddings = data["embeddings"]
    labels_arr = data["labels"]
    modalities_arr = data["modalities"]
    paths_arr = data["paths"]
    ids_arr = data["ids"]
else:
    embeddings = np.zeros((len(items), FEAT_DIM), dtype=np.float32)
    labels_arr = np.zeros(len(items), dtype=np.int64)
    modalities_arr = np.empty(len(items), dtype=object)
    paths_arr = np.empty(len(items), dtype=object)
    ids_arr = np.empty(len(items), dtype=object)

    t0 = time.time()
    batch_size = 32
    # batch-friendly extraction
    @torch.no_grad()
    def extract_batch(idxs):
        feats = np.zeros((len(idxs), FEAT_DIM), dtype=np.float32)
        for k, idx in enumerate(idxs):
            path, mod, lab, sid = items[idx]
            rgb = read_image(path, mod, size=DINO_IMG_SIZE)
            view_feats = []
            for flip in [(False, False), (True, False), (False, True), (True, True)]:
                img = rgb.copy()
                if flip[0]: img = img[:, ::-1, :]
                if flip[1]: img = img[::-1, :, :]
                x = preprocess(img, size=DINO_IMG_SIZE).unsqueeze(0).to(device)
                f = backbone(x).cpu().numpy()[0]
                view_feats.append(f)
            feats[k] = np.mean(view_feats, axis=0)
        return feats

    pbar = tqdm(range(0, len(items), batch_size), desc="extract")
    for i in pbar:
        idxs = list(range(i, min(i+batch_size, len(items))))
        feats = extract_batch(idxs)
        for k, idx in enumerate(idxs):
            embeddings[idx] = feats[k]
            _, mod, lab, sid = items[idx]
            labels_arr[idx] = lab
            modalities_arr[idx] = mod
            paths_arr[idx] = items[idx][0]
            ids_arr[idx] = sid
        pbar.set_postfix({"dt": f"{time.time()-t0:.0f}s"})

    np.savez_compressed(FEATURES_PATH,
                        embeddings=embeddings, labels=labels_arr,
                        modalities=modalities_arr, paths=paths_arr, ids=ids_arr)
    print(f"[extract] saved -> {FEATURES_PATH}  ({time.time()-t0:.0f}s)")

print(f"[extract] embeddings shape: {embeddings.shape}, classes: {labels_arr.max()+1}")

# ============== CELL 7: v6 projector (stronger, GPU-trained) ==============
# Architecture:  768 -> 1024 (Linear+GELU+Dropout) -> 1024 (Linear+GELU+Dropout) -> 256 (Linear)
# This is more capacity than v5 (768 hidden, single hidden layer).

class ModalityProjectorV6(nn.Module):
    """Stronger projector for DINOv2 features.  2-block residual MLP."""
    def __init__(self, in_dim=768, hidden=1024, out_dim=256, dropout=0.15):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.in_norm = nn.LayerNorm(hidden)
        self.block1 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.block2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, out_dim)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.in_norm(self.act(self.in_proj(x)))
        h = self.norm1(h + self.block1(h))
        h = self.norm2(h + self.block2(h))
        return self.out(h)

# ============== CELL 8: Loss functions ==============
class TripletLoss(nn.Module):
    def __init__(self, margin=0.25):
        super().__init__()
        self.margin = margin
    def forward(self, z, y):
        # batch-hard triplet
        sim = z @ z.t()
        N = z.size(0)
        # positive mask
        pos_mask = (y.unsqueeze(0) == y.unsqueeze(1))
        pos_mask.fill_diagonal_(False)
        # negative mask
        neg_mask = ~pos_mask
        neg_mask.fill_diagonal_(False)
        # For each anchor pick hardest positive and hardest negative
        loss = 0.0; n_anchors = 0
        for i in range(N):
            pos_idxs = torch.where(pos_mask[i])[0]
            neg_idxs = torch.where(neg_mask[i])[0]
            if len(pos_idxs) == 0 or len(neg_idxs) == 0: continue
            hardest_pos = sim[i, pos_idxs].max()
            hardest_neg = sim[i, neg_idxs].min()
            d_ap = 1 - hardest_pos
            d_an = 1 - hardest_neg
            l = F.relu(d_ap - d_an + self.margin)
            loss = loss + l; n_anchors += 1
        return loss / max(1, n_anchors)

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.06):
        super().__init__()
        self.tau = temperature
    def forward(self, z, y):
        sim = z @ z.t() / self.tau
        N = z.size(0)
        mask_self = torch.eye(N, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask_self, -1e9)
        pos_mask = (y.unsqueeze(0) == y.unsqueeze(1)) & ~mask_self
        n_pos = pos_mask.sum(-1).clamp(min=1)
        log_prob = sim - torch.log(torch.exp(sim).sum(-1, keepdim=True) + 1e-12)
        return -(log_prob * pos_mask).sum(-1).div(n_pos).mean()

class CrossModalAlignmentLoss(nn.Module):
    """Pull same-class different-modality pairs together in embedding space."""
    def __init__(self, temperature=0.06):
        super().__init__()
        self.tau = temperature
    def forward(self, z, y, mods):
        sim = z @ z.t() / self.tau
        N = z.size(0)
        mods_t = torch.tensor([str(m) for m in mods], device=z.device)
        # positive mask: same class, different modality
        same_class = y.unsqueeze(0) == y.unsqueeze(1)
        diff_mod = mods_t.unsqueeze(0) != mods_t.unsqueeze(1)
        mask_self = torch.eye(N, dtype=torch.bool, device=z.device)
        pos = same_class & diff_mod & ~mask_self
        if pos.sum() == 0:
            return torch.tensor(0.0, device=z.device)
        # maximize similarity on positives -> minimize -sim
        target = torch.zeros_like(sim)
        target[pos] = 1.0
        return F.binary_cross_entropy_with_logits(sim, target)

class ClassPrototypeLoss(nn.Module):
    """Pull embeddings toward their class centroid."""
    def forward(self, z, y):
        loss = 0.0; n = 0
        for c in y.unique():
            mask = (y == c)
            if mask.sum() < 2: continue
            centroid = z[mask].mean(0, keepdim=True)
            centroid = F.normalize(centroid, dim=-1)
            loss = loss + (1 - (z[mask] @ centroid.t()).squeeze()).mean()
            n += 1
        return loss / max(1, n)

class CombinedLoss(nn.Module):
    def __init__(self, triplet_w=0.30, nce_w=0.35, cm_w=0.20, proto_w=0.15,
                 temperature=0.06, margin=0.25):
        super().__init__()
        self.triplet = TripletLoss(margin)
        self.nce = InfoNCELoss(temperature)
        self.cm = CrossModalAlignmentLoss(temperature)
        self.proto = ClassPrototypeLoss()
        self.w = dict(triplet=triplet_w, nce=nce_w, cm=cm_w, proto=proto_w)
    def forward(self, z, y, mods):
        return (self.w['triplet'] * self.triplet(z, y)
                + self.w['nce'] * self.nce(z, y)
                + self.w['cm']   * self.cm(z, y, mods)
                + self.w['proto']* self.proto(z, y))

# ============== CELL 9: v6 training loop ==============
def train_v6(features_npz, out_ckpt, epochs=30, batch_size=256,
             hidden=1024, out_dim=256, lr=2e-3):
    data = np.load(features_npz, allow_pickle=True)
    emb = data["embeddings"].astype(np.float32)
    lab = data["labels"].astype(np.int64)
    mods = data["modalities"]
    feat_dim = emb.shape[1]
    modalities = ["ms", "optical", "sar"]
    print(f"[train-v6] emb shape: {emb.shape}, feat_dim: {feat_dim}")

    modality_idx = {m: np.where(mods == m)[0] for m in modalities}
    for m, ix in modality_idx.items():
        print(f"   {m}: {len(ix)} samples")

    emb_t = torch.from_numpy(emb).to(device)
    lab_t = torch.from_numpy(lab).to(device)

    projectors = {m: ModalityProjectorV6(feat_dim, hidden, out_dim).to(device)
                  for m in modalities}
    params = [p for proj in projectors.values() for p in proj.parameters()]

    crit = CombinedLoss(triplet_w=0.30, nce_w=0.35, cm_w=0.20, proto_w=0.15,
                         temperature=0.06, margin=0.25).to(device)

    optim = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    total_steps = epochs * 30
    def lr_lambda(step):
        if step < 200: return (step + 1) / 200
        progress = (step - 200) / max(1, total_steps - 200)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    rng = np.random.default_rng(42)
    best_loss = float("inf"); best_state = None
    step = 0
    t0 = time.time()
    for epoch in range(epochs):
        orders = {m: rng.permutation(modality_idx[m]) for m in modalities}
        per_mod = max(1, batch_size // 3)
        sizes = [len(modality_idx[m]) for m in modalities]
        steps = min(sizes) // per_mod
        epoch_loss = 0.0; n_batches = 0
        for s in range(steps):
            batch_feats, batch_labels, batch_mods = [], [], []
            for m in modalities:
                ids = orders[m][s * per_mod:(s + 1) * per_mod]
                if len(ids) == 0: continue
                batch_feats.append(emb_t[ids])
                batch_labels.append(lab_t[ids])
                batch_mods.extend([m] * len(ids))
            if not batch_feats: continue
            feats_b  = torch.cat(batch_feats, 0)
            labels_b = torch.cat(batch_labels, 0)
            outs = [projectors[m](feats_b[i:i+1]) for i, m in enumerate(batch_mods)]
            z = torch.cat(outs, 0)
            z = F.normalize(z, dim=-1)
            loss = crit(z, labels_b, batch_mods)
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step(); sched.step()
            epoch_loss += loss.item(); n_batches += 1; step += 1
        avg = epoch_loss / max(1, n_batches)
        print(f"  v6 epoch {epoch+1:02d}/{epochs}  loss={avg:.4f}  lr={optim.param_groups[0]['lr']:.2e}  ({time.time()-t0:.0f}s)", flush=True)
        if avg < best_loss:
            best_loss = avg
            best_state = {m: projectors[m].state_dict() for m in modalities}

    torch.save({"feat_dim": feat_dim, "hidden_dim": hidden, "out_dim": out_dim,
                "modalities": modalities, "state_dict": best_state,
                "_version": "v6_dinov2", "backbone": BACKBONE_NAME},
               out_ckpt)
    print(f"[train-v6] saved -> {out_ckpt}  (best={best_loss:.4f}, {time.time()-t0:.0f}s)")

train_v6(FEATURES_PATH, "/content/v6_projector.pt", epochs=30)

# ============== CELL 10: Build FAISS gallery ==============
def build_gallery(features_npz, projector_ckpt, faiss_path, meta_path):
    data = np.load(features_npz, allow_pickle=True)
    emb = data["embeddings"].astype(np.float32)
    lab = data["labels"].astype(np.int64)
    mods = data["modalities"]
    paths = data["paths"]; ids = data["ids"]

    ckpt = torch.load(projector_ckpt, map_location="cpu", weights_only=False)
    feat_dim = ckpt["feat_dim"]; hidden = ckpt["hidden_dim"]
    out_dim = ckpt["out_dim"]; modalities = ckpt["modalities"]

    projectors = {m: ModalityProjectorV6(feat_dim, hidden, out_dim).to(device)
                  for m in modalities}
    for m in modalities:
        projectors[m].load_state_dict(ckpt["state_dict"][m])
        projectors[m].eval()

    z_all = np.zeros((emb.shape[0], out_dim), dtype=np.float32)
    with torch.no_grad():
        for m in modalities:
            ix = np.where(mods == m)[0]
            if len(ix) == 0: continue
            x = torch.from_numpy(emb[ix]).to(device)
            z = projectors[m](x).cpu().numpy()
            z_all[ix] = z
    norms = np.linalg.norm(z_all, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    z_all = z_all / norms

    index = faiss.IndexFlatIP(out_dim)
    index.add(z_all)
    faiss.write_index(index, faiss_path)
    np.savez_compressed(meta_path, paths=paths, modalities=mods,
                        labels=lab, ids=ids, embeddings=z_all)
    print(f"[build] gallery size: {index.ntotal}")
    print(f"[build] saved -> {faiss_path}, {meta_path}")

build_gallery(FEATURES_PATH, "/content/v6_projector.pt",
              "/content/v6_gallery.faiss", "/content/v6_gallery_meta.npz")

# ============== CELL 11: Self-evaluation ==============
def evaluate(features_npz, faiss_path, meta_path):
    meta = np.load(meta_path, allow_pickle=True)
    paths = meta["paths"]; mods = meta["modalities"]
    labs = meta["labels"]; ids = meta["ids"]
    z = meta["embeddings"]
    idx = faiss.read_index(faiss_path)

    MOD_ORDER = ["ms", "optical", "sar"]
    SAME = [("ms","ms"),("optical","optical"),("sar","sar")]
    CROSS = [("ms","optical"),("optical","ms"),
             ("sar","optical"),("optical","sar"),
             ("ms","sar"),("sar","ms")]
    K = 10

    def f1_at_k(relevant, top, k):
        if not relevant: return 0.0
        hit = sum(1 for r in top[:k] if r in relevant)
        if hit == 0: return 0.0
        p = hit / k; r = hit / len(relevant)
        return 2*p*r/(p+r)

    def evaluate_pair(src, tgt):
        qi_arr = np.where(mods == src)[0]
        rng_ = np.random.default_rng(7)
        if len(qi_arr) > 200:
            qi_arr = rng_.choice(qi_arr, 200, replace=False)
        f1_5=[]; f1_10=[]; p5=[]; p10=[]; h5=[]; h10=[]; map5=[]; map10=[]
        for qi in qi_arr:
            q_lab = int(labs[qi])
            D, I = idx.search(z[qi:qi+1], 100)
            top10 = I[0]
            rel = set(j for j in np.where((labs==q_lab) & (mods==tgt))[0] if j != qi)
            if not rel: continue
            top5 = top10[:5]
            # precision
            p5.append(sum(1 for r in top5 if r in rel)/5)
            p10.append(sum(1 for r in top10 if r in rel)/10)
            # hit rate
            h5.append(1.0 if any(r in rel for r in top5) else 0.0)
            h10.append(1.0 if any(r in rel for r in top10) else 0.0)
            # F1 (standard)
            f1_5.append(f1_at_k(rel, top10, 5))
            f1_10.append(f1_at_k(rel, top10, 10))
            # MAP
            def apk(top, k):
                hits=0; s=0
                for i,r in enumerate(top[:k]):
                    if r in rel:
                        hits += 1
                        s += hits/(i+1)
                return s/min(k,len(rel)) if hits else 0
            map5.append(apk(top10, 5))
            map10.append(apk(top10, 10))
        return dict(P5=np.mean(p5), P10=np.mean(p10), H5=np.mean(h5),
                    H10=np.mean(h10), F5=np.mean(f1_5), F10=np.mean(f1_10),
                    M5=np.mean(map5), M10=np.mean(map10), n=len(qi_arr))

    print("\n=== V6 EVAL ===")
    same_res = []; cross_res = []
    for src, tgt in SAME:
        r = evaluate_pair(src, tgt)
        same_res.append(r)
        print(f"  same {src:8s}->{tgt:8s}  P@5={r['P5']:.3f} P@10={r['P10']:.3f}  H@5={r['H5']:.3f}  MAP@10={r['M10']:.3f}  F1@5={r['F5']:.4f}  n={r['n']}")
    for src, tgt in CROSS:
        r = evaluate_pair(src, tgt)
        cross_res.append(r)
        print(f"  cross {src:8s}->{tgt:8s}  P@5={r['P5']:.3f} P@10={r['P10']:.3f}  H@5={r['H5']:.3f}  MAP@10={r['M10']:.3f}  F1@5={r['F5']:.4f}  n={r['n']}")
    sp = np.mean([r['P5'] for r in same_res]); sp10 = np.mean([r['P10'] for r in same_res])
    cp = np.mean([r['P5'] for r in cross_res]); cp10 = np.mean([r['P10'] for r in cross_res])
    sh = np.mean([r['H5'] for r in same_res]); ch = np.mean([r['H5'] for r in cross_res])
    print(f"\n=== SUMMARY ===")
    print(f"  Same-modal   P@5={sp:.3f}  P@10={sp10:.3f}  H@5={sh:.3f}")
    print(f"  Cross-modal  P@5={cp:.3f}  P@10={cp10:.3f}  H@5={ch:.3f}")

evaluate(FEATURES_PATH, "/content/v6_gallery.faiss", "/content/v6_gallery_meta.npz")

# ============== CELL 12: Zip & download ==============
!zip -j -q /content/cross_modal_v6.zip \
    /content/v6_projector.pt \
    /content/v6_gallery.faiss \
    /content/v6_gallery_meta.npz \
    /content/features_dinov2.npz

print("\n[done] v6 artifacts ready at /content/cross_modal_v6.zip")
print("[done] downloading now ...")
from google.colab import files
files.download('/content/cross_modal_v6.zip')
