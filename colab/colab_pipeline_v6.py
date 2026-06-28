# =====================================================================
#  BAH 2026 — DINOv2 V6 Training Pipeline (Improved Edition)
# =====================================================================
#
#  Single-script Colab pipeline.  User does:
#    1. Opens Colab → T4 GPU
#    2. Pastes the whole file into ONE cell
#    3. Uploads `BAH2026_data.zip` when prompted (contains EuroSAT/ + Sentinel/)
#    4. Waits ~25 min
#    5. Downloads `cross_modal_v6.zip` with all v6 artifacts
#
#  What changed vs colab_pipeline.py (v5 → v6 improved):
#    • DINOv2-Base@518 (vs ResNet-50@224) — much stronger visual features
#    • 4-view TTA during extraction (orig + hflip + vflip + hvflip)
#    • 2-block residual projector (1024 hidden, residual + LayerNorm)
#    • 50 epochs cosine schedule + warmup (vs 30)
#    • Hard negative mining inside batch
#    • Class-balanced sampler (every batch has all 14 classes equally)
#    • Validation split (10%) → save best checkpoint by val loss, not train
#    • Eval: 60 queries per (src, tgt) pair, not 200 random cap
#    • F1@5/F1@10 + P@k + Hit@k + MAP@k all reported
#    • Saves per-modality stats
#
#  Expected (vs current local v5):
#    Same-modal   P@5  0.986 → ~0.992-0.995
#    Cross-modal  P@5  0.984 → ~0.990-0.995
#    Latency       0.02ms → 0.02ms (FAISS, same)
# =====================================================================

# ===================== 0. CONFIG =====================
INSTALL = True         # set True if packages missing
MOUNT_DRIVE = False       # set True if data is in Drive (else upload zip)
DATA_ZIP_NAME = "BAH2026_data.zip"

# Training
BACKBONE_NAME = "vit_base_patch14_dinov2.lvd142m"
DINO_IMG_SIZE = 518
EPOCHS = 50
BATCH_SIZE = 384
HIDDEN = 1024
OUT_DIM = 256
LR = 2e-3
VAL_FRAC = 0.10           # 10% of items held out for validation
SEED = 42

# Loss weights (sum to 1.0)
W_TRIPLET = 0.30
W_NCE     = 0.35
W_CM      = 0.20
W_PROTO   = 0.15

# Paths (under /content)
FEATURES_PATH = "/content/features_dinov2.npz"
PROJECTOR_CKPT = "/content/v6_projector.pt"
FAISS_PATH = "/content/v6_gallery.faiss"
META_PATH = "/content/v6_gallery_meta.npz"
EVAL_JSON  = "/content/v6_eval.json"
ZIP_OUTPUT = "/content/cross_modal_v6.zip"

# ===================== 1. INSTALL + IMPORT =====================
import os, sys, time, json, io, zipfile, re
from pathlib import Path
import numpy as np

if INSTALL:
    !pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    !pip install -q timm faiss-cpu tifffile opencv-python-headless Pillow tqdm flask flask-cors

import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss, cv2, tifffile
from PIL import Image
from tqdm import tqdm
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[boot] torch={torch.__version__} cuda={torch.cuda.is_available()} device={device}", flush=True)
assert device == "cuda", "T4 GPU required. Set Runtime → Change runtime type → T4 GPU."

# ===================== 2. DATA: discover + zip upload =====================
BASE_DIR = "/content/BAH2026"

if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = "/content/drive/MyDrive/BAH2026"

if not Path(BASE_DIR).exists():
    print(f"[setup] data not found at {BASE_DIR}")
    print(f"[setup] upload {DATA_ZIP_NAME} (should contain EuroSAT/ + Sentinel/ at top level)")
    from google.colab import files
    uploaded = files.upload()
    zip_name = list(uploaded.keys())[0]
    print(f"[setup] received {zip_name}, unzipping...")
    !mkdir -p /content/BAH2026
    !unzip -q -o "{zip_name}" -d /content/
    print("[setup] unzipped. contents:")
    !ls /content/BAH2026

ROOT = Path(BASE_DIR)
EUROSAT = ROOT / "EuroSAT"
SENTINEL = ROOT / "Sentinel"
assert EUROSAT.exists() and SENTINEL.exists(), \
    f"missing data dirs: {EUROSAT.exists()=} {SENTINEL.exists()=}"

ALL_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
    "agri", "barrenland", "grassland", "urban",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(ALL_CLASSES)}
NUM_CLASSES = len(ALL_CLASSES)
print(f"[setup] {NUM_CLASSES} classes", flush=True)

# ===================== 3. SCAN: pair images across modalities =====================
def _patch_id(name: str) -> str:
    m = re.search(r"_(p\d+)\.", name)
    return m.group(1) if m else name

def scan_eurosat(root):
    """Pair by image id suffix (e.g. PermanentCrop_1.tif <-> PermanentCrop_1.jpg)."""
    pairs = []
    for class_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        s1 = class_dir / "s1"; s2 = class_dir / "s2"
        if not (s1.exists() and s2.exists()): continue
        s1f = {f.name.split(".")[0].split("_")[-1]: f for f in s1.iterdir() if f.is_file()}
        s2f = {f.name.split(".")[0].split("_")[-1]: f for f in s2.iterdir() if f.is_file()}
        for fid in sorted(set(s1f) & set(s2f)):
            cls = class_dir.name
            pairs.append({
                "id": f"{cls}_{fid}", "class_name": cls,
                "label": CLASS_TO_IDX[cls], "dataset": "eurosat",
                "ms_path": str(s1f[fid]), "optical_path": str(s2f[fid]),
            })
    return pairs

def scan_sentinel(root):
    """Pair by patch id (e.g. ROIs..._p10.png <-> ROIs..._p10.png)."""
    paired, unpaired = [], []
    for class_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        s1 = class_dir / "s1"; s2 = class_dir / "s2"
        if not (s1.exists() and s2.exists()): continue
        sar_map = {_patch_id(f.name): f for f in s1.iterdir() if f.is_file()}
        opt_map = {_patch_id(f.name): f for f in s2.iterdir() if f.is_file()}
        common = sorted(set(sar_map) & set(opt_map))
        cls = class_dir.name
        lab = CLASS_TO_IDX.get(cls, -1)
        for i, fid in enumerate(common):
            paired.append({
                "id": f"{cls}_{i}", "class_name": cls, "label": lab, "dataset": "sentinel",
                "sar_path": str(sar_map[fid]), "optical_path": str(opt_map[fid]),
            })
        for j, fid in enumerate(sorted(sar_map.keys())[len(common):]):
            unpaired.append({
                "id": f"{cls}_sar{j}", "class_name": cls, "label": lab, "dataset": "sentinel",
                "sar_path": str(sar_map[fid]),
            })
        for j, fid in enumerate(sorted(opt_map.keys())[len(common):]):
            unpaired.append({
                "id": f"{cls}_opt{j}", "class_name": cls, "label": lab, "dataset": "sentinel",
                "optical_path": str(opt_map[fid]),
            })
    return paired, unpaired

eurosat = scan_eurosat(EUROSAT)
sent_paired, sent_unpaired = scan_sentinel(SENTINEL)
print(f"[scan] EuroSAT pairs: {len(eurosat)}, Sentinel paired: {len(sent_paired)}, unpaired: {len(sent_unpaired)}", flush=True)

# Flatten into (path, modality, label, id) tuples
items = []
for s in eurosat:
    items.append((s["ms_path"], "ms",      s["label"], s["id"]))
    items.append((s["optical_path"], "optical", s["label"], s["id"]))
for s in sent_paired:
    items.append((s["sar_path"],   "sar",     s["label"], s["id"]))
    items.append((s["optical_path"], "optical", s["label"], s["id"]))
for s in sent_unpaired:
    if "sar_path" in s:    items.append((s["sar_path"], "sar",     s["label"], s["id"]))
    if "optical_path" in s: items.append((s["optical_path"], "optical", s["label"], s["id"]))
items = [x for x in items if x[2] >= 0]
print(f"[scan] total items: {len(items)}")
for m in ["ms", "optical", "sar"]:
    print(f"   {m}: {sum(1 for x in items if x[1] == m)}")

# ===================== 4. IMAGE READERS =====================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

def _to_uint8_rgb(path, modality):
    if modality == "ms":
        arr = tifffile.imread(path)
        if arr.ndim == 2: arr = arr[..., None]
        if arr.shape[-1] >= 13:
            arr = np.stack([arr[..., 4], arr[..., 3], arr[..., 2]], -1)   # RGB from Sentinel-2
        elif arr.shape[-1] >= 3:
            arr = arr[..., :3]
        else:
            arr = np.repeat(arr, 3, -1)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            mn, mx = float(arr.min()), float(arr.max())
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
    else:  # optical
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None: img = np.array(Image.open(path).convert("RGB"))
        else: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = img
    return arr

def read_image(path, modality, size=DINO_IMG_SIZE):
    arr = _to_uint8_rgb(path, modality)
    if arr.shape[0] != size or arr.shape[1] != size:
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    return arr

def preprocess(rgb):
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1).float()

# ===================== 5. DINOv2 BACKBONE =====================
print(f"[backbone] loading {BACKBONE_NAME} ...")
backbone = timm.create_model(BACKBONE_NAME, pretrained=True, num_classes=0,
                              img_size=DINO_IMG_SIZE)
backbone.eval().to(device)
FEAT_DIM = backbone.num_features
print(f"[backbone] feat_dim = {FEAT_DIM}", flush=True)

@torch.no_grad()
def extract_one(path, modality):
    """4-view TTA: orig + hflip + vflip + hvflip."""
    rgb = read_image(path, modality)
    feats = []
    for flip in [(False, False), (True, False), (False, True), (True, True)]:
        img = rgb
        if flip[0]: img = img[:, ::-1, :]
        if flip[1]: img = img[::-1, :, :]
        x = preprocess(img).unsqueeze(0).to(device)
        f = backbone(x)
        feats.append(f.cpu().numpy()[0])
    return np.mean(feats, axis=0).astype(np.float32)

# ===================== 6. EXTRACT ALL FEATURES =====================
if Path(FEATURES_PATH).exists():
    print(f"[extract] cached features at {FEATURES_PATH}, skipping extraction")
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
    for i in tqdm(range(len(items)), desc="extract DINOv2 (TTA x4)"):
        path, mod, lab, sid = items[i]
        try:
            embeddings[i] = extract_one(path, mod)
        except Exception as e:
            print(f"[extract] WARN failed on {path}: {e}; using zeros")
        labels_arr[i] = lab
        modalities_arr[i] = mod
        paths_arr[i] = path
        ids_arr[i] = sid
        if (i + 1) % 500 == 0:
            print(f"[extract] {i+1}/{len(items)}  elapsed {time.time()-t0:.0f}s", flush=True)

    np.savez_compressed(FEATURES_PATH,
                        embeddings=embeddings, labels=labels_arr,
                        modalities=modalities_arr, paths=paths_arr, ids=ids_arr)
    print(f"[extract] saved -> {FEATURES_PATH}  ({time.time()-t0:.0f}s total)")

print(f"[extract] shape={embeddings.shape}  classes={int(labels_arr.max())+1}")

# ===================== 7. PROJECTOR V6 =====================
class ModalityProjectorV6(nn.Module):
    """2-block residual MLP:  in → 1024 → (residual×2) → 256."""
    def __init__(self, in_dim, hidden=1024, out_dim=256, dropout=0.15):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.in_norm = nn.LayerNorm(hidden)
        self.block1 = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.norm1 = nn.LayerNorm(hidden)
        self.block2 = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.norm2 = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, out_dim)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.in_norm(self.act(self.in_proj(x)))
        h = self.norm1(h + self.block1(h))
        h = self.norm2(h + self.block2(h))
        return self.out(h)

# ===================== 8. LOSSES =====================
class TripletLoss(nn.Module):
    def __init__(self, margin=0.25):
        super().__init__(); self.margin = margin
    def forward(self, z, y):
        sim = z @ z.t()
        N = z.size(0)
        pos = (y.unsqueeze(0) == y.unsqueeze(1))
        pos.fill_diagonal_(False)
        neg = ~pos.clone(); neg.fill_diagonal_(False)
        loss = z.new_zeros(()); n = 0
        for i in range(N):
            pi = torch.where(pos[i])[0]; ni = torch.where(neg[i])[0]
            if len(pi) == 0 or len(ni) == 0: continue
            hp = sim[i, pi].max(); hn = sim[i, ni].min()
            loss = loss + F.relu(1 - hp - (1 - hn) + self.margin); n += 1
        return loss / max(1, n)

class InfoNCELoss(nn.Module):
    def __init__(self, tau=0.06):
        super().__init__(); self.tau = tau
    def forward(self, z, y):
        sim = z @ z.t() / self.tau
        N = z.size(0)
        eye = torch.eye(N, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(eye, -1e9)
        pos = (y.unsqueeze(0) == y.unsqueeze(1)) & ~eye
        n_pos = pos.sum(-1).clamp(min=1)
        lp = sim - torch.log(torch.exp(sim).sum(-1, keepdim=True) + 1e-12)
        return -(lp * pos).sum(-1).div(n_pos).mean()

class CrossModalAlignmentLoss(nn.Module):
    """Pull same-class different-modality pairs together."""
    def __init__(self, tau=0.06):
        super().__init__(); self.tau = tau
    def forward(self, z, y, mods):
        sim = z @ z.t() / self.tau
        N = z.size(0)
        mods_t = torch.tensor([str(m) for m in mods], device=z.device)
        same_class = y.unsqueeze(0) == y.unsqueeze(1)
        diff_mod = mods_t.unsqueeze(0) != mods_t.unsqueeze(1)
        eye = torch.eye(N, dtype=torch.bool, device=z.device)
        pos = same_class & diff_mod & ~eye
        if pos.sum() == 0: return z.new_zeros(())
        target = torch.zeros_like(sim); target[pos] = 1.0
        return F.binary_cross_entropy_with_logits(sim, target)

class ClassPrototypeLoss(nn.Module):
    def forward(self, z, y):
        loss = z.new_zeros(()); n = 0
        for c in y.unique():
            m = (y == c)
            if m.sum() < 2: continue
            cent = F.normalize(z[m].mean(0, keepdim=True), dim=-1)
            loss = loss + (1 - (z[m] @ cent.t()).squeeze()).mean(); n += 1
        return loss / max(1, n)

class CombinedLoss(nn.Module):
    def __init__(self, w_t=0.30, w_n=0.35, w_c=0.20, w_p=0.15, tau=0.06, margin=0.25):
        super().__init__()
        self.triplet = TripletLoss(margin); self.nce = InfoNCELoss(tau)
        self.cm = CrossModalAlignmentLoss(tau); self.proto = ClassPrototypeLoss()
        self.w = dict(t=w_t, n=w_n, c=w_c, p=w_p)
    def forward(self, z, y, mods):
        return (self.w['t'] * self.triplet(z, y)
                + self.w['n'] * self.nce(z, y)
                + self.w['c'] * self.cm(z, y, mods)
                + self.w['p'] * self.proto(z, y))

# ===================== 9. CLASS-BALANCED SAMPLER + TRAIN/VAL SPLIT =====================
def split_train_val(items, val_frac=0.10, seed=SEED):
    """Stratified split per (label, modality)."""
    rng = np.random.default_rng(seed)
    by_key = {}
    for i, (_, mod, lab, _) in enumerate(items):
        by_key.setdefault((int(lab), mod), []).append(i)
    train_idx, val_idx = [], []
    for key, ix in by_key.items():
        ix = np.array(ix); rng.shuffle(ix)
        n_val = max(1, int(len(ix) * val_frac))
        val_idx.extend(ix[:n_val].tolist())
        train_idx.extend(ix[n_val:].tolist())
    rng.shuffle(train_idx); rng.shuffle(val_idx)
    return train_idx, val_idx

train_idx, val_idx = split_train_val(items, VAL_FRAC, SEED)
print(f"[split] train={len(train_idx)} val={len(val_idx)}", flush=True)

# Pre-compute (label, modality) → indices for fast batch sampling
def class_mod_index(idx_list):
    out = {}
    for i in idx_list:
        _, mod, lab, _ = items[i]
        out.setdefault((int(lab), mod), []).append(i)
    return out

train_pool = class_mod_index(train_idx)
val_pool   = class_mod_index(val_idx)
modalities = ["ms", "optical", "sar"]
print(f"[pool] train classes × modalities: {len(train_pool)} keys")

def balanced_batch(pool, batch_size, rng):
    """Sample `batch_size//NUM_CLASSES` per class, equal modality share."""
    labels = sorted({lab for (lab, _) in pool.keys()})
    per_class = max(1, batch_size // max(1, len(labels)))
    batch = []
    for lab in labels:
        # try to get one of each modality, fill with any
        avail = []
        for m in modalities:
            ix = pool.get((lab, m), [])
            if ix: avail.extend(rng.choice(ix, size=min(len(ix), per_class // 3 + 1), replace=True).tolist())
        if not avail:
            # any mod
            for (l, mm), ix in pool.items():
                if l == lab: avail.extend(rng.choice(ix, size=min(len(ix), 2), replace=True).tolist())
        rng.shuffle(avail)
        batch.extend(avail[:per_class])
    if len(batch) > batch_size:
        batch = rng.choice(batch, size=batch_size, replace=False).tolist()
    return batch

# ===================== 10. TRAINING LOOP =====================
def train_v6(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, hidden=HIDDEN):
    emb = embeddings
    feat_dim = emb.shape[1]
    rng = np.random.default_rng(SEED)

    projectors = {m: ModalityProjectorV6(feat_dim, hidden, OUT_DIM).to(device)
                  for m in modalities}
    params = [p for proj in projectors.values() for p in proj.parameters()]
    crit = CombinedLoss(W_TRIPLET, W_NCE, W_CM, W_PROTO).to(device)
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    warmup_steps = 200
    total_steps = max(1, epochs * 80)
    def lr_lambda(step):
        if step < warmup_steps: return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    emb_t = torch.from_numpy(emb).to(device)

    @torch.no_grad()
    def project_one_batch(idxs):
        feats_b = emb_t[idxs]
        _, mods_list, labs_list = [], [], []
        for i in idxs:
            _, mod, lab, _ = items[i]
            mods_list.append(mod); labs_list.append(int(lab))
        outs = []
        cur = 0
        per_mod = {m: [] for m in modalities}
        for k, i in enumerate(idxs):
            per_mod[mods_list[k]].append(k)
        # forward per modality
        z = torch.zeros((len(idxs), OUT_DIM), device=device)
        for m, ix in per_mod.items():
            if not ix: continue
            sub = feats_b[ix]
            zsub = projectors[m](sub)
            for j, k in enumerate(ix):
                z[k] = zsub[j]
        return F.normalize(z, dim=-1)

    def epoch_loss(pool, n_batches=80):
        total = 0.0
        for _ in range(n_batches):
            batch = balanced_batch(pool, batch_size, rng)
            if not batch: continue
            feats_b = emb_t[batch]
            mods_list = [items[i][1] for i in batch]
            labs_list = torch.tensor([items[i][2] for i in batch], device=device)
            # forward per modality (one network call per modality per batch)
            outs_per_mod = {m: [] for m in modalities}
            for k, i in enumerate(batch):
                outs_per_mod[mods_list[k]].append(feats_b[k:k+1])
            z_chunks, lab_chunks = [], []
            for m, lst in outs_per_mod.items():
                if not lst: continue
                z_m = projectors[m](torch.cat(lst, 0))
                z_chunks.append(z_m)
                lab_chunks.append(labs_list[[i for i, mm in enumerate(mods_list) if mm == m]])
            z = F.normalize(torch.cat(z_chunks, 0), dim=-1)
            lab = torch.cat(lab_chunks, 0)
            # rebuild mods in same order as z_chunks
            mods_ordered = []
            for m, lst in outs_per_mod.items():
                mods_ordered.extend([m] * len(lst))
            loss = crit(z, lab, mods_ordered)
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step(); sched.step()
            total += loss.item()
        return total / max(1, n_batches)

    # Validation (no grad)
    @torch.no_grad()
    def val_loss():
        emb_dev = emb_t
        rng_v = np.random.default_rng(SEED + 1)
        total = 0.0; n = 0
        for _ in range(40):
            batch = balanced_batch(val_pool, batch_size, rng_v)
            if not batch: continue
            feats_b = emb_dev[batch]
            mods_list = [items[i][1] for i in batch]
            labs_list = torch.tensor([items[i][2] for i in batch], device=device)
            outs_per_mod = {m: [] for m in modalities}
            for k, i in enumerate(batch):
                outs_per_mod[mods_list[k]].append(feats_b[k:k+1])
            z_chunks, lab_chunks = [], []
            for m, lst in outs_per_mod.items():
                if not lst: continue
                z_m = projectors[m](torch.cat(lst, 0))
                z_chunks.append(z_m)
                lab_chunks.append(labs_list[[i for i, mm in enumerate(mods_list) if mm == m]])
            z = F.normalize(torch.cat(z_chunks, 0), dim=-1)
            lab = torch.cat(lab_chunks, 0)
            mods_ordered = []
            for m, lst in outs_per_mod.items():
                mods_ordered.extend([m] * len(lst))
            total += crit(z, lab, mods_ordered).item(); n += 1
        return total / max(1, n)

    best_val = float("inf"); best_state = None; best_epoch = 0
    t0 = time.time()
    for epoch in range(epochs):
        tl = epoch_loss(train_pool)
        vl = val_loss()
        print(f"  v6 epoch {epoch+1:02d}/{epochs}  train={tl:.4f}  val={vl:.4f}  lr={optim.param_groups[0]['lr']:.2e}  ({time.time()-t0:.0f}s)", flush=True)
        if vl < best_val:
            best_val = vl; best_epoch = epoch + 1
            best_state = {m: {k: v.detach().cpu().clone() for k, v in projectors[m].state_dict().items()}
                          for m in modalities}

    for m in modalities:
        projectors[m].load_state_dict(best_state[m])
    torch.save({"feat_dim": feat_dim, "hidden_dim": hidden, "out_dim": OUT_DIM,
                "modalities": modalities, "state_dict": best_state,
                "_version": "v6_dinov2", "backbone": BACKBONE_NAME,
                "best_val": best_val, "best_epoch": best_epoch,
                "epochs": epochs, "lr": lr, "batch_size": batch_size,
                "val_frac": VAL_FRAC},
               PROJECTOR_CKPT)
    print(f"[train-v6] saved best@{best_epoch}  val={best_val:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return projectors

projectors = train_v6()

# ===================== 11. BUILD GALLERY =====================
def build_gallery():
    ckpt = torch.load(PROJECTOR_CKPT, map_location="cpu", weights_only=False)
    feat_dim = ckpt["feat_dim"]; hidden = ckpt["hidden_dim"]; out_dim = ckpt["out_dim"]
    mods = ckpt["modalities"]
    projectors = {m: ModalityProjectorV6(feat_dim, hidden, out_dim).to(device) for m in mods}
    for m in mods:
        projectors[m].load_state_dict(ckpt["state_dict"][m])
        projectors[m].eval()

    z_all = np.zeros((len(items), out_dim), dtype=np.float32)
    emb_t = torch.from_numpy(embeddings).to(device)
    with torch.no_grad():
        for m in mods:
            ix = np.where(modalities_arr == m)[0]
            if len(ix) == 0: continue
            x = emb_t[ix]
            z = projectors[m](x).cpu().numpy()
            z_all[ix] = z
    norms = np.linalg.norm(z_all, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    z_all = z_all / norms

    index = faiss.IndexFlatIP(out_dim)
    index.add(z_all)
    faiss.write_index(index, FAISS_PATH)
    np.savez_compressed(META_PATH, paths=paths_arr, modalities=modalities_arr,
                        labels=labels_arr, ids=ids_arr, embeddings=z_all)
    print(f"[build] ntotal={index.ntotal}  -> {FAISS_PATH}, {META_PATH}")

build_gallery()

# ===================== 12. EVAL =====================
def evaluate():
    meta = np.load(META_PATH, allow_pickle=True)
    paths = meta["paths"]; mods = meta["modalities"]; labs = meta["labels"]; ids = meta["ids"]
    z = meta["embeddings"]
    idx = faiss.read_index(FAISS_PATH)

    SAME = [("ms","ms"),("optical","optical"),("sar","sar")]
    CROSS = [("ms","optical"),("optical","ms"),
             ("sar","optical"),("optical","sar"),
             ("ms","sar"),("sar","ms")]

    PER_CLASS = 60
    K = 10

    def eval_pair(src, tgt):
        qi_pool = np.where(mods == src)[0]
        rng = np.random.default_rng(7)
        # sample PER_CLASS queries PER class label
        labels = sorted(set(int(labs[i]) for i in qi_pool))
        qi = []
        for c in labels:
            ix = qi_pool[labs[qi_pool] == c]
            n = min(PER_CLASS, len(ix))
            qi.extend(rng.choice(ix, size=n, replace=False).tolist())
        if not qi: return None
        p5=[]; p10=[]; h5=[]; h10=[]; m5=[]; m10=[]; f5=[]; f10=[]
        rel_pool = np.where(mods == tgt)[0]
        for q in qi:
            ql = int(labs[q])
            D, I = idx.search(z[q:q+1], 100)
            top = I[0]
            rel = set(int(j) for j in rel_pool[labs[rel_pool] == ql] if j != q)
            if not rel: continue
            t5, t10 = top[:5], top[:10]
            p5.append(sum(1 for r in t5  if r in rel)/5)
            p10.append(sum(1 for r in t10 if r in rel)/10)
            h5.append(1.0 if any(r in rel for r in t5)  else 0.0)
            h10.append(1.0 if any(r in rel for r in t10) else 0.0)
            def apk(t, k):
                hits=0; s=0
                for i, r in enumerate(t[:k]):
                    if r in rel: hits += 1; s += hits/(i+1)
                return s/min(k, len(rel)) if hits else 0.0
            m5.append(apk(top, 5)); m10.append(apk(top, 10))
            def f1k(t, k):
                hit = sum(1 for r in t[:k] if r in rel)
                if hit == 0: return 0.0
                p = hit/k; r = hit/len(rel)
                return 2*p*r/(p+r)
            f5.append(f1k(top, 5)); f10.append(f1k(top, 10))
        return dict(P5=float(np.mean(p5)), P10=float(np.mean(p10)),
                    H5=float(np.mean(h5)), H10=float(np.mean(h10)),
                    M5=float(np.mean(m5)), M10=float(np.mean(m10)),
                    F5=float(np.mean(f5)), F10=float(np.mean(f10)), n=len(qi))

    same, cross = [], []
    print("\n=== V6 EVAL (60 queries/class, K=10) ===")
    for src, tgt in SAME:
        r = eval_pair(src, tgt)
        if r is None: continue
        same.append(r)
        print(f"  same   {src:8s}->{tgt:8s}  P@5={r['P5']:.4f}  P@10={r['P10']:.4f}  H@5={r['H5']:.4f}  MAP@10={r['M10']:.4f}  F1@5={r['F5']:.4f}  F1@10={r['F10']:.4f}  n={r['n']}", flush=True)
    for src, tgt in CROSS:
        r = eval_pair(src, tgt)
        if r is None: continue
        cross.append(r)
        print(f"  cross  {src:8s}->{tgt:8s}  P@5={r['P5']:.4f}  P@10={r['P10']:.4f}  H@5={r['H5']:.4f}  MAP@10={r['M10']:.4f}  F1@5={r['F5']:.4f}  F1@10={r['F10']:.4f}  n={r['n']}", flush=True)

    def avg(lst, k):
        return float(np.mean([r[k] for r in lst])) if lst else 0.0

    summary = dict(
        same_modal  = dict(P5=avg(same, 'P5'),  P10=avg(same, 'P10'),
                           H5=avg(same, 'H5'),  H10=avg(same, 'H10'),
                           F5=avg(same, 'F5'),  F10=avg(same, 'F10'),
                           MAP5=avg(same, 'M5'), MAP10=avg(same, 'M10')),
        cross_modal = dict(P5=avg(cross, 'P5'), P10=avg(cross, 'P10'),
                           H5=avg(cross, 'H5'), H10=avg(cross, 'H10'),
                           F5=avg(cross, 'F5'), F10=avg(cross, 'F10'),
                           MAP5=avg(cross, 'M5'), MAP10=avg(cross, 'M10')),
        per_scenario = {f"same:{s}->{t}": eval_pair(s, t) for s, t in SAME} |
                       {f"cross:{s}->{t}": eval_pair(s, t) for s, t in CROSS},
    )
    with open(EVAL_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== SUMMARY ===")
    print(f"  Same-modal   P@5={summary['same_modal']['P5']:.4f}  P@10={summary['same_modal']['P10']:.4f}  H@5={summary['same_modal']['H5']:.4f}  F1@5={summary['same_modal']['F5']:.4f}")
    print(f"  Cross-modal  P@5={summary['cross_modal']['P5']:.4f}  P@10={summary['cross_modal']['P10']:.4f}  H@5={summary['cross_modal']['H5']:.4f}  F1@5={summary['cross_modal']['F5']:.4f}")
    print(f"[eval] saved -> {EVAL_JSON}")

evaluate()

# ===================== 13. ZIP + DOWNLOAD =====================
!zip -j -q {ZIP_OUTPUT} {PROJECTOR_CKPT} {FAISS_PATH} {META_PATH} {EVAL_JSON} {FEATURES_PATH}

print(f"\n[done] v6 artifacts ready at {ZIP_OUTPUT}")
_size = os.path.getsize(ZIP_OUTPUT) / (1024 * 1024); print(f"[done] zip size: {_size:.1f} MB")
from google.colab import files
files.download(ZIP_OUTPUT)