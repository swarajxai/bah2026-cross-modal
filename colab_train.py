"""
Colab-friendly full-pipeline training script.
================================================

Single-script end-to-end pipeline, optimized for a free T4 GPU (15 GB VRAM):

  1. Extract DINOv2-Base (or other) features for all gallery images
  2. Train v2 projection heads (deeper MLP + class-conditional norm)
     with v2 combined loss (Triplet + InfoNCE + CrossModal + Prototype)
  3. Build the FAISS gallery index
  4. Evaluate F1@5, F1@10, P@K, R@K, HitRate@K, MAP@K, latency

Usage in Colab (T4 GPU):
    !python colab_train.py
    # or with options:
    !python colab_train.py --backbone dinov2_base_518 --epochs 20 --batch 16

After the run, download these files and replace the local ones:
  outputs/features.npz
  outputs/projector.pt
  outputs/gallery.faiss
  outputs/gallery_meta.npz
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import faiss
from PIL import Image
import cv2
import tifffile
from tqdm import tqdm

# Local imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.backbones import build_backbone
from src.heads import ModalityProjector
from src.losses import CombinedLoss
from src.dataset import (
    _read_image, _preprocess, CLASS_TO_IDX, ALL_CLASSES, discover_datasets,
    IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE,
)


# ====================================================================
# 1) EXTRACT BACKBONE FEATURES  (on GPU, with mixed precision)
# ====================================================================
class ImgDataset(Dataset):
    def __init__(self, items, img_size):
        self.items = items
        self.img_size = img_size
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, modality, label, sid = self.items[idx]
        try:
            rgb = _read_image(path, modality)
        except Exception:
            rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        # resize to backbone's expected size
        if rgb.shape[0] != self.img_size or rgb.shape[1] != self.img_size:
            rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        x = rgb.astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1)
        x = (x - self.mean) / self.std
        return x, label, modality, path, sid


def extract_features(base_dir, backbone_name, batch_size, num_workers, out_path):
    print(f"\n========== [1/4] EXTRACT FEATURES ({backbone_name}) ==========")
    info = discover_datasets(base_dir)

    # build flat items list
    items = []
    for s in info["eurosat"]:
        lab = s["label"]
        if s.get("ms_path"):      items.append((s["ms_path"], "ms", lab, s["id"]))
        if s.get("optical_path"): items.append((s["optical_path"], "optical", lab, s["id"]))
    for s in info["sentinel_paired"]:
        lab = s["label"]
        if s.get("sar_path"):     items.append((s["sar_path"], "sar", lab, s["id"]))
        if s.get("optical_path"): items.append((s["optical_path"], "optical", lab, s["id"]))
    for s in info["sentinel_unpaired"]:
        lab = s["label"]
        if s.get("sar_path"):     items.append((s["sar_path"], "sar", lab, s["id"]))
        if s.get("optical_path"): items.append((s["optical_path"], "optical", lab, s["id"]))
    items = [x for x in items if x[2] >= 0]
    print(f"  total items: {len(items)}")

    backbone = build_backbone(backbone_name).cuda().eval()
    img_size = backbone.img_size

    ds = ImgDataset(items, img_size=img_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)

    feats, labels, mods, paths, ids = [], [], [], [], []
    t0 = time.time()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        for x, y, m, p, sid in tqdm(dl, ncols=100):
            x = x.cuda(non_blocking=True)
            z = backbone(x).float().cpu().numpy()
            feats.append(z)
            labels.append(y.numpy())
            mods.extend(m)
            paths.extend(p)
            ids.extend(sid)
    feats = np.concatenate(feats, 0).astype(np.float32)
    labels = np.concatenate(labels, 0).astype(np.int64)
    mods = np.array(mods)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, embeddings=feats, labels=labels,
                        modalities=mods, paths=np.array(paths), ids=np.array(ids))
    print(f"  saved -> {out_path}  ({feats.shape}, {time.time()-t0:.1f}s)")


# ====================================================================
# 2) TRAIN V2 PROJECTION HEADS  (deeper, better loss, GPU)
# ====================================================================
def train_projectors(features_npz, projector_out,
                     out_dim=256, hidden_dim=1024,
                     epochs=20, batch_size=256, lr=2e-3,
                     triplet_w=0.3, nce_w=0.4, cm_w=0.2, proto_w=0.1,
                     temperature=0.07, margin=0.2, seed=42):
    print(f"\n========== [2/4] TRAIN V2 PROJECTORS (GPU) ==========")
    data = np.load(features_npz, allow_pickle=True)
    emb  = data["embeddings"].astype(np.float32)
    lab  = data["labels"].astype(np.int64)
    mods = data["modalities"]
    n_classes = int(lab.max() + 1)
    feat_dim = emb.shape[1]
    print(f"  emb shape: {emb.shape}, n_classes: {n_classes}")

    modalities = ["ms", "optical", "sar"]
    projectors = {m: ModalityProjector(feat_dim, hidden_dim, out_dim,
                                       n_classes=n_classes, dropout=0.2).cuda()
                  for m in modalities}
    optim = torch.optim.AdamW(
        [p for proj in projectors.values() for p in proj.parameters()],
        lr=lr, weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    crit = CombinedLoss(triplet_w=triplet_w, nce_w=nce_w, cm_w=cm_w, proto_w=proto_w,
                        temperature=temperature, triplet_margin=margin)

    # per-modality index arrays
    idx_by_mod = {m: np.where(mods == m)[0] for m in modalities}
    for m, ix in idx_by_mod.items():
        print(f"  {m}: {len(ix)} samples")

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    feat_t = torch.from_numpy(emb).cuda()
    lab_t  = torch.from_numpy(lab).cuda()

    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        orders = {m: rng.permutation(idx_by_mod[m]) for m in modalities}
        per_mod = max(1, batch_size // 3)
        sizes = [len(idx_by_mod[m]) for m in modalities]
        steps = min(sizes) // per_mod
        if steps == 0:
            steps = 1; per_mod = min(sizes)

        epoch_loss = 0.0
        n_batches = 0
        for step in range(steps):
            batch_feats, batch_labels, batch_mods = [], [], []
            for m in modalities:
                ids = orders[m][step * per_mod:(step + 1) * per_mod]
                if len(ids) == 0: continue
                batch_feats.append(feat_t[ids])
                batch_labels.append(lab_t[ids])
                batch_mods.extend([m] * len(ids))
            if not batch_feats: continue
            feats_b   = torch.cat(batch_feats, 0)
            labels_b  = torch.cat(batch_labels, 0)
            # forward through each projector (passing class_id for class-cond norm)
            outs = [projectors[m](feats_b[i:i+1], labels_b[i:i+1])
                    for i, m in enumerate(batch_mods)]
            z = torch.cat(outs, 0)
            loss = crit(z, labels_b, batch_mods)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for proj in projectors.values() for p in proj.parameters()], 1.0)
            optim.step()
            epoch_loss += loss.item()
            n_batches += 1

        sched.step()
        avg = epoch_loss / max(1, n_batches)
        print(f"  [epoch {epoch+1:02d}/{epochs}] loss={avg:.4f}  lr={optim.param_groups[0]['lr']:.2e}")
        if avg < best_loss:
            best_loss = avg
            best_state = {m: projectors[m].state_dict() for m in modalities}

    os.makedirs(os.path.dirname(projector_out), exist_ok=True)
    torch.save({"feat_dim": feat_dim, "hidden_dim": hidden_dim, "out_dim": out_dim,
                "modalities": modalities, "state_dict": best_state,
                "n_classes": n_classes}, projector_out)
    print(f"  saved -> {projector_out}  (best_loss={best_loss:.4f})")


# ====================================================================
# 3) BUILD FAISS INDEX
# ====================================================================
def build_index(features_npz, projector_ckpt, faiss_path, meta_path):
    print(f"\n========== [3/4] BUILD FAISS INDEX ==========")
    data = np.load(features_npz, allow_pickle=True)
    emb  = data["embeddings"].astype(np.float32)
    lab  = data["labels"].astype(np.int64)
    mods = data["modalities"]
    paths = data["paths"]
    ids   = data["ids"]

    ckpt = torch.load(projector_ckpt, map_location="cpu", weights_only=False)
    feat_dim = ckpt["feat_dim"]; hidden_dim = ckpt["hidden_dim"]; out_dim = ckpt["out_dim"]
    n_classes = ckpt.get("n_classes", 14)
    modalities = ckpt["modalities"]
    print(f"  feat_dim={feat_dim}, out_dim={out_dim}, n_classes={n_classes}")

    projectors = {m: ModalityProjector(feat_dim, hidden_dim, out_dim,
                                       n_classes=n_classes).cuda().eval()
                  for m in modalities}
    for m in modalities:
        projectors[m].load_state_dict(ckpt["state_dict"][m])

    z_all = np.zeros((emb.shape[0], out_dim), dtype=np.float32)
    with torch.inference_mode():
        for m in modalities:
            ix = np.where(mods == m)[0]
            if len(ix) == 0: continue
            x = torch.from_numpy(emb[ix]).cuda()
            z = projectors[m](x).cpu().numpy()
            z_all[ix] = z
    norms = np.linalg.norm(z_all, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    z_all = z_all / norms

    index = faiss.IndexFlatIP(out_dim)
    index.add(z_all)
    print(f"  gallery size: {index.ntotal}")
    faiss.write_index(index, faiss_path)
    np.savez_compressed(meta_path, paths=paths, modalities=mods,
                        labels=lab, ids=ids, embeddings=z_all)
    print(f"  saved -> {faiss_path} and {meta_path}")


# ====================================================================
# 4) EVALUATE
# ====================================================================
def _ap_at_k(is_correct, K):
    """MAP@K averaged over queries."""
    aps = []
    for row in is_correct:
        if not row.any():
            aps.append(0.0); continue
        cum_correct = 0
        ap = 0.0
        for r in range(K):
            if row[r]:
                cum_correct += 1
                ap += cum_correct / (r + 1)
        aps.append(ap / max(cum_correct, 1))
    return float(np.mean(aps))


def evaluate(features_npz, projector_ckpt, faiss_path, meta_path, n_samples_per_class=20):
    print(f"\n========== [4/4] EVALUATE ==========")
    data = np.load(features_npz, allow_pickle=True)
    emb  = data["embeddings"].astype(np.float32)
    lab  = data["labels"].astype(np.int64)
    mods = data["modalities"]

    index = faiss.read_index(faiss_path)
    meta  = np.load(meta_path, allow_pickle=True)
    gallery_mods = meta["modalities"]; gallery_lab = meta["labels"]

    ckpt = torch.load(projector_ckpt, map_location="cpu", weights_only=False)
    feat_dim = ckpt["feat_dim"]; hidden_dim = ckpt["hidden_dim"]; out_dim = ckpt["out_dim"]
    n_classes = ckpt.get("n_classes", 14)
    modalities = ckpt["modalities"]
    projectors = {m: ModalityProjector(feat_dim, hidden_dim, out_dim,
                                       n_classes=n_classes).cuda().eval()
                  for m in modalities}
    for m in modalities:
        projectors[m].load_state_dict(ckpt["state_dict"][m])

    rng = np.random.default_rng(0)
    metrics = {}
    K_list = [5, 10]

    pairs = [
        ("ms", "ms"), ("optical", "optical"), ("sar", "sar"),
        ("ms", "optical"), ("ms", "sar"), ("optical", "sar"),
    ]

    for src, tgt in pairs:
        src_idx = np.where(mods == src)[0]
        lab_arr = lab[src_idx]
        chosen = []
        for c in np.unique(lab_arr):
            ix = src_idx[lab_arr == c]
            k = min(n_samples_per_class, len(ix))
            if k > 0: chosen.extend(rng.choice(ix, size=k, replace=False))
        if not chosen: continue

        q_feats = torch.from_numpy(emb[chosen]).cuda()
        with torch.inference_mode():
            z_q = projectors[src](q_feats).cpu().numpy()
        norms = np.linalg.norm(z_q, axis=1, keepdims=True)
        z_q = z_q / (norms + 1e-12)

        t0 = time.time()
        D, I = index.search(z_q.astype(np.float32), max(K_list))
        dt_ms = (time.time() - t0) * 1000 / len(chosen)

        labels_q = lab[chosen]
        for K in K_list:
            retrieved_lab = gallery_lab[I[:, :K]]
            retrieved_mod = gallery_mods[I[:, :K]]
            same_mod = (retrieved_mod == tgt)
            is_correct = (retrieved_lab == labels_q[:, None]) & same_mod
            hit = same_mod.any(axis=1)
            gt_counts = np.array([((gallery_lab == q) & (gallery_mods == tgt)).sum()
                                  for q in labels_q])
            p_at_k = is_correct.sum(axis=1) / K
            r_at_k = is_correct.sum(axis=1) / np.maximum(gt_counts, 1)
            f1_at_k = 2 * p_at_k * r_at_k / np.maximum(p_at_k + r_at_k, 1e-9)
            hit_rate = hit.mean()
            map_k = _ap_at_k(is_correct, K)

            key = f"{src}->{tgt}@K={K}"
            metrics[key] = {
                "P": float(p_at_k.mean()),
                "R": float(r_at_k.mean()),
                "F1": float(f1_at_k.mean()),
                "HitRate": float(hit_rate),
                "MAP": float(map_k),
                "n": len(chosen),
            }

    print(f"\n  PER-PAIR METRICS (n_samples_per_class={n_samples_per_class}):")
    print(f"  {'PAIR':<25} {'K':>3} {'P':>6} {'R':>6} {'F1':>6} {'HitR':>6} {'MAP':>6}")
    for key, m in metrics.items():
        src, rest = key.split("->"); tgt, k = rest.split("@K=")
        print(f"  {src+'->'+tgt:<25} {k:>3} {m['P']:>6.3f} {m['R']:>6.3f} {m['F1']:>6.3f} {m['HitRate']:>6.3f} {m['MAP']:>6.3f}")

    # aggregate
    print()
    for K in K_list:
        for split in ["same", "cross"]:
            sel = [k for k in metrics if f"@K={K}" in k]
            if split == "same":
                sel = [k for k in sel if k.split("->")[0] == k.split("->")[1].split("@")[0]]
            else:
                sel = [k for k in sel if k.split("->")[0] != k.split("->")[1].split("@")[0]]
            if not sel: continue
            f1_avg = np.mean([metrics[k]["F1"] for k in sel])
            p_avg  = np.mean([metrics[k]["P"] for k in sel])
            r_avg  = np.mean([metrics[k]["R"] for k in sel])
            hr_avg = np.mean([metrics[k]["HitRate"] for k in sel])
            print(f"  === {split.upper()}-MODAL K={K}: F1={f1_avg:.4f}  P={p_avg:.4f}  R={r_avg:.4f}  HitRate={hr_avg:.4f}")

    print(f"\n  Avg retrieval time per query (K={max(K_list)}): {dt_ms:.2f} ms")
    return metrics


# ====================================================================
# MAIN
# ====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="/content/drive/MyDrive/BAH2026",
                    help="Path to dataset root (contains EuroSAT/, Sentinel/)")
    ap.add_argument("--out_dir", default=str(ROOT / "outputs"))
    ap.add_argument("--backbone", default="dinov2_base_518",
                    help="dinov2_base_518 | dinov2_base_224 | dinov2_small_224 | resnet50")
    ap.add_argument("--batch_extract", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_train", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--n_samples_per_class", type=int, default=20)
    ap.add_argument("--skip_extract", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_index", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    features_npz   = str(out_dir / "features.npz")
    projector_ckpt = str(out_dir / "projector.pt")
    faiss_path     = str(out_dir / "gallery.faiss")
    meta_path      = str(out_dir / "gallery_meta.npz")

    if not args.skip_extract:
        extract_features(args.base_dir, args.backbone,
                         args.batch_extract, args.num_workers, features_npz)
    if not args.skip_train:
        train_projectors(features_npz, projector_ckpt,
                         out_dim=256, hidden_dim=args.hidden,
                         epochs=args.epochs, batch_size=args.batch_train,
                         lr=args.lr)
    if not args.skip_index:
        build_index(features_npz, projector_ckpt, faiss_path, meta_path)
    if not args.skip_eval:
        evaluate(features_npz, projector_ckpt, faiss_path, meta_path,
                 n_samples_per_class=args.n_samples_per_class)

    print("\n========== DONE ==========")
    print("Download these files from this folder to your laptop's outputs/:")
    for f in ["features.npz", "projector.pt", "gallery.faiss", "gallery_meta.npz"]:
        print(f"  - {f}")


if __name__ == "__main__":
    main()