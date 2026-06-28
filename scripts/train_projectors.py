"""
Train modality-specific projection heads on top of cached features.

Reads outputs/features.npz (backbone features, labels, modalities),
learns a small MLP per modality that projects into a shared 256-D space.
Loss: Triplet (hard mining) + InfoNCE combined.

Saves: outputs/projector.pt  (a state_dict mapping modality -> projector)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.heads import ModalityProjector
from src.losses import TripletLoss, InfoNCELoss


def train(features_npz: str, output_ckpt: str,
          out_dim: int = 256, hidden_dim: int = 512,
          epochs: int = 12, batch_size: int = 256, lr: float = 1e-3,
          temperature: float = 0.07, triplet_weight: float = 0.5,
          nce_weight: float = 0.5, seed: int = 42):
    print(f"[train] loading features from {features_npz}")
    data = np.load(features_npz, allow_pickle=True)
    emb = data["embeddings"].astype(np.float32)
    lab = data["labels"].astype(np.int64)
    mods = data["modalities"]
    print(f"[train] emb shape: {emb.shape}, classes: {len(np.unique(lab))}, mods: {set(mods)}")

    feat_dim = emb.shape[1]
    device = "cpu"

    # one projector per modality
    modalities = ["ms", "optical", "sar"]
    projectors = {m: ModalityProjector(feat_dim, hidden_dim, out_dim) for m in modalities}
    optim = torch.optim.AdamW(
        [p for proj in projectors.values() for p in proj.parameters()],
        lr=lr, weight_decay=1e-4,
    )

    triplet = TripletLoss(margin=0.2)
    nce = InfoNCELoss(temperature=temperature)

    # Build per-modality arrays for sampling
    idx_by_mod = {m: np.where(mods == m)[0] for m in modalities}
    for m, ix in idx_by_mod.items():
        print(f"  {m}: {len(ix)} samples")

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    N = emb.shape[0]
    feat_t = torch.from_numpy(emb)
    lab_t = torch.from_numpy(lab)

    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        # build a mixed batch: equal share of each modality
        per_mod = max(1, batch_size // 3)
        sizes = [len(idx_by_mod[m]) for m in modalities]
        steps = min(sizes) // per_mod
        if steps == 0:
            steps = 1
            per_mod = min(sizes)
        # shuffle within modality
        orders = {m: rng.permutation(idx_by_mod[m]) for m in modalities}

        epoch_loss = 0.0
        n_batches = 0
        for step in range(steps):
            batch_feats, batch_labels, batch_mods = [], [], []
            for k, m in enumerate(modalities):
                ids = orders[m][step * per_mod:(step + 1) * per_mod]
                if len(ids) == 0:
                    continue
                batch_feats.append(feat_t[ids])
                batch_labels.append(lab_t[ids])
                batch_mods.extend([m] * len(ids))
            if not batch_feats:
                continue
            feats_b = torch.cat(batch_feats, 0).to(device)
            labels_b = torch.cat(batch_labels, 0).to(device)
            # encode
            outs = [projectors[m](feats_b[i:i+1]) for i, m in enumerate(batch_mods)]
            z = torch.cat(outs, 0)
            # combined loss
            lt = triplet(z, labels_b)
            ln = nce(z, labels_b)
            loss = triplet_weight * lt + nce_weight * ln

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for proj in projectors.values() for p in proj.parameters()], 1.0)
            optim.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(1, n_batches)
        print(f"[epoch {epoch+1:02d}/{epochs}] loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            best_state = {m: projectors[m].state_dict() for m in modalities}

    # save best
    torch.save({"feat_dim": feat_dim, "hidden_dim": hidden_dim, "out_dim": out_dim,
                "modalities": modalities, "state_dict": best_state}, output_ckpt)
    print(f"[train] saved -> {output_ckpt} (best_loss={best_loss:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "outputs" / "features.npz"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "projector.pt"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    train(args.features, args.out, epochs=args.epochs, batch_size=args.batch_size)