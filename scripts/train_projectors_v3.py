"""
HYBRID v3: TTA features (richer) + v1-style projector (proven best on CPU).

Hypothesis: TTA features give better class-discriminative signal; v1-style
MLP avoids the BN eval-mode issues. Combine for higher P@K.

Loss: same as v2 (Triplet + InfoNCE + CrossModal + Prototype) but with
simpler v1-style projector architecture.

Trains on the TTA-extracted features.npz with 25 epochs and cosine LR.
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
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.heads import ModalityProjectorV1
from src.losses import CombinedLoss


def train(features_npz: str, output_ckpt: str,
          out_dim: int = 256, hidden_dim: int = 768,
          epochs: int = 25, batch_size: int = 256, lr: float = 2e-3,
          temperature: float = 0.07, margin: float = 0.2,
          triplet_w: float = 0.3, nce_w: float = 0.4,
          cm_w: float = 0.2, proto_w: float = 0.1,
          seed: int = 42, dropout: float = 0.1,
          warmup_epochs: int = 2):
    print(f"[train-v3] loading features from {features_npz}")
    data = np.load(features_npz, allow_pickle=True)
    emb = data["embeddings"].astype(np.float32)
    lab = data["labels"].astype(np.int64)
    mods = data["modalities"]
    n_classes = int(lab.max() + 1)
    feat_dim = emb.shape[1]
    print(f"[train-v3] emb shape: {emb.shape}, classes: {n_classes}, mods: {set(mods)}")
    print(f"[train-v3] hidden_dim={hidden_dim}, dropout={dropout}, lr={lr}, epochs={epochs}")

    modalities = ["ms", "optical", "sar"]
    # v1-style projector (2-layer MLP, no BN), but with bigger hidden_dim
    projectors = {m: ModalityProjectorV1(feat_dim, hidden_dim, out_dim, dropout=dropout)
                  for m in modalities}
    params = [p for proj in projectors.values() for p in proj.parameters()]
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    crit = CombinedLoss(triplet_w=triplet_w, nce_w=nce_w, cm_w=cm_w, proto_w=proto_w,
                        temperature=temperature, triplet_margin=margin)

    idx_by_mod = {m: np.where(mods == m)[0] for m in modalities}
    for m, ix in idx_by_mod.items():
        print(f"  {m}: {len(ix)} samples")

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    feat_t = torch.from_numpy(emb)
    lab_t = torch.from_numpy(lab)

    best_loss = float("inf")
    best_state = None

    print(f"\n[train-v3] starting {epochs} epochs ...\n")
    t0_total = time.time()
    for epoch in range(epochs):
        orders = {m: rng.permutation(idx_by_mod[m]) for m in modalities}
        per_mod = max(1, batch_size // 3)
        sizes = [len(idx_by_mod[m]) for m in modalities]
        steps = min(sizes) // per_mod
        if steps == 0:
            steps = 1; per_mod = min(sizes)

        epoch_loss = 0.0
        n_batches = 0
        ep_t0 = time.time()
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
            outs = [projectors[m](feats_b[i:i+1]) for i, m in enumerate(batch_mods)]
            z = torch.cat(outs, 0)
            loss = crit(z, labels_b, batch_mods)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            epoch_loss += loss.item()
            n_batches += 1

        sched.step()
        avg = epoch_loss / max(1, n_batches)
        ep_dt = time.time() - ep_t0
        cur_lr = optim.param_groups[0]["lr"]
        print(f"  [epoch {epoch+1:02d}/{epochs}] loss={avg:.4f}  lr={cur_lr:.2e}  ({ep_dt:.1f}s)")
        if avg < best_loss:
            best_loss = avg
            best_state = {m: projectors[m].state_dict() for m in modalities}

    total_dt = time.time() - t0_total
    os.makedirs(os.path.dirname(output_ckpt), exist_ok=True)
    # save with the SAME structure as v1 (no n_classes key) so the server's
    # auto-detect picks v1 branch (which is what we want for this projector)
    torch.save({"feat_dim": feat_dim, "hidden_dim": hidden_dim, "out_dim": out_dim,
                "modalities": modalities, "state_dict": best_state,
                "_version": "v3_hybrid"}, output_ckpt)
    print(f"\n[train-v3] saved -> {output_ckpt}  (best_loss={best_loss:.4f}, total {total_dt:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "outputs" / "features.npz"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "projector.pt"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    train(args.features, args.out,
          epochs=args.epochs, batch_size=args.batch_size,
          hidden_dim=args.hidden, lr=args.lr)