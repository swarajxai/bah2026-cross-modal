"""
V5 enhanced training - sharper contrastive learning without mixup.

Lessons from v4 failure:
  - Mixup blurs discriminative features -> regression
  - Need harder negatives but not feature-space augmentation

V5 strategy:
  - Phase 1 (10 epochs): InfoNCE with HARD NEGATIVE MINING within batch
  - Phase 2 (15 epochs): Triplet + InfoNCE + CrossModal + Prototype (no mixup)
  - Cosine LR + warmup
  - hidden_dim=1024 (proven better than 768)
  - Label smoothing for the InfoNCE denominator (reduces overconfidence)
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.heads import ModalityProjectorV1
from src.losses import TripletLoss, InfoNCELoss, CrossModalAlignmentLoss, ClassPrototypeLoss


class HardNegativeMiner:
    """Mine the hardest negatives per anchor: same-class different-mod first."""

    def __init__(self, lab, mods, modalities):
        self.lab = lab
        self.mods = mods
        self.modalities = modalities
        # indices of (class, modality) pairs
        self.cls_mod_idx = {}
        for i, (l, m) in enumerate(zip(lab, mods)):
            self.cls_mod_idx.setdefault((int(l), m), []).append(i)

    def hard_negatives(self, anchor_idx, k=8):
        """Return indices of hardest negatives for a given anchor."""
        a_lab = int(self.lab[anchor_idx])
        a_mod = self.mods[anchor_idx]
        # get same-class-different-mod first
        out = []
        for m in self.modalities:
            if m == a_mod: continue
            pool = self.cls_mod_idx.get((a_lab, m), [])
            if pool:
                # take up to 2 from each other modality, same class
                n = min(2, len(pool))
                idx = np.random.choice(len(pool), size=n, replace=False)
                out.extend([pool[i] for i in idx])
        # fill remaining with random negatives (different class)
        while len(out) < k:
            j = np.random.randint(0, len(self.lab))
            if int(self.lab[j]) != a_lab and j not in out:
                out.append(j)
        return out[:k]


def info_nce_with_mining(z, lab, mods, miner, temperature=0.07, hard_neg_weight=2.0):
    """InfoNCE with hard negative mining — harder negs have higher weight."""
    sim = z @ z.t() / temperature
    N = z.size(0)
    mask_self = torch.eye(N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask_self, -1e9)

    labels_eq = lab.unsqueeze(0) == lab.unsqueeze(1)
    pos_mask = labels_eq & ~mask_self

    # Weight matrix: 1.0 for random negatives, hard_neg_weight for hard ones
    weights = torch.ones_like(sim)
    mod_eq = torch.tensor([mods[i] == mods[j]
                            for i in range(N) for j in range(N)],
                           device=z.device).view(N, N)
    # for each anchor, mark hard negatives (same class, different mod)
    for i in range(N):
        hn = miner.hard_negatives(i, k=8)
        for j in hn:
            if j < N:
                weights[i, j] = hard_neg_weight

    # weighted InfoNCE
    exp_sim = torch.exp(sim) * weights
    denom = exp_sim.sum(dim=-1)
    log_prob = sim - torch.log(denom.unsqueeze(-1) + 1e-12)
    n_pos = pos_mask.sum(dim=-1).clamp(min=1)
    loss = -(log_prob * pos_mask).sum(dim=-1) / n_pos
    return loss.mean()


def train(features_npz: str, output_ckpt: str,
          out_dim: int = 256, hidden_dim: int = 1024,
          epochs_phase1: int = 12, epochs_phase2: int = 15,
          batch_size: int = 256, lr: float = 1.5e-3,
          temperature: float = 0.06, margin: float = 0.3,
          dropout: float = 0.1, seed: int = 42,
          hard_neg_weight: float = 2.0):
    print(f"[train-v5] loading features from {features_npz}")
    data = np.load(features_npz, allow_pickle=True)
    emb = data["embeddings"].astype(np.float32)
    lab = data["labels"].astype(np.int64)
    mods = data["modalities"]
    n_classes = int(lab.max() + 1)
    feat_dim = emb.shape[1]
    print(f"[train-v5] emb shape: {emb.shape}, classes: {n_classes}, mods: {set(mods)}")

    modalities = ["ms", "optical", "sar"]
    modality_idx = {m: np.where(mods == m)[0] for m in modalities}
    for m, ix in modality_idx.items():
        print(f"  {m}: {len(ix)} samples")

    emb_t = torch.from_numpy(emb)
    lab_t = torch.from_numpy(lab)
    miner = HardNegativeMiner(lab, mods, modalities)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    projectors = {m: ModalityProjectorV1(feat_dim, hidden_dim, out_dim, dropout=dropout)
                  for m in modalities}
    params = [p for proj in projectors.values() for p in proj.parameters()]

    triplets_loss_fn = TripletLoss(margin=margin)
    cm_loss_fn = CrossModalAlignmentLoss(temperature=temperature)
    proto_loss_fn = ClassPrototypeLoss()

    best_loss = float("inf")
    best_state = None
    t0_total = time.time()

    # === PHASE 1: hard-negative-mined InfoNCE (12 epochs) ===
    print(f"\n[train-v5] === PHASE 1: HARD-NEG INFO NCE ({epochs_phase1} epochs) ===")
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    p1_total = epochs_phase1
    def p1_lr(epoch):
        if epoch < 2: return (epoch + 1) / 2
        return 0.5 * (1.0 + np.cos(np.pi * (epoch - 2) / max(1, p1_total - 2)))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, p1_lr)

    for epoch in range(epochs_phase1):
        orders = {m: rng.permutation(modality_idx[m]) for m in modalities}
        per_mod = max(1, batch_size // 3)
        sizes = [len(modality_idx[m]) for m in modalities]
        steps = min(sizes) // per_mod
        if steps == 0: steps = 1; per_mod = min(sizes)

        epoch_loss = 0.0; n_batches = 0
        ep_t0 = time.time()
        for step in range(steps):
            batch_feats, batch_labels, batch_mods = [], [], []
            for m in modalities:
                ids = orders[m][step * per_mod:(step + 1) * per_mod]
                if len(ids) == 0: continue
                batch_feats.append(emb_t[ids])
                batch_labels.append(lab_t[ids])
                batch_mods.extend([m] * len(ids))
            if not batch_feats: continue
            feats_b   = torch.cat(batch_feats, 0)
            labels_b  = torch.cat(batch_labels, 0)
            outs = [projectors[m](feats_b[i:i+1]) for i, m in enumerate(batch_mods)]
            z = torch.cat(outs, 0)

            # Phase 1: InfoNCE with hard-negative mining + small Triplet
            ln = info_nce_with_mining(z, labels_b, batch_mods, miner,
                                       temperature=temperature,
                                       hard_neg_weight=hard_neg_weight)
            lt = triplets_loss_fn(z, labels_b) * 0.3
            loss = ln + lt

            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            epoch_loss += loss.item(); n_batches += 1

        sched.step()
        avg = epoch_loss / max(1, n_batches)
        ep_dt = time.time() - ep_t0
        print(f"  P1 [epoch {epoch+1:02d}/{epochs_phase1}] loss={avg:.4f}  lr={optim.param_groups[0]['lr']:.2e}  ({ep_dt:.1f}s)")
        if avg < best_loss:
            best_loss = avg
            best_state = {m: projectors[m].state_dict() for m in modalities}

    # === PHASE 2: refine with all losses (no mixup) ===
    print(f"\n[train-v5] === PHASE 2: ALL LOSSES ({epochs_phase2} epochs) ===")
    optim = torch.optim.AdamW(params, lr=lr * 0.3, weight_decay=1e-5)
    nce_loss_fn = InfoNCELoss(temperature=temperature)
    p2_total = epochs_phase2
    def p2_lr(epoch):
        return 0.5 * (1.0 + np.cos(np.pi * epoch / p2_total))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, p2_lr)

    for epoch in range(epochs_phase2):
        orders = {m: rng.permutation(modality_idx[m]) for m in modalities}
        per_mod = max(1, batch_size // 3)
        sizes = [len(modality_idx[m]) for m in modalities]
        steps = min(sizes) // per_mod
        if steps == 0: steps = 1; per_mod = min(sizes)

        epoch_loss = 0.0; n_batches = 0
        ep_t0 = time.time()
        for step in range(steps):
            batch_feats, batch_labels, batch_mods = [], [], []
            for m in modalities:
                ids = orders[m][step * per_mod:(step + 1) * per_mod]
                if len(ids) == 0: continue
                batch_feats.append(emb_t[ids])
                batch_labels.append(lab_t[ids])
                batch_mods.extend([m] * len(ids))
            if not batch_feats: continue
            feats_b   = torch.cat(batch_feats, 0)
            labels_b  = torch.cat(batch_labels, 0)
            outs = [projectors[m](feats_b[i:i+1]) for i, m in enumerate(batch_mods)]
            z = torch.cat(outs, 0)

            # Phase 2: all 4 losses, no mixup, no hard-negative mining
            lt = triplets_loss_fn(z, labels_b)
            ln = nce_loss_fn(z, labels_b)
            lc = cm_loss_fn(z, labels_b, batch_mods)
            lp = proto_loss_fn(z, labels_b)
            loss = 0.30 * lt + 0.35 * ln + 0.20 * lc + 0.15 * lp

            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            epoch_loss += loss.item(); n_batches += 1

        sched.step()
        avg = epoch_loss / max(1, n_batches)
        ep_dt = time.time() - ep_t0
        print(f"  P2 [epoch {epoch+1:02d}/{epochs_phase2}] loss={avg:.4f}  lr={optim.param_groups[0]['lr']:.2e}  ({ep_dt:.1f}s)")
        if avg < best_loss:
            best_loss = avg
            best_state = {m: projectors[m].state_dict() for m in modalities}

    total_dt = time.time() - t0_total
    os.makedirs(os.path.dirname(output_ckpt), exist_ok=True)
    torch.save({"feat_dim": feat_dim, "hidden_dim": hidden_dim, "out_dim": out_dim,
                "modalities": modalities, "state_dict": best_state,
                "_version": "v5_hard_neg"}, output_ckpt)
    print(f"\n[train-v5] saved -> {output_ckpt}  (best_loss={best_loss:.4f}, total {total_dt:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "outputs" / "features.npz"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "projector.pt"))
    ap.add_argument("--epochs_phase1", type=int, default=12)
    ap.add_argument("--epochs_phase2", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--hard_neg_weight", type=float, default=2.0)
    args = ap.parse_args()
    train(args.features, args.out,
          epochs_phase1=args.epochs_phase1,
          epochs_phase2=args.epochs_phase2,
          batch_size=args.batch_size,
          hidden_dim=args.hidden, lr=args.lr,
          hard_neg_weight=args.hard_neg_weight)
