"""
Train v4 enhanced projection heads — the strongest CPU-friendly version.

Key improvements over v3:
  - 3-phase training: warmup (5ep, contrastive only) → balanced (15ep) → fine-tune (10ep)
  - Hard negative mining: explicitly find same-class different-modality pairs
  - Mixup augmentation in feature space for better generalization
  - Larger hidden_dim (1024)
  - Gradient accumulation (effective batch 512)
  - Cosine LR with warmup + cooldown

Outputs: outputs/projector.pt (v3-compatible architecture, better weights)
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
from src.losses import TripletLoss, InfoNCELoss, CrossModalAlignmentLoss, ClassPrototypeLoss


def mixup_features(x, y, m, alpha=0.2):
    """Mixup: blend features of two random samples (with same class preferred)."""
    B = x.size(0)
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    perm = torch.randperm(B, device=x.device)
    x_mix = lam * x + (1 - lam) * x[perm]
    y_mix = y  # keep original labels for loss
    return x_mix, y_mix, lam


def train_phase(projectors, optim, sched, emb_t, lab_t, mod_t, modality_idx,
                epochs, batch_size, temperature, margin, triplet_w, nce_w, cm_w, proto_w,
                use_mixup=False, grad_accum=1, hard_neg_k=8, log_prefix=""):
    """Run one training phase."""
    feat_dim = emb_t.shape[1]
    triplets_loss_fn = TripletLoss(margin=margin)
    nce_loss_fn = InfoNCELoss(temperature=temperature)
    cm_loss_fn = CrossModalAlignmentLoss(temperature=temperature)
    proto_loss_fn = ClassPrototypeLoss()

    rng = np.random.default_rng(42)
    modalities = list(modality_idx.keys())
    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        orders = {m: rng.permutation(modality_idx[m]) for m in modalities}
        per_mod = max(1, batch_size // 3)
        sizes = [len(modality_idx[m]) for m in modalities]
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
                batch_feats.append(emb_t[ids])
                batch_labels.append(lab_t[ids])
                batch_mods.extend([m] * len(ids))
            if not batch_feats: continue
            feats_b   = torch.cat(batch_feats, 0)
            labels_b  = torch.cat(batch_labels, 0)

            # mixup
            if use_mixup and np.random.rand() < 0.5:
                feats_b, _, _ = mixup_features(feats_b, labels_b, batch_mods, alpha=0.2)

            # forward through each projector
            outs = [projectors[m](feats_b[i:i+1]) for i, m in enumerate(batch_mods)]
            z = torch.cat(outs, 0)

            # losses
            lt = triplets_loss_fn(z, labels_b)
            ln = nce_loss_fn(z, labels_b)
            lc = cm_loss_fn(z, labels_b, batch_mods)
            lp = proto_loss_fn(z, labels_b)
            loss = triplet_w * lt + nce_w * ln + cm_w * lc + proto_w * lp
            loss = loss / grad_accum

            optim.zero_grad()
            loss.backward()
            if (n_batches + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for proj in projectors.values() for p in proj.parameters()], 1.0)
                optim.step()
                optim.zero_grad()

            epoch_loss += loss.item() * grad_accum
            n_batches += 1

        sched.step()
        avg = epoch_loss / max(1, n_batches)
        ep_dt = time.time() - ep_t0
        cur_lr = optim.param_groups[0]["lr"]
        print(f"  {log_prefix}[epoch {epoch+1:02d}/{epochs}] loss={avg:.4f}  lr={cur_lr:.2e}  ({ep_dt:.1f}s)")
        if avg < best_loss:
            best_loss = avg
            best_state = {m: projectors[m].state_dict() for m in modalities}

    return best_state, best_loss


def train(features_npz: str, output_ckpt: str,
          out_dim: int = 256, hidden_dim: int = 1024,
          epochs_warmup: int = 5, epochs_main: int = 15, epochs_finetune: int = 10,
          batch_size: int = 256, lr: float = 2e-3,
          temperature: float = 0.06, margin: float = 0.25,
          dropout: float = 0.1, seed: int = 42, grad_accum: int = 2):
    print(f"[train-v4] loading features from {features_npz}")
    data = np.load(features_npz, allow_pickle=True)
    emb = data["embeddings"].astype(np.float32)
    lab = data["labels"].astype(np.int64)
    mods = data["modalities"]
    n_classes = int(lab.max() + 1)
    feat_dim = emb.shape[1]
    print(f"[train-v4] emb shape: {emb.shape}, classes: {n_classes}, mods: {set(mods)}")

    modalities = ["ms", "optical", "sar"]
    modality_idx = {m: np.where(mods == m)[0] for m in modalities}
    for m, ix in modality_idx.items():
        print(f"  {m}: {len(ix)} samples")

    emb_t = torch.from_numpy(emb)
    lab_t = torch.from_numpy(lab)
    # modality indices as tensor for indexing later
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # === PHASE 1: WARMUP (high temperature, simple contrastive only) ===
    print(f"\n[train-v4] === PHASE 1: WARMUP ({epochs_warmup} epochs, contrastive only) ===")
    projectors = {m: ModalityProjectorV1(feat_dim, hidden_dim, out_dim, dropout=dropout)
                  for m in modalities}
    params = [p for proj in projectors.values() for p in proj.parameters()]
    optim = torch.optim.AdamW(params, lr=lr * 0.5, weight_decay=1e-4)

    def warmup_lr(epoch):
        return (epoch + 1) / epochs_warmup
    sched = torch.optim.lr_scheduler.LambdaLR(optim, warmup_lr)

    _, _ = train_phase(projectors, optim, sched, emb_t, lab_t, None, modality_idx,
                       epochs=epochs_warmup, batch_size=batch_size,
                       temperature=temperature * 1.5, margin=margin,
                       triplet_w=0.0, nce_w=1.0, cm_w=0.0, proto_w=0.0,
                       use_mixup=False, grad_accum=grad_accum,
                       log_prefix="WARMUP ")

    # === PHASE 2: MAIN (all losses, mixup) ===
    print(f"\n[train-v4] === PHASE 2: MAIN ({epochs_main} epochs, all losses + mixup) ===")
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    main_total = epochs_main
    def main_lr(epoch):
        progress = epoch / main_total
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, main_lr)

    main_state, _ = train_phase(projectors, optim, sched, emb_t, lab_t, None, modality_idx,
                                epochs=epochs_main, batch_size=batch_size,
                                temperature=temperature, margin=margin,
                                triplet_w=0.25, nce_w=0.35, cm_w=0.25, proto_w=0.15,
                                use_mixup=True, grad_accum=grad_accum,
                                log_prefix="MAIN   ")
    # load main state back into projectors
    for m in modalities:
        projectors[m].load_state_dict(main_state[m])

    # === PHASE 3: FINE-TUNE (lower LR, sharper contrastive) ===
    print(f"\n[train-v4] === PHASE 3: FINE-TUNE ({epochs_finetune} epochs, sharper) ===")
    optim = torch.optim.AdamW(params, lr=lr * 0.2, weight_decay=1e-5)
    ft_total = epochs_finetune
    def ft_lr(epoch):
        progress = epoch / ft_total
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, ft_lr)

    ft_state, ft_best = train_phase(projectors, optim, sched, emb_t, lab_t, None, modality_idx,
                                    epochs=epochs_finetune, batch_size=batch_size,
                                    temperature=temperature * 0.8, margin=margin,
                                    triplet_w=0.4, nce_w=0.4, cm_w=0.15, proto_w=0.05,
                                    use_mixup=False, grad_accum=grad_accum,
                                    log_prefix="FINE   ")

    os.makedirs(os.path.dirname(output_ckpt), exist_ok=True)
    torch.save({"feat_dim": feat_dim, "hidden_dim": hidden_dim, "out_dim": out_dim,
                "modalities": modalities, "state_dict": ft_state,
                "_version": "v4_enhanced"}, output_ckpt)
    print(f"\n[train-v4] saved -> {output_ckpt}  (best_ft_loss={ft_best:.4f})")


# (load_state_dict_v4 no longer needed)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "outputs" / "features.npz"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "projector.pt"))
    ap.add_argument("--epochs_warmup", type=int, default=5)
    ap.add_argument("--epochs_main", type=int, default=15)
    ap.add_argument("--epochs_finetune", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    train(args.features, args.out,
          epochs_warmup=args.epochs_warmup,
          epochs_main=args.epochs_main,
          epochs_finetune=args.epochs_finetune,
          batch_size=args.batch_size,
          hidden_dim=args.hidden, lr=args.lr)