"""
Extract embeddings for all multi-modal samples using a pretrained backbone.

Run:  python -m scripts.extract_features --backbone dinov2_small

Saves:
  outputs/features.npz         # embeddings, labels, modalities, paths, ids
  outputs/backbone_<name>.pt   # (optional) cache of pretrained trunk
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# allow imports from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import discover_datasets, MultiModalDataset
from src.backbones import build_backbone


def collate(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    ms = [b[2] for b in batch]
    paths = [b[3] for b in batch]
    ids = [b[4] for b in batch]
    return xs, ys, ms, paths, ids


@torch.no_grad()
def extract(backbone_name: str, output_path: str, batch_size: int = 32, num_workers: int = 0,
            eurosat_per_class: int = 300, sentinel_per_class: int = 400):
    print(f"[extract] backbone={backbone_name}, batch_size={batch_size}, workers={num_workers}")
    device = "cpu"
    backbone = build_backbone(backbone_name).to(device).eval()

    info = discover_datasets(str(ROOT.parent),
                              eurosat_per_class=eurosat_per_class,
                              sentinel_per_class=sentinel_per_class)
    samples = info["eurosat"] + info["sentinel_paired"] + info["sentinel_unpaired"]
    print(f"[extract] Total samples (paired): {len(info['eurosat']) + len(info['sentinel_paired'])}, unpaired: {len(info['sentinel_unpaired'])}")

    ds = MultiModalDataset(samples)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)

    all_emb = []
    all_lab = []
    all_mod = []
    all_path = []
    all_id = []
    t0 = time.time()
    for i, (xs, ys, ms, paths, ids) in enumerate(tqdm(loader, desc="extract")):
        xs = xs.to(device)
        feats = backbone(xs).cpu().numpy()
        all_emb.append(feats)
        all_lab.append(ys.numpy())
        all_mod.extend(ms)
        all_path.extend(paths)
        all_id.extend(ids)
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  batch {i+1}/{len(loader)} | elapsed {elapsed:.1f}s")

    emb = np.concatenate(all_emb, axis=0).astype(np.float32)
    lab = np.concatenate(all_lab, axis=0)
    print(f"[extract] Done. embeddings: {emb.shape}, elapsed {time.time()-t0:.1f}s")

    np.savez_compressed(
        output_path,
        embeddings=emb,
        labels=lab,
        modalities=np.array(all_mod),
        paths=np.array(all_path),
        ids=np.array(all_id),
    )
    print(f"[extract] saved -> {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "features.npz"))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--eurosat_per_class", type=int, default=300)
    ap.add_argument("--sentinel_per_class", type=int, default=400)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    extract(args.backbone, args.out, args.batch_size, args.num_workers,
            eurosat_per_class=args.eurosat_per_class,
            sentinel_per_class=args.sentinel_per_class)