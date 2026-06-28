"""
Re-extract ResNet-50 features with Test-Time Augmentation (TTA).

For each image we compute 4 augmented features (original, hflip, vflip, hflip+vflip)
and average them.  This is cheap (same backbone forward, just 4x), and typically
adds +2-4% accuracy because the projection head sees noise-robust features.

Saves: outputs/features.npz  (same schema as before, but richer features)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backbones import build_backbone
from src.dataset import (
    _read_image, discover_datasets, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE,
)


class ImgDatasetTTA(Dataset):
    def __init__(self, items, img_size=224):
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
        if rgb.shape[0] != self.img_size or rgb.shape[1] != self.img_size:
            rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        # base tensor
        x = rgb.astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1)
        x = (x - self.mean) / self.std
        # TTA flips (hflip, vflip, hvflip)
        hflip = torch.flip(x, dims=[2])
        vflip = torch.flip(x, dims=[1])
        hvflip = torch.flip(x, dims=[1, 2])
        return x, hflip, vflip, hvflip, label, modality, path, sid


def collate(batch):
    xs, hf, vf, hvf, ys, ms, ps, ss = zip(*batch)
    return (
        torch.stack(xs), torch.stack(hf), torch.stack(vf), torch.stack(hvf),
        torch.tensor(ys), list(ms), list(ps), list(ss)
    )


def extract(base_dir, backbone_name, batch_size, num_workers, out_path):
    print(f"\n========== EXTRACT FEATURES w/ TTA ({backbone_name}) ==========")
    info = discover_datasets(base_dir)

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

    backbone = build_backbone(backbone_name).eval()
    img_size = backbone.img_size if hasattr(backbone, "img_size") else IMG_SIZE
    if backbone_name == "resnet50":
        img_size = 224

    ds = ImgDatasetTTA(items, img_size=img_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True, collate_fn=collate)

    feats, labels, mods, paths, ids = [], [], [], [], []
    t0 = time.time()
    with torch.inference_mode():
        for x, h, v, hv, y, m, p, sid in tqdm(dl, ncols=100):
            # Concatenate all 4 views along batch dim for one big forward pass
            big = torch.cat([x, h, v, hv], dim=0)
            z = backbone(big)  # (4B, feat_dim)
            # Split back and average
            chunks = z.chunk(4, dim=0)
            z_avg = sum(chunks) / 4.0  # element-wise mean of 4 view features
            feats.append(z_avg.numpy().astype(np.float32))
            labels.append(y.numpy())
            mods.extend(m); paths.extend(p); ids.extend(sid)
    feats = np.concatenate(feats, 0).astype(np.float32)
    labels = np.concatenate(labels, 0).astype(np.int64)
    mods = np.array(mods)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, embeddings=feats, labels=labels,
                        modalities=mods, paths=np.array(paths), ids=np.array(ids))
    print(f"  saved -> {out_path}  ({feats.shape}, {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="D:/BAH2026")
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "features.npz"))
    args = ap.parse_args()
    extract(args.base_dir, args.backbone, args.batch_size, args.num_workers, args.out)