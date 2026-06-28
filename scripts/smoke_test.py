"""Quick smoke test: load backbone, run a few images, time it."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from src.dataset import discover_datasets, MultiModalDataset
from src.backbones import build_backbone

info = discover_datasets(r"D:\BAH2026")
samples = info["eurosat"][:30] + info["sentinel_paired"][:30]
ds = MultiModalDataset(samples)
print(f"items: {len(ds)}")

m = build_backbone("resnet50")
m.eval()
print(f"feat_dim: {m.feat_dim}")

# Run a small batch
xs, ys, ms, paths, ids = [], [], [], [], []
for i in range(min(8, len(ds))):
    x, y, mo, p, sid = ds[i]
    xs.append(x); ys.append(y); ms.append(mo); paths.append(p); ids.append(sid)
xb = torch.stack(xs, 0)

t0 = time.time()
with torch.no_grad():
    feats = m(xb)
print(f"8 images in {time.time()-t0:.2f}s, feat shape: {feats.shape}")
print("First feat norm:", feats[0].norm().item())
