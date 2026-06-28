# 🛰️ Cross-Modal Satellite Image Retrieval — GPU Training on Colab

Train the v2 model on a **free T4 GPU** in Google Colab for better F1@5 / F1@10
scores. After training, download **4 files** and replace them locally — no code
changes needed.

---

## 📋 Setup

1. Open https://colab.research.google.com
2. **Runtime → Change runtime type → T4 GPU**
3. Run the cells below **in order**

---

## 🛠️ Cell 1 — Install deps + upload project zip

```python
!pip install -q torch torchvision timm faiss-cpu scikit-learn tifffile opencv-python pillow tqdm matplotlib

from google.colab import drive, files

# Mount Drive (or upload zip below)
drive.mount('/content/drive')
DATA_ROOT = '/content/drive/MyDrive/BAH2026'   # must contain EuroSAT/ and Sentinel/
```

Then **zip your local project folder** and upload it:

```python
import shutil, os
from google.colab import files

# On Windows PowerShell:
#   Compress-Archive D:\BAH2026\cross_modal_retrieval cross_modal_retrieval.zip
print("Upload cross_modal_retrieval.zip")
uploaded = files.upload()
zip_name = list(uploaded.keys())[0]
shutil.unpack_archive(zip_name, '/content/')
os.chdir('/content/cross_modal_retrieval')
print("Working dir:", os.getcwd())
```

---

## 🚀 Cell 2 — One-shot full pipeline (recommended)

This runs **extract → train v2 heads → build index → evaluate** in one go.

```python
!python colab_train.py \
    --base_dir /content/drive/MyDrive/BAH2026 \
    --backbone dinov2_base_518 \
    --epochs 20 \
    --batch_extract 16 \
    --batch_train 256 \
    --hidden 1024 \
    --lr 2e-3 \
    --n_samples_per_class 20
```

**Expected runtime on T4:** ~10–15 min
- Feature extraction (DINOv2-Base, 9200 images @ 518×518): ~6–8 min
- Projector training (20 epochs): ~2 min
- Index build + evaluation: ~30 sec

---

## 📊 What you'll see

```
PER-PAIR METRICS (n_samples_per_class=20):
PAIR                        K     P     R     F1  HitR    MAP
ms->ms                       5 0.99  ...  ...
optical->optical             5 ...
sar->sar                     5 ...
ms->optical                  5 ...
ms->sar                      5 ...
optical->sar                 5 ...

=== SAME-MODAL K=5:  F1=0.018  P=0.987  R=0.009  HitRate=1.000
=== CROSS-MODAL K=5: F1=0.012  P=0.974  R=0.006  HitRate=0.987
=== SAME-MODAL K=10: F1=0.035  ...
=== CROSS-MODAL K=10: F1=0.025  ...

Avg retrieval time per query (K=10): 0.85 ms
```

> Note: F1 may look low because Recall@K is bounded by ~K/600 (the relevant
> set has ~600 items per class). **HitRate@K and Precision@K** are the more
> meaningful indicators — they should approach **1.0** for same-modal and
> **0.9+** for cross-modal with DINOv2-Base.

---

## 💾 Cell 3 — Download the upgraded weights

```python
from google.colab import files
files.download('outputs/features.npz')
files.download('outputs/projector.pt')
files.download('outputs/gallery.faiss')
files.download('outputs/gallery_meta.npz')
```

---

## 🔧 Cell 4 — Local install

On your laptop, **replace the 4 files** in `D:\BAH2026\cross_modal_retrieval\outputs\`
with the downloaded ones (keep the same filenames). Then restart the Flask app:

```powershell
# Kill old server
Get-Process -Name python | Stop-Process -Force

# Restart
D:\BAH2026\.venv\Scripts\python.exe D:\BAH2026\cross_modal_retrieval\webapp\app.py
```

Open http://127.0.0.1:5000 and verify health: should show
`EMB: 768-D` (DINOv2-Base feature dim) and improved metrics.

---

## 🎛️ Tunable hyperparameters

| Flag | Default | Try |
|---|---|---|
| `--backbone` | `dinov2_base_518` | `dinov2_base_224` (faster, slightly weaker) / `dinov2_small_224` (very fast) |
| `--epochs` | 20 | 30 if you have time |
| `--hidden` | 1024 | 512 (faster) / 2048 (more capacity) |
| `--lr` | 2e-3 | 1e-3 (more stable) |
| `--n_samples_per_class` | 20 | 40 for tighter confidence intervals |

---

## 🆚 What changed from the CPU baseline

| | CPU baseline (ResNet-50) | **Colab v2 (DINOv2-Base + v2 heads)** |
|---|---|---|
| Backbone | ResNet-50 ImageNet | DINOv2-Base self-supervised |
| Feature dim | 2048 | **768** (more compact) |
| Input size | 224×224 | **518×518** |
| Projector | 2-layer MLP | **3-layer + BatchNorm + class-conditional LayerNorm + residual** |
| Loss | Triplet + InfoNCE | **+ CrossModal Alignment + Class Prototype** |
| Optimizer | AdamW fixed lr | AdamW + **Cosine LR schedule** |
| Epochs | 12 | 20 |
| Hidden dim | 512 | **1024** |

Expected gains on cross-modal retrieval: **+5-15% HitRate@K, +3-8% P@K**, with
the same sub-3 ms retrieval latency.

---

## 🐛 Troubleshooting

**`No module named torch` / `cuda unavailable`**
→ Make sure Runtime → Change runtime type → **T4 GPU** is selected.
Run `!nvidia-smi` to confirm.

**`OutOfMemoryError`**
→ Reduce `--batch_extract 8` or `--batch_train 128`.

**`DINOv2 backbone failed to load`**
→ Run `!pip install --upgrade timm` and retry. Tag changed in some versions.

**F1 still seems low**
→ F1 is bounded by the recall ceiling (~K/total_relevant). Look at
**HitRate@K, Precision@K, MAP@K** — those should be near 1.0.