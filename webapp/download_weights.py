#!/usr/bin/env python3
"""
Download model artifacts from HuggingFace Hub at runtime.

This script runs inside the Docker container at startup.  It pulls the
heavy .pt / .faiss / .npz files from a separate HF Dataset repo into
/app/outputs/ BEFORE gunicorn starts.

Why:  HF Spaces git push rejects files > 10 MB, so we keep the repo
light and pull weights from a sibling HF Dataset repo at startup.
"""
import os
import sys
from pathlib import Path

OUT = Path("/app/outputs")
OUT.mkdir(parents=True, exist_ok=True)

# === CONFIGURE THESE ===
HF_DATASET_REPO = "swarajkamila/bah2026-weights"   # sibling dataset repo
FILES = [
    "projector.pt",
    "gallery.faiss",
    "gallery_meta.npz",
    "features.npz",
]
# ========================

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("[hf-download] installing huggingface_hub...")
    os.system(f"{sys.executable} -m pip install --quiet huggingface_hub")
    from huggingface_hub import hf_hub_download

# HF_TOKEN is auto-injected by HF Spaces for private datasets.
TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

missing = []
for fname in FILES:
    dst = OUT / fname
    if dst.exists() and dst.stat().st_size > 1024:
        print(f"[hf-download] {fname}: present ({dst.stat().st_size/1e6:.1f} MB)")
        continue
    try:
        print(f"[hf-download] {fname}: downloading from {HF_DATASET_REPO}...")
        p = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=fname,
            repo_type="dataset",
            token=TOKEN,
            cache_dir="/tmp/hf_cache",
        )
        # Move into /app/outputs/
        import shutil
        shutil.move(p, dst)
        print(f"[hf-download] {fname}: OK ({dst.stat().st_size/1e6:.1f} MB)")
    except Exception as e:
        print(f"[hf-download] {fname}: FAILED — {e}")
        missing.append(fname)

if missing:
    print(f"[hf-download] WARNING: missing {missing}")
    sys.exit(1)

print("[hf-download] all weights ready")