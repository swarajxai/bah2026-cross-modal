#!/usr/bin/env python3
"""
Download model artifacts from HuggingFace Hub at runtime.

This script runs inside the Docker container at startup.  It pulls the
heavy .pt / .faiss / .npz files from a sibling HF Dataset repo into
/app/outputs/ BEFORE gunicorn starts.

Why:  HF Spaces git push rejects files > 10 MB, so we keep the repo
light and pull weights from a sibling HF Dataset repo at startup.
"""
import os
import sys
import shutil
from pathlib import Path

# Force HF_HOME so huggingface_hub uses /app (writable) instead of /tmp (may be read-only)
HF_HOME = Path("/app/hf_cache")
HF_HOME.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_HOME)

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

# HF_TOKEN is auto-injected by HF Spaces
TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
print(f"[hf-download] HF_HOME={HF_HOME}", flush=True)
print(f"[hf-download] OUT={OUT}", flush=True)
print(f"[hf-download] repo={HF_DATASET_REPO}", flush=True)
if TOKEN:
    print(f"[hf-download] using HF_TOKEN (len={len(TOKEN)})", flush=True)
else:
    print("[hf-download] WARNING: no HF_TOKEN set — using unauthenticated", flush=True)

missing = []
for fname in FILES:
    dst = OUT / fname
    if dst.exists() and dst.stat().st_size > 1024:
        print(f"[hf-download] {fname}: present ({dst.stat().st_size/1e6:.1f} MB)", flush=True)
        continue
    try:
        print(f"[hf-download] {fname}: downloading from {HF_DATASET_REPO}...", flush=True)
        # Download directly to /app/outputs/<fname>
        # hf_hub_download returns the path of the cached file
        cached_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=fname,
            repo_type="dataset",
            token=TOKEN,
            cache_dir=str(HF_HOME),
            local_dir=None,                # use default cache structure
        )
        print(f"[hf-download] {fname}: cached at {cached_path}", flush=True)
        # Copy (not move — cache may be needed for subsequent runs / restarts)
        shutil.copyfile(cached_path, dst)
        # Verify destination
        if not dst.exists() or dst.stat().st_size < 1024:
            raise RuntimeError(f"destination missing after copy: {dst}")
        print(f"[hf-download] {fname}: OK ({dst.stat().st_size/1e6:.1f} MB) -> {dst}", flush=True)
    except Exception as e:
        print(f"[hf-download] {fname}: FAILED -- {type(e).__name__}: {e}", flush=True)
        missing.append(fname)

if missing:
    print(f"[hf-download] WARNING: missing {missing}", flush=True)
    sys.exit(1)

print("[hf-download] all weights ready", flush=True)