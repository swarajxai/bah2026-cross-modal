#!/usr/bin/env python3
"""
Embed base64-encoded PNG thumbnails into gallery_meta.npz so the webapp
can serve them without needing access to the original dataset files
(which only exist on the training machine).

Each thumbnail is 96x96 px PNG — ~3-6 KB compressed.  For ~9200 items
this adds ~30-50 MB to gallery_meta.npz.

Run after training (after build_index.py has produced the meta file).
"""
import base64
import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image


def read_rgb(path: str, modality: str, target_size: int = 96):
    """Read image as uint8 HxWx3 RGB numpy array, resized to target_size."""
    try:
        if modality == "ms":
            arr = tifffile.imread(path)
            if arr.ndim == 2:
                arr = arr[..., None]
            if arr.shape[-1] >= 13:
                arr = np.stack([arr[..., 4], arr[..., 3], arr[..., 2]], -1)
            elif arr.shape[-1] >= 3:
                arr = arr[..., :3]
            else:
                arr = np.repeat(arr, 3, -1)
            if arr.dtype != np.uint8:
                arr = arr.astype(np.float32)
                mn, mx = float(arr.min()), float(arr.max())
                if mx > mn:
                    arr = (arr - mn) / (mx - mn) * 255.0
                arr = arr.astype(np.uint8)
        elif modality == "sar":
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                img = np.array(Image.open(path))
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            arr = np.stack([img, img, img], -1)
            if arr.dtype != np.uint8:
                mn, mx = float(arr.min()), float(arr.max())
                if mx > mn:
                    arr = (arr - mn) / (mx - mn) * 255.0
                arr = arr.astype(np.uint8)
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                img = np.array(Image.open(path).convert("RGB"))
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            arr = img
        if arr.shape[0] != target_size or arr.shape[1] != target_size:
            arr = cv2.resize(arr, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return arr
    except Exception as e:
        # Return a placeholder black image on error
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)


def make_thumbnail_b64(rgb: np.ndarray, size: int = 96) -> str:
    """Encode RGB array as base64 PNG string."""
    pil = Image.fromarray(rgb).resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    meta_path = Path("outputs/gallery_meta.npz")
    if not meta_path.exists():
        print(f"[ERR] {meta_path} not found")
        sys.exit(1)

    print(f"[load] {meta_path}")
    data = np.load(meta_path, allow_pickle=True)
    paths = data["paths"]
    mods = data["modalities"]
    print(f"[load] {len(paths)} gallery items")

    out_path = meta_path  # overwrite same file
    tmp_path = meta_path.with_suffix(".npz.tmp")

    # Encode thumbnails one by one with progress
    thumbs = np.empty(len(paths), dtype=object)
    t0 = time.time()
    n_err = 0
    for i in range(len(paths)):
        p = str(paths[i])
        m = str(mods[i])
        try:
            rgb = read_rgb(p, m, target_size=96)
            thumbs[i] = make_thumbnail_b64(rgb, size=96)
        except Exception as e:
            thumbs[i] = ""  # blank if error
            n_err += 1
        if (i + 1) % 500 == 0:
            dt = time.time() - t0
            rate = (i + 1) / dt
            eta = (len(paths) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(paths)}] {rate:.0f}/s, ETA {eta:.0f}s, errors={n_err}")

    print(f"[done] {len(paths)} thumbnails encoded in {time.time()-t0:.0f}s, errors={n_err}")

    # Save with thumbnails field added (preserving all other fields)
    print(f"[save] writing {tmp_path}")
    np.savez_compressed(
        tmp_path,
        paths=data["paths"],
        modalities=data["modalities"],
        labels=data["labels"],
        ids=data["ids"],
        embeddings=data["embeddings"],
        thumbs=thumbs,
    )

    # Replace original
    tmp_path.replace(out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[save] {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()