"""
Multi-modal Satellite Image Dataset for Cross-Modal Retrieval.

Handles 4 modalities:
  - Multispectral (.tif)  -> EuroSAT <class>/s1/
  - Optical     (.jpg)    -> EuroSAT <class>/s2/  AND Sentinel <class>/s2/
  - SAR                    -> Sentinel <class>/s1/

Pairing strategy:
  - EuroSAT: s1_<id>.tif  <-->  s2_<id>.jpg  (same numeric ID within a class)
  - Sentinel: s1/<files>  <-->  s2/<files>   (sorted list, 1-to-1 by index)

Each sample yields:
  image  : Tensor[3, 224, 224]   (already modality-normalized)
  label  : int                   (class id, used as ground-truth relevance)
  modality : str                 ('ms' | 'optical' | 'sar')
  path   : str                   (for debugging)
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import tifffile
import cv2


# ImageNet normalization (works well with timm/DINOv2 pretrained backbones)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_SIZE = 224

# Class label mapping (EuroSAT 10 classes, Sentinel 4 classes)
# We map by name where possible to keep a unified label space.
ALL_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
    # Sentinel extras (mapped to nearest EuroSAT class)
    "agri", "barrenland", "grassland", "urban",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(ALL_CLASSES)}


def _read_image(path: str, modality: str) -> np.ndarray:
    """Read image file and return uint8 HxWx3 RGB numpy array."""
    if modality == "ms":
        # Multispectral .tif: may have multiple bands; we average & replicate to 3-ch.
        arr = tifffile.imread(path)
        if arr.ndim == 2:
            arr = arr[..., None]
        # If many bands, take a representative subset (RGB-like) or average.
        if arr.shape[-1] >= 3:
            # pick 3 spread-out bands, or just take first 3 / mean
            if arr.shape[-1] >= 13:  # Sentinel-2 has 13 bands; pick R,G,B
                r, g, b = arr[..., 4], arr[..., 3], arr[..., 2]
                arr = np.stack([r, g, b], axis=-1)
            else:
                arr = arr[..., :3]
        else:
            arr = np.repeat(arr, 3, axis=-1)
        # normalize to 0..255 if needed
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            mn, mx = arr.min(), arr.max()
            if mx - mn > 0:
                arr = (arr - mn) / (mx - mn) * 255.0
            arr = arr.astype(np.uint8)
        return arr
    elif modality == "sar":
        # Sentinel SAR: typically grayscale single channel image
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            img = np.array(Image.open(path))
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.stack([img, img, img], axis=-1)
        if img.dtype != np.uint8:
            mn, mx = float(img.min()), float(img.max())
            if mx - mn > 0:
                img = (img - mn) / (mx - mn) * 255.0
            img = img.astype(np.uint8)
        return img
    else:  # optical / RGB
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            pil = Image.open(path).convert("RGB")
            img = np.array(pil)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


def _preprocess(rgb: np.ndarray) -> torch.Tensor:
    """Resize to 224x224 and normalize with ImageNet stats."""
    if rgb.shape[0] != IMG_SIZE or rgb.shape[1] != IMG_SIZE:
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    return torch.from_numpy(x).permute(2, 0, 1).float()  # CHW


def _extract_id(filename: str) -> str:
    """Pull the numeric/alphanumeric ID from filename e.g. PermanentCrop_1619.tif -> 1619."""
    m = re.search(r"_(\d+)", filename)
    return m.group(1) if m else Path(filename).stem


def scan_eurosat(root: str) -> List[Dict]:
    """Scan EuroSAT/{class}/{s1|s2}/ and produce paired samples.

    Returns a list of dicts with keys:
        ms_path, optical_path, class_name, label, pair_id
    For a 'pair', we use the intersection of IDs in s1 and s2 of a class.
    """
    root = Path(root)
    pairs = []
    for class_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        s1 = class_dir / "s1"
        s2 = class_dir / "s2"
        if not (s1.exists() and s2.exists()):
            continue
        s1_files = {f.name.split(".")[0].split("_")[-1]: f for f in s1.iterdir() if f.is_file()}
        s2_files = {f.name.split(".")[0].split("_")[-1]: f for f in s2.iterdir() if f.is_file()}
        common = sorted(set(s1_files) & set(s2_files))
        for fid in common:
            cls_name = class_dir.name
            pairs.append({
                "id": f"{cls_name}_{fid}",
                "class_name": cls_name,
                "label": CLASS_TO_IDX[cls_name],
                "ms_path": str(s1_files[fid]),
                "optical_path": str(s2_files[fid]),
                "dataset": "eurosat",
            })
    return pairs


def scan_sentinel(root: str) -> Tuple[List[Dict], List[Dict]]:
    """Scan Sentinel/{class}/{s1|s2}/ and produce paired samples by numeric patch id.

    s1 = SAR, s2 = Optical.
    Files are named like 'ROIs1868_summer_s1_59_p10.png' / '..._s2_..._p10.png'.
    We pair by the trailing '_p<digits>' suffix.
    Returns (paired_list, unpaired_list)
    """
    import re as _re
    def _patch_id(name: str) -> str:
        m = _re.search(r"_(p\d+)\.", name)
        return m.group(1) if m else name

    root = Path(root)
    paired, unpaired = [], []
    for class_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        s1 = class_dir / "s1"
        s2 = class_dir / "s2"
        if not (s1.exists() and s2.exists()):
            continue
        sar_map = {_patch_id(f.name): f for f in s1.iterdir() if f.is_file()}
        opt_map = {_patch_id(f.name): f for f in s2.iterdir() if f.is_file()}
        common = sorted(set(sar_map) & set(opt_map))
        sar_files = [sar_map[n_] for n_ in common]
        opt_files = [opt_map[n_] for n_ in common]
        cls_name = class_dir.name
        n = len(common)
        for i in range(n):
            paired.append({
                "id": f"{cls_name}_{i}",
                "class_name": cls_name,
                "label": CLASS_TO_IDX.get(cls_name, -1),
                "sar_path": str(sar_files[i]),
                "optical_path": str(opt_files[i]),
                "dataset": "sentinel",
            })
        # if unbalanced, keep extras as unpaired for gallery
        for j in range(n, len(sar_files)):
            unpaired.append({
                "id": f"{cls_name}_sar{j}",
                "class_name": cls_name,
                "label": CLASS_TO_IDX.get(cls_name, -1),
                "sar_path": str(sar_files[j]),
                "dataset": "sentinel",
            })
        for j in range(n, len(opt_files)):
            unpaired.append({
                "id": f"{cls_name}_opt{j}",
                "class_name": cls_name,
                "label": CLASS_TO_IDX.get(cls_name, -1),
                "optical_path": str(opt_files[j]),
                "dataset": "sentinel",
            })
    return paired, unpaired


class MultiModalDataset(Dataset):
    """Dataset that yields (image, label, modality, path) tuples.

    modality in {'ms','optical','sar'}.
    """

    def __init__(self, samples: List[Dict]):
        # Flatten: each pair contributes up to 3 samples (ms, optical, sar)
        self.items: List[Tuple[str, str, int, str]] = []  # (path, modality, label, id)
        for s in samples:
            label = s["label"]
            if s.get("ms_path"):
                self.items.append((s["ms_path"], "ms", label, s["id"]))
            if s.get("optical_path"):
                self.items.append((s["optical_path"], "optical", label, s["id"]))
            if s.get("sar_path"):
                self.items.append((s["sar_path"], "sar", label, s["id"]))
        # Filter unknown labels
        self.items = [x for x in self.items if x[2] >= 0]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, modality, label, sid = self.items[idx]
        try:
            rgb = _read_image(path, modality)
            x = _preprocess(rgb)
        except Exception as e:
            # corrupt file: return a zero tensor so collate doesn't fail
            x = torch.zeros(3, IMG_SIZE, IMG_SIZE)
        return x, label, modality, path, sid


def discover_datasets(base_dir: str, eurosat_per_class: int = 300, sentinel_per_class: int = 400):
    """Top-level helper used everywhere else.

    To stay within CPU/time budget, we sub-sample both datasets to a
    balanced per-class cap (default 300 EuroSAT pairs + 400 Sentinel
    pairs per class).  Set `eurosat_per_class=None` / `sentinel_per_class=None`
    to disable sub-sampling and use everything.
    """
    eurosat = scan_eurosat(os.path.join(base_dir, "EuroSAT"))
    sent_paired, sent_unpaired = scan_sentinel(os.path.join(base_dir, "Sentinel"))

    # balanced sub-sample
    if eurosat_per_class is not None and eurosat_per_class > 0:
        by_cls = {}
        for s in eurosat:
            by_cls.setdefault(s["class_name"], []).append(s)
        sub = []
        for c, items in by_cls.items():
            sub.extend(items[:eurosat_per_class])
        eurosat = sub
    if sentinel_per_class is not None and sentinel_per_class > 0:
        by_cls = {}
        for s in sent_paired:
            by_cls.setdefault(s["class_name"], []).append(s)
        sub = []
        for c, items in by_cls.items():
            sub.extend(items[:sentinel_per_class])
        sent_paired = sub
    return {
        "eurosat": eurosat,
        "sentinel_paired": sent_paired,
        "sentinel_unpaired": sent_unpaired,
    }


if __name__ == "__main__":
    import sys
    base = r"D:\BAH2026"
    info = discover_datasets(base)
    print(f"EuroSAT pairs: {len(info['eurosat'])}")
    print(f"Sentinel paired: {len(info['sentinel_paired'])}")
    print(f"Sentinel unpaired: {len(info['sentinel_unpaired'])}")
    ds = MultiModalDataset(info["eurosat"][:5] + info["sentinel_paired"][:2])
    print(f"Flattened items: {len(ds)}")
    x, y, m, p, sid = ds[0]
    print("sample:", x.shape, y, m, sid)
