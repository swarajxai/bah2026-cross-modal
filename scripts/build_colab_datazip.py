"""
build_colab_datazip.py
======================

Local helper: builds the data zip that gets uploaded to Colab.

Creates  D:\\BAH2026_data.zip  with this structure:

    BAH2026_data.zip
    +-- EuroSAT/
    |   +-- AnnualCrop/
    |   |   +-- s1/ (multispectral .tif files)
    |   |   +-- s2/ (optical .jpg files)
    |   +-- Forest/ ...
    |   +-- ... (10 classes)
    +-- Sentinel/
        +-- agri/
        |   +-- s1/ (SAR .png files)
        |   +-- s2/ (optical .png files)
        +-- barrenland/ ...

Usage (run from anywhere):
    D:\\BAH2026\\.venv\\Scripts\\python.exe D:\\BAH2026\\cross_modal_retrieval\\scripts\\build_colab_datazip.py

Options:
    --output   path of zip (default: D:\\BAH2026_data.zip)
    --max-per-class N  cap each class folder to N files per modality (useful for testing)

After zip is built, in Colab simply upload this zip when prompted.
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path
import time


def build_zip(root: Path, out_path: Path, max_per_class: int | None = None,
              exclude_globs: tuple[str, ...] = ("*.pyc", "__pycache__/*")):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # First, count what will go in
    eurosat = root / "EuroSAT"
    sentinel = root / "Sentinel"
    total_files = 0
    for base in (eurosat, sentinel):
        if not base.exists():
            print(f"[warn] missing: {base}")
            continue
        for cls_dir in sorted(base.iterdir()):
            if not cls_dir.is_dir():
                continue
            for mod_dir in ("s1", "s2"):
                mod_path = cls_dir / mod_dir
                if not mod_path.exists():
                    continue
                files = sorted(p for p in mod_path.iterdir() if p.is_file())
                if max_per_class is not None:
                    files = files[:max_per_class]
                total_files += len(files)

    print(f"[zip] will package ~{total_files} files into {out_path}")
    t0 = time.time()

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED,
                         allowZip64=True) as zf:
        written = 0
        for base in (eurosat, sentinel):
            if not base.exists():
                continue
            for cls_dir in sorted(base.iterdir()):
                if not cls_dir.is_dir():
                    continue
                for mod_dir in ("s1", "s2"):
                    mod_path = cls_dir / mod_dir
                    if not mod_path.exists():
                        continue
                    files = sorted(p for p in mod_path.iterdir() if p.is_file())
                    if max_per_class is not None:
                        files = files[:max_per_class]
                    for p in files:
                        # arcname should preserve EuroSAT/AnnualCrop/s1/...
                        arc = p.relative_to(root).as_posix()
                        zf.write(p, arcname=arc)
                        written += 1
                        if written % 1000 == 0:
                            print(f"  ... {written}/{total_files}  ({time.time()-t0:.0f}s)")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[zip] done: {written} files, {size_mb:.0f} MB, {time.time()-t0:.0f}s")
    print(f"[zip] -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(r"D:\BAH2026"),
                    help="Project root (default D:\\BAH2026)")
    ap.add_argument("--output", type=Path, default=Path(r"D:\BAH2026_data.zip"),
                    help="Output zip path (default D:\\BAH2026_data.zip)")
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="Cap each class folder to N files per modality (testing only)")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"[err] root not found: {args.root}")
        sys.exit(1)
    if (args.output).exists():
        resp = input(f"[warn] {args.output} already exists. Overwrite? [y/N] ").strip().lower()
        if resp != "y":
            print("aborted.")
            sys.exit(0)

    build_zip(args.root, args.output, args.max_per_class)


if __name__ == "__main__":
    main()