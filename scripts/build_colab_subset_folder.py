"""
build_colab_subset_folder.py
=============================

Build a SUBSET folder (no zip) for fast & reliable Drive upload.

Creates  D:\\BAH2026_colab_subset\\  with this structure:
    BAH2026_colab_subset/
    +-- EuroSAT/  (max N files/class/modality)
    +-- Sentinel/ (max N files/class/modality)

Folder upload to Drive is more reliable than zip upload because:
  - No central-directory corruption possible
  - Drive resumes partial uploads automatically
  - Failed files can be re-uploaded individually

Usage:
    D:\\BAH2026\\.venv\\Scripts\\python.exe D:\\BAH2026\\cross_modal_retrieval\\scripts\\build_colab_subset_folder.py
    D:\\BAH2026\\.venv\\Scripts\\python.exe D:\\BAH2026\\cross_modal_retrieval\\scripts\\build_colab_subset_folder.py --max-per-class 300
"""

import argparse
import shutil
import sys
from pathlib import Path


def build(root: Path, out_root: Path, max_per_class: int):
    out_root.mkdir(parents=True, exist_ok=True)

    total = 0
    for base_name in ("EuroSAT", "Sentinel"):
        base = root / base_name
        if not base.exists():
            print(f"[warn] missing source: {base}")
            continue
        out_base = out_root / base_name
        out_base.mkdir(parents=True, exist_ok=True)

        for cls_dir in sorted([d for d in base.iterdir() if d.is_dir()]):
            cls_name = cls_dir.name
            out_cls = out_base / cls_name
            out_cls.mkdir(parents=True, exist_ok=True)

            for mod in ("s1", "s2"):
                src_mod = cls_dir / mod
                if not src_mod.exists():
                    continue
                dst_mod = out_cls / mod
                dst_mod.mkdir(parents=True, exist_ok=True)

                files = sorted([p for p in src_mod.iterdir() if p.is_file()])
                if max_per_class is not None:
                    files = files[:max_per_class]

                for src in files:
                    dst = dst_mod / src.name
                    if not dst.exists():
                        shutil.copy2(src, dst)
                        total += 1

    n_files = sum(1 for _ in out_root.rglob("*") if _.is_file())
    print(f"[done] copied {total} files -> {out_root}")
    print(f"[done] total {n_files} files in subset folder")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(r"D:\BAH2026"))
    ap.add_argument("--output", type=Path, default=Path(r"D:\BAH2026_colab_subset"))
    ap.add_argument("--max-per-class", type=int, default=300,
                    help="cap each (class, modality) to N files (default 300)")
    args = ap.parse_args()

    if args.output.exists():
        print(f"[info] removing existing {args.output}")
        shutil.rmtree(args.output)

    build(args.root, args.output, args.max_per_class)


if __name__ == "__main__":
    main()