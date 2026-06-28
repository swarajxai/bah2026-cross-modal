"""
activate_v6.py
==============

Local activation: takes a v6 artifact zip (downloaded from Colab) and replaces
the current projector.pt / gallery.faiss / gallery_meta.npz with the v6
versions, after taking a timestamped backup of v5.

Usage:
    D:\\BAH2026\\.venv\\Scripts\\python.exe D:\\BAH2026\\cross_modal_retrieval\\scripts\\activate_v6.py
    D:\\BAH2026\\.venv\\Scripts\\python.exe D:\\BAH2026\\cross_modal_retrieval\\scripts\\activate_v6.py --zip <path-to-cross_modal_v6.zip>
    D:\\BAH2026\\.venv\\Scripts\\python.exe D:\\BAH2026\\cross_modal_retrieval\\scripts\\activate_v6.py --rollback
"""

import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

# files that the zip may carry (under their v6_* names)
V6_FILES = {
    "v6_projector.pt":     "projector.pt",
    "v6_gallery.faiss":    "gallery.faiss",
    "v6_gallery_meta.npz": "gallery_meta.npz",
    "v6_eval.json":        "v6_eval.json",          # kept as sidecar, not overwritten
    "features_dinov2.npz": "features_dinov2.npz",   # cached DINOv2 features
}

# files in outputs/ that we always back up before swapping
BACKUP_TARGETS = ["projector.pt", "gallery.faiss", "gallery_meta.npz"]


def log(msg):
    print(f"[activate-v6] {msg}", flush=True)


def list_backups():
    return sorted(p for p in OUTPUTS.iterdir()
                  if p.is_dir() and p.name.startswith("backup_"))


def find_zip(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"zip not found: {explicit}")
        return explicit
    # try common locations
    candidates = [
        OUTPUTS / "cross_modal_v6.zip",
        Path(r"D:\Downloads\cross_modal_v6.zip"),
        Path.home() / "Downloads" / "cross_modal_v6.zip",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "v6 zip not found. Pass --zip <path> or place at "
        "D:\\BAH2026\\cross_modal_retrieval\\outputs\\cross_modal_v6.zip"
    )


def backup_current():
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = OUTPUTS / f"backup_v5_pre_v6_{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for fname in BACKUP_TARGETS:
        src = OUTPUTS / fname
        if src.exists():
            shutil.copy2(src, backup_dir / fname)
            log(f"backup {fname} -> {backup_dir.name}/{fname}")
    return backup_dir


def activate(zip_path: Path):
    if not OUTPUTS.exists():
        raise FileNotFoundError(f"outputs/ not found: {OUTPUTS}")

    log(f"v6 zip: {zip_path}  ({zip_path.stat().st_size/1024/1024:.0f} MB)")

    # Read zip contents
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        log(f"zip contains {len(names)} entries: {names}")
        present = {n.split("/")[-1]: n for n in names}

        # require the three critical files
        required = ["v6_projector.pt", "v6_gallery.faiss", "v6_gallery_meta.npz"]
        for r in required:
            if r not in present:
                raise RuntimeError(f"zip missing required file: {r}")

        # backup v5 first
        backup_dir = backup_current()

        # extract to a staging dir
        stage = OUTPUTS / "_v6_stage"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir()
        for arc in names:
            zf.extract(arc, stage)

        # swap each file
        for v6_name, target_name in V6_FILES.items():
            staged = stage / v6_name
            if not staged.exists():
                log(f"  skip (not in zip): {v6_name}")
                continue
            dst = OUTPUTS / target_name
            shutil.copy2(staged, dst)
            log(f"  installed {target_name} ({dst.stat().st_size/1024/1024:.1f} MB)")

        # cleanup staging
        shutil.rmtree(stage)

    log(f"v6 ACTIVE. v5 backed up at: {backup_dir.name}")


def rollback(backup_name: str):
    """Restore the most recent (or named) backup."""
    if backup_name:
        backup_dir = OUTPUTS / backup_name
    else:
        backups = list_backups()
        if not backups:
            raise RuntimeError("no backups found")
        backup_dir = backups[-1]
    if not backup_dir.exists():
        raise FileNotFoundError(f"backup not found: {backup_dir}")
    log(f"rolling back from {backup_dir.name}")
    for fname in BACKUP_TARGETS:
        src = backup_dir / fname
        if src.exists():
            shutil.copy2(src, OUTPUTS / fname)
            log(f"  restored {fname}")
    log("rollback done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=None,
                    help="path to cross_modal_v6.zip (default: auto-detect)")
    ap.add_argument("--rollback", nargs="?", const="", default=None,
                    help="rollback to last/named backup (pass name as arg)")
    args = ap.parse_args()

    try:
        if args.rollback is not None:
            rollback(args.rollback)
        else:
            zip_path = find_zip(args.zip)
            activate(zip_path)
    except Exception as e:
        print(f"[activate-v6] ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()