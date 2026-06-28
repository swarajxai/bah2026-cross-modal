"""
Colab transfer helper - downloads v6 artifacts from Colab and integrates them.

This script helps the user get their v6 results from Google Colab back
into the local project.  Two options:

  1. MANUAL: user downloads `cross_modal_v6.zip` from Colab to local disk.
     Run:  python scripts/colab_transfer.py --zip <path-to-zip>
     -> unpacked into outputs/v6_projector.pt, v6_gallery.faiss, v6_gallery_meta.npz, v6_features.npz
     -> originals backed up to outputs/backup_v5/
     -> activated (projector.pt, gallery.faiss, gallery_meta.npz swapped)

  2. GDRIVE: if user mounted Drive and saved outputs to /content/drive/...
     Run:  python scripts/colab_transfer.py --drive <path-in-drive>
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

ap = argparse.ArgumentParser()
ap.add_argument("--zip", help="path to downloaded cross_modal_v6.zip from Colab")
ap.add_argument("--drive", help="path inside Google Drive where Colab saved outputs (e.g. /content/drive/MyDrive/cross_modal_v6)")
ap.add_argument("--dry-run", action="store_true", help="just show what would happen")
args = ap.parse_args()

if not args.zip and not args.drive:
    ap.error("provide either --zip <path> or --drive <path>")

# find source files
src = {}
if args.zip:
    zpath = Path(args.zip)
    if not zpath.exists():
        sys.exit(f"zip not found: {zpath}")
    print(f"[transfer] reading zip: {zpath}  ({zpath.stat().st_size / 1e6:.1f} MB)")
    extract_dir = OUT / "_colab_v6_extract"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(extract_dir)
    # The zip was made with -j so all files are at top level
    for f in extract_dir.iterdir():
        if f.name == "v6_projector.pt":
            src["projector"] = f
        elif f.name == "v6_gallery.faiss":
            src["faiss"] = f
        elif f.name == "v6_gallery_meta.npz":
            src["meta"] = f
        elif f.name == "features_dinov2.npz":
            src["features"] = f

elif args.drive:
    dpath = Path(args.drive)
    if not dpath.exists():
        sys.exit(f"drive path not found: {dpath}")
    for name, key in [("v6_projector.pt", "projector"),
                       ("v6_gallery.faiss", "faiss"),
                       ("v6_gallery_meta.npz", "meta"),
                       ("features_dinov2.npz", "features")]:
        p = dpath / name
        if p.exists():
            src[key] = p

print(f"[transfer] found: {sorted(src.keys())}")
required = {"projector", "faiss", "meta"}
missing = required - set(src.keys())
if missing:
    sys.exit(f"[transfer] ERROR - missing files: {missing}")

if args.dry_run:
    print("\n[transfer] DRY RUN - nothing copied")
    print("Would backup outputs/ -> outputs/backup_v5/")
    print(f"Would copy {src['projector']} -> outputs/projector.pt")
    print(f"Would copy {src['faiss']} -> outputs/gallery.faiss")
    print(f"Would copy {src['meta']} -> outputs/gallery_meta.npz")
    sys.exit(0)

# backup current outputs (v5 / v6-prev) into backup_v5
backup = OUT / "backup_v5"
if backup.exists():
    shutil.rmtree(backup)
backup.mkdir()
for fname in ["projector.pt", "gallery.faiss", "gallery_meta.npz", "features.npz"]:
    src_path = OUT / fname
    if src_path.exists():
        shutil.copy2(src_path, backup / fname)
        print(f"[transfer] backed up {fname}")

# also save features if present
if "features" in src:
    shutil.copy2(src["features"], OUT / "features_dinov2.npz")
    print(f"[transfer] saved features_dinov2.npz")

# activate v6
shutil.copy2(src["projector"], OUT / "projector.pt")
shutil.copy2(src["faiss"],     OUT / "gallery.faiss")
shutil.copy2(src["meta"],      OUT / "gallery_meta.npz")
print(f"\n[transfer] v6 activated:")
print(f"   outputs/projector.pt        <- {src['projector'].name}")
print(f"   outputs/gallery.faiss      <- {src['faiss'].name}")
print(f"   outputs/gallery_meta.npz   <- {src['meta'].name}")
print(f"   outputs/backup_v5/         <- previous v5 weights (rollback)")

# cleanup extract dir if we made one
if args.zip and extract_dir.exists():
    shutil.rmtree(extract_dir)

print("\n[transfer] done.  next: restart Flask server.")