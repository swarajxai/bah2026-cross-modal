#!/usr/bin/env python3
"""
Upload model artifacts to HuggingFace Hub dataset repo.

Usage:  python scripts/upload_weights_to_hf.py --repo swarajkamila/bah2026-weights
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, login


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="swarajkamila/bah2026-weights",
                        help="HF dataset repo (user/repo)")
    parser.add_argument("--outputs", default="outputs",
                        help="Local outputs directory")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="HF write token (or env HF_TOKEN)")
    args = parser.parse_args()

    if args.token:
        login(token=args.token)

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo, repo_type="dataset", exist_ok=True,
                    token=args.token)

    files = ["projector.pt", "gallery.faiss", "gallery_meta.npz", "features.npz"]
    for fname in files:
        local = Path(args.outputs) / fname
        if not local.exists():
            print(f"[skip] {fname}: not found locally")
            continue
        size_mb = local.stat().st_size / (1024*1024)
        print(f"[upload] {fname} ({size_mb:.1f} MB) -> {args.repo}")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=fname,
            repo_id=args.repo,
            repo_type="dataset",
            token=args.token,
        )
        print(f"[upload] {fname}: OK")

    print(f"\nAll files uploaded to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()