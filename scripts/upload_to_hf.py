"""
upload_to_hf.py
================

Upload the dataset to HuggingFace Hub so Colab can download it instantly.

Why HF over Drive:
  - No 2.5 GB zip corruption issues
  - HF CDN is fast (~50 MB/s sustained)
  - One-time upload, reused forever
  - Free for public datasets

Steps:
  1. Sign up at https://huggingface.co (free, 2 min)
  2. Get a write token at https://huggingface.co/settings/tokens
  3. Run: python upload_to_hf.py --token hf_xxx --repo-id yourname/BAH2026
  4. In Colab: !pip install -q huggingface_hub
              !huggingface-cli download yourname/BAH2026 --repo-type dataset --local-dir /content/data

Time on Colab: ~2-3 min for 2.5 GB via HF CDN (vs 30-60 min on Drive).
"""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True,
                    help="HF write token (https://huggingface.co/settings/tokens)")
    ap.add_argument("--repo-id", required=True,
                    help="e.g. 'yourname/BAH2026' (will be created as a dataset repo)")
    ap.add_argument("--root", type=Path, default=Path(r"D:\BAH2026"),
                    help="Project root containing EuroSAT/ and Sentinel/")
    args = ap.parse_args()

    api = HfApi(token=args.token)
    repo_type = "dataset"
    repo_id = args.repo_id

    print(f"[hf] creating dataset repo {repo_id} ...")
    create_repo(repo_id, repo_type=repo_type, token=args.token, exist_ok=True,
                private=False)
    print(f"[hf] uploading {args.root} (EuroSAT/ + Sentinel/) ...")
    print(f"[hf] this may take 10-30 minutes for 2.5 GB depending on internet")
    print(f"[hf] press Ctrl+C to cancel")

    # Upload the two folders individually so a failure in one doesn't kill the other
    for sub in ("EuroSAT", "Sentinel"):
        src = args.root / sub
        if not src.exists():
            print(f"[hf] WARN: missing {src}, skipping")
            continue
        print(f"[hf] uploading {src} ...")
        upload_folder(
            folder_path=str(src),
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo=sub,
            token=args.token,
            commit_message=f"upload {sub}",
            ignore_patterns=["*.pyc", "__pycache__/*", "._*"],
        )
        print(f"[hf] {sub} uploaded")

    print(f"\n[hf] DONE. Dataset available at:")
    print(f"     https://huggingface.co/datasets/{repo_id}")
    print(f"\n[hf] To use in Colab:")
    print(f"     !pip install -q huggingface_hub")
    print(f"     !huggingface-cli download {repo_id} --repo-type dataset --local-dir /content/BAH2026")


if __name__ == "__main__":
    main()