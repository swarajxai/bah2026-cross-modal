"""One-shot pipeline runner: extract -> train -> index -> evaluate."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    from scripts.extract_features import extract
    from scripts.train_projectors import train
    from scripts.build_index import build
    from scripts.evaluate import run as eval_run

    feats = str(ROOT / "outputs" / "features.npz")
    proj  = str(ROOT / "outputs" / "projector.pt")
    faiss_path = str(ROOT / "outputs" / "gallery.faiss")
    meta  = str(ROOT / "outputs" / "gallery_meta.npz")
    ej    = str(ROOT / "outputs" / "eval.json")
    os.makedirs(ROOT / "outputs", exist_ok=True)

    print("\n[1/4] Feature extraction")
    if not os.path.isfile(feats):
        extract("resnet50", feats, batch_size=32)
    else:
        print(f"  -> {feats} already exists, skipping")

    print("\n[2/4] Training modality projectors")
    if not os.path.isfile(proj):
        train(feats, proj, epochs=12)
    else:
        print(f"  -> {proj} already exists, skipping")

    print("\n[3/4] Building FAISS index")
    if not os.path.isfile(faiss_path):
        build(feats, proj, faiss_path, meta)
    else:
        print(f"  -> {faiss_path} already exists, skipping")

    print("\n[4/4] Evaluation")
    eval_run(faiss_path, meta, ej, query_per_class=30)


if __name__ == "__main__":
    main()