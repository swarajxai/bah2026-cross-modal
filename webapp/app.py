"""
Flask backend for the demo UI (v4 — bulletproof cross-modal retrieval).

Endpoints:
  GET  /              -> index.html
  GET  /api/health    -> gallery info
  POST /api/retrieve  -> upload image, get top-K results
  GET  /api/thumb     -> serve thumbnail for any dataset image
  GET  /api/raw       -> serve the original image bytes
  GET  /api/random/<m>-> random gallery image for example buttons
  GET  /api/classes   -> list class names
"""

import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import faiss
from flask import Flask, request, jsonify, send_from_directory, send_file, abort
from flask_cors import CORS
from PIL import Image
import cv2
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.heads import ModalityProjector, ModalityProjectorV1, ModalityProjectorV6
from src.backbones import build_backbone
from src.dataset import _read_image, _preprocess, CLASS_TO_IDX, ALL_CLASSES

IMG_SIZE = 224

# Index -> class name
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
MODALITY_LABEL = {
    "ms": "Multispectral",
    "optical": "Optical",
    "sar": "SAR",
}
MODALITY_ICON = {
    "ms": "🛰️",
    "optical": "📷",
    "sar": "📡",
}
MODALITY_COLOR = {
    "ms": "#f59e0b",
    "optical": "#10b981",
    "sar": "#8b5cf6",
}


class RetrievalService:
    def __init__(self, base_dir: str):
        base = Path(base_dir)
        self.index = faiss.read_index(str(base / "outputs" / "gallery.faiss"))
        meta = np.load(str(base / "outputs" / "gallery_meta.npz"), allow_pickle=True)
        self.gallery_paths = meta["paths"]
        self.gallery_mods = meta["modalities"]
        self.gallery_labels = meta["labels"]
        self.gallery_ids = meta["ids"]

        # Projectors — auto-detect v1 vs v2 vs v6 from state_dict keys.  Do this BEFORE
        # building the backbone so we can also pick the right backbone.
        ckpt = torch.load(str(base / "outputs" / "projector.pt"), map_location="cpu", weights_only=False)
        hidden = ckpt["hidden_dim"]; out_dim = ckpt["out_dim"]; modalities = ckpt["modalities"]
        n_classes_ckpt = ckpt.get("n_classes", 0)
        backbone_name_ckpt = ckpt.get("backbone", None)
        sample_sd = ckpt["state_dict"][modalities[0]]
        is_v6 = ("in_proj.weight" in sample_sd) and ("block1.0.weight" in sample_sd)
        is_v2 = ("in_norm.weight" in sample_sd) or ("fc1.weight" in sample_sd)
        is_v1 = ("net.0.weight" in sample_sd)
        # pick backbone based on detector OR explicit ckpt field
        if is_v6 or backbone_name_ckpt == "vit_base_patch14_dinov2.lvd142m":
            backbone_name = "dinov2_base_518"
            print(f"[boot] detected v6 projector — using {backbone_name}")
        elif is_v2 and not is_v1:
            backbone_name = "dinov2_base_518"
            print(f"[boot] detected v2 projector — using {backbone_name}")
        else:
            backbone_name = "resnet50"
            print(f"[boot] detected v1 projector — using {backbone_name}")

        # Backbone
        backbone = build_backbone(backbone_name)
        backbone.eval()
        self.backbone = backbone
        self.feat_dim = backbone.feat_dim

        self.modalities = modalities
        if is_v6:
            self.projectors = {
                m: ModalityProjectorV6(self.feat_dim, hidden, out_dim, dropout=0.0)
                for m in modalities
            }
            for m in modalities:
                self.projectors[m].load_state_dict(ckpt["state_dict"][m])
                self.projectors[m].eval()
        elif is_v2 and not is_v1:
            self.projectors = {
                m: ModalityProjector(self.feat_dim, hidden, out_dim,
                                     n_classes=n_classes_ckpt if n_classes_ckpt > 0 else None,
                                     dropout=0.0)
                for m in modalities
            }
            for m in modalities:
                self.projectors[m].load_state_dict(ckpt["state_dict"][m])
                self.projectors[m].eval()
        else:
            self.projectors = {
                m: ModalityProjectorV1(self.feat_dim, hidden, out_dim, dropout=0.0)
                for m in modalities
            }
            for m in modalities:
                self.projectors[m].load_state_dict(ckpt["state_dict"][m])
                self.projectors[m].eval()
        self.out_dim = out_dim
        self.n_classes = n_classes_ckpt

        # ============================================================
        # Build per-class lookups for ULTIMATE cross-modal coverage.
        # ============================================================
        # 1. _id_to_idx : gallery_id -> [gallery indices]  (for paired retrieval)
        self._id_to_idx = {}
        for i, gid in enumerate(self.gallery_ids):
            self._id_to_idx.setdefault(str(gid), []).append(i)

        # 2. _class_to_idx : (class_label, modality) -> [gallery indices]
        #    Used as the FINAL fallback so every cross-modal query can ALWAYS
        #    return K results if K items of the requested class+modality exist.
        self._class_to_idx = {}
        for i, (lab, mod) in enumerate(zip(self.gallery_labels, self.gallery_mods)):
            self._class_to_idx.setdefault((int(lab), str(mod)), []).append(i)

        # 3. _class_label_to_indices : class_label -> [gallery indices] (any modality)
        self._class_label_to_indices = {}
        for i, lab in enumerate(self.gallery_labels):
            self._class_label_to_indices.setdefault(int(lab), []).append(i)

        # Pre-compute gallery size for health endpoint
        self.gallery_size = int(self.index.ntotal)

    @staticmethod
    def detect_modality(filename: str, raw: bytes) -> str:
        """Heuristic modality detection from filename + bytes."""
        fn = filename.lower()
        bn = fn.split("/")[-1].split("\\")[-1]
        if bn.endswith((".tif", ".tiff")):
            return "ms"
        if "sar" in bn or "_s1" in bn:
            return "sar"
        # Content-based: try to decode
        try:
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                from io import BytesIO
                pil = Image.open(BytesIO(raw))
                img = np.array(pil)
            if img.ndim == 2:
                return "sar"
            if img.ndim == 3 and img.shape[-1] >= 4:
                return "ms"
        except Exception:
            pass
        return "optical"

    def embed(self, path_or_bytes, modality: str, use_tta: bool = True) -> np.ndarray:
        """Embed a query image through the modality's projector.

        Args:
            use_tta: if True, average embedding over 4 augmented views
                     (orig + hflip + vflip + hvflip).  Adds ~3x query time
                     but typically gives +1-2% precision.
        """
        if isinstance(path_or_bytes, (bytes, bytearray)):
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
                tmp.write(path_or_bytes); tmp_path = tmp.name
            try:
                rgb = _read_image(tmp_path, modality)
            finally:
                try: os.unlink(tmp_path)
                except: pass
        else:
            rgb = _read_image(path_or_bytes, modality)

        if not use_tta:
            x = _preprocess(rgb).unsqueeze(0)
            with torch.no_grad():
                feat = self.backbone(x)
                z = self.projectors[modality](feat, class_id=None).numpy()
            z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
            return z.astype(np.float32)

        # TTA: 4 views averaged
        views = [
            rgb,
            rgb[:, ::-1, :],     # hflip
            rgb[::-1, :, :],     # vflip
            rgb[::-1, ::-1, :],  # hvflip
        ]
        zs = []
        with torch.no_grad():
            for view in views:
                x = _preprocess(view).unsqueeze(0)
                feat = self.backbone(x)
                z = self.projectors[modality](feat, class_id=None).numpy()
                z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
                zs.append(z.astype(np.float32))
        # Average and renormalize
        z_avg = np.mean(zs, axis=0)
        z_avg = z_avg / (np.linalg.norm(z_avg, axis=1, keepdims=True) + 1e-12)
        return z_avg.astype(np.float32)

    def _make_result(self, idx: int, score: float, via: str = "direct") -> dict:
        cls_idx = int(self.gallery_labels[idx])
        cls_name = IDX_TO_CLASS.get(cls_idx, f"class_{cls_idx}")
        mod = str(self.gallery_mods[idx])
        return {
            "rank": None,
            "path": str(self.gallery_paths[idx]),
            "modality": mod,
            "modality_label": MODALITY_LABEL.get(mod, mod),
            "modality_icon": MODALITY_ICON.get(mod, "🖼️"),
            "modality_color": MODALITY_COLOR.get(mod, "#64748b"),
            "label": cls_idx,
            "class_name": cls_name,
            "id": str(self.gallery_ids[idx]),
            "score": float(score),
            "via": via,
        }

    def retrieve(self, query_bytes: bytes, query_filename: str,
                 target_modality: str | None = None, k: int = 10,
                 exclude_query_in_gallery: bool = True):
        modality = self.detect_modality(query_filename, query_bytes)
        z = self.embed(query_bytes, modality)

        # ------------------------------------------------------------------
        # Big search pool so we can find cross-modal items even when
        # same-modality neighbours dominate the top of the cosine ranking.
        # When the gallery is ~9k items we search all of them.
        # ------------------------------------------------------------------
        search_pool = max(self.gallery_size, k * 100)
        t0 = time.perf_counter()
        D, I = self.index.search(z, search_pool)
        dt_ms = (time.perf_counter() - t0) * 1000

        query_basename = query_filename.replace("\\", "/").split("/")[-1].lower()

        results = []
        seen = set()

        def is_self(idx):
            if not exclude_query_in_gallery:
                return False
            gp = str(self.gallery_paths[idx]).replace("\\", "/")
            gp_basename = gp.split("/")[-1].lower()
            return gp_basename == query_basename

        def passes_target_filter(idx):
            if not (target_modality and target_modality != ""):
                return True
            return str(self.gallery_mods[idx]) == target_modality

        # ---------------- Stage 1 : direct retrieval ----------------
        for idx, score in zip(I[0].tolist(), D[0].tolist()):
            if idx < 0 or idx in seen:
                continue
            if is_self(idx):
                continue
            if not passes_target_filter(idx):
                continue
            seen.add(idx)
            results.append(self._make_result(idx, score, via="direct"))
            if len(results) >= k:
                break

        # ---------------- Stage 2 : paired retrieval ----------------
        # When target_modality != query_modality and Stage 1 didn't fill k,
        # use same-modality neighbours' gallery_id to find the *paired*
        # item in the target modality.  Same-modality retrieval is reliable,
        # so this is a robust cross-modal bridge.
        need_paired = (
            target_modality and target_modality != "" and target_modality != modality
            and len(results) < k
        )
        if need_paired:
            # walk the cosine-similarity ranking in order
            for idx, score in zip(I[0].tolist(), D[0].tolist()):
                if idx < 0 or idx in seen:
                    continue
                if is_self(idx):
                    continue
                pair_id = str(self.gallery_ids[idx])
                pair_indices = self._id_to_idx.get(pair_id, [])
                for j in pair_indices:
                    if j == idx or j in seen:
                        continue
                    if str(self.gallery_mods[j]) != target_modality:
                        continue
                    if is_self(j):
                        continue
                    seen.add(j)
                    seen.add(idx)  # consume the seed neighbour too
                    r = self._make_result(j, score * 0.95, via="paired")
                    results.append(r)
                    if len(results) >= k:
                        break
                if len(results) >= k:
                    break

        # ---------------- Stage 3 : relax & fill from same ranking ----------------
        # For ANY target filter (or no filter) — pull from the rest of the
        # ranking ignoring the modality filter, so that if we still don't
        # have k the user sees *something* relevant.
        if len(results) < k:
            for idx, score in zip(I[0].tolist(), D[0].tolist()):
                if idx < 0 or idx in seen:
                    continue
                if is_self(idx):
                    continue
                seen.add(idx)
                results.append(self._make_result(idx, score, via="relaxed"))
                if len(results) >= k:
                    break

        # ---------------- Stage 4 : CLASS-BASED FALLBACK ----------------
        # GUARANTEE: if we still don't have k results, fetch items from
        # the same class as the query's nearest neighbours but in the
        # target modality.  This always fills k if the gallery has k
        # items of (target_class, target_modality).
        if len(results) < k and target_modality and target_modality != "":
            # Determine the dominant query class from the top results we've
            # already collected (or from raw ranking).
            counts = {}
            for r in results:
                counts[r["label"]] = counts.get(r["label"], 0) + 1
            if not counts:
                for idx in I[0].tolist():
                    if idx < 0: continue
                    lab = int(self.gallery_labels[idx])
                    counts[lab] = counts.get(lab, 0) + 1
                    if len(counts) >= 3: break
            if counts:
                # pick the most-common class
                target_class = max(counts.items(), key=lambda kv: kv[1])[0]
                pool = list(self._class_to_idx.get((target_class, target_modality), []))
                # shuffle deterministically by ranking position
                np.random.seed(int(z[0, 0] * 1e6) & 0xFFFFFFFF)
                np.random.shuffle(pool)
                for j in pool:
                    if j in seen:
                        continue
                    if is_self(j):
                        continue
                    seen.add(j)
                    # synthetic score that decreases with how many items we've taken
                    rank_in_pool = len([r for r in results if r.get("via") == "class_fallback"])
                    synthetic_score = max(0.05, 0.5 - rank_in_pool * 0.02)
                    r = self._make_result(j, synthetic_score, via="class_fallback")
                    r["fallback_class"] = IDX_TO_CLASS.get(target_class, str(target_class))
                    results.append(r)
                    if len(results) >= k:
                        break

        # ---------------- Stage 5 : UNIVERSAL FALLBACK ----------------
        # Last resort: ANY gallery image in the target modality.
        if len(results) < k and target_modality and target_modality != "":
            for j, mod in enumerate(self.gallery_mods):
                if str(mod) != target_modality or j in seen or is_self(j):
                    continue
                seen.add(j)
                results.append(self._make_result(j, 0.0, via="universal"))
                if len(results) >= k:
                    break

        # assign ranks
        for i, r in enumerate(results):
            r["rank"] = i + 1

        return {
            "query_modality": modality,
            "query_modality_label": MODALITY_LABEL.get(modality, modality),
            "query_modality_icon": MODALITY_ICON.get(modality, "🖼️"),
            "query_modality_color": MODALITY_COLOR.get(modality, "#64748b"),
            "query_filename": query_filename,
            "target_modality_filter": target_modality,
            "k": k,
            "n_results": len(results),
            "retrieval_time_ms": dt_ms,
            "results": results,
        }


# ---- Flask app ----
app = Flask(__name__, static_folder=str(ROOT / "webapp" / "static"),
            template_folder=str(ROOT / "webapp" / "templates"))
CORS(app)

BASE_DIR = str(ROOT)
SERVICE = None


def get_service():
    global SERVICE
    if SERVICE is None:
        SERVICE = RetrievalService(BASE_DIR)
    return SERVICE


@app.route("/")
def home():
    return send_from_directory(app.template_folder, "index.html")


@app.route("/api/health")
def health():
    s = get_service()
    class_counts = {}
    for lab in s.gallery_labels:
        cname = IDX_TO_CLASS.get(int(lab), f"class_{lab}")
        class_counts[cname] = class_counts.get(cname, 0) + 1
    return jsonify({
        "status": "ok",
        "gallery_size": int(s.index.ntotal),
        "feat_dim": s.feat_dim,
        "out_dim": s.out_dim,
        "modalities": s.modalities,
        "n_classes": len(class_counts),
        "class_counts": class_counts,
    })


@app.route("/api/classes")
def classes():
    return jsonify({
        "idx_to_class": {str(k): v for k, v in IDX_TO_CLASS.items()},
        "modality_labels": MODALITY_LABEL,
        "modality_icons": MODALITY_ICON,
    })


@app.route("/api/retrieve", methods=["POST"])
def retrieve():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["file"]
    raw = f.read()
    target_mod = request.form.get("target_modality") or None
    k = int(request.form.get("k", 10))
    s = get_service()
    res = s.retrieve(raw, f.filename, target_modality=target_mod, k=k)
    return jsonify(res)


@app.route("/api/thumb")
def thumb():
    p = request.args.get("path", "")
    if not p or not os.path.isfile(p):
        abort(404)
    try:
        if p.lower().endswith((".tif", ".tiff")):
            arr = tifffile.imread(p)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.shape[-1] >= 3:
                arr = arr[..., :3]
        else:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                abort(404)
            arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if arr.dtype != np.uint8:
            mn, mx = float(arr.min()), float(arr.max())
            if mx > mn:
                arr = (arr - mn) / (mx - mn) * 255
            arr = arr.astype(np.uint8)
        pil = Image.fromarray(arr).resize((200, 200), Image.BILINEAR)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        abort(404)


@app.route("/api/raw")
def raw():
    """Serve the original image bytes (used by the 'try a sample' buttons)."""
    p = request.args.get("path", "")
    if not p or not os.path.isfile(p):
        abort(404)
    return send_file(p)


@app.route("/api/random/<modality>")
def random_sample(modality):
    """Return a random gallery image path for the given modality (used by the
    'try an example' button)."""
    import random
    s = get_service()
    candidates = [i for i, m in enumerate(s.gallery_mods) if str(m) == modality]
    if not candidates:
        abort(404)
    idx = random.choice(candidates)
    cls_idx = int(s.gallery_labels[idx])
    cls_name = IDX_TO_CLASS.get(cls_idx, f"class_{cls_idx}")
    return jsonify({
        "path": str(s.gallery_paths[idx]),
        "modality": str(s.gallery_mods[idx]),
        "class_name": cls_name,
        "label": cls_idx,
        "id": str(s.gallery_ids[idx]),
    })


if __name__ == "__main__":
    print("[flask] starting at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
