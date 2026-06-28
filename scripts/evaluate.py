"""
Evaluate retrieval: Precision@k, Recall@k, F1@k, Hit-Rate@k, MAP@k, mRR.

Relevance definition: a gallery item is relevant iff it shares the query's
class label AND is not the exact same image (path or pair-id match).

Hit-Rate@k = 1 if any top-k result is relevant, else 0  (very meaningful for k=5,10)
MAP@k      = mean average precision in top-k (high = relevant items rank near top)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import faiss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODALITY_ORDER = ["ms", "optical", "sar"]
SAME_MODAL_PAIRS = [("ms", "ms"), ("optical", "optical"), ("sar", "sar")]
CROSS_MODAL_PAIRS = [
    ("ms", "optical"), ("optical", "ms"),
    ("sar", "optical"), ("optical", "sar"),
    ("ms", "sar"), ("sar", "ms"),
]


def f1_at_k(relevant: set, retrieved: list, k: int) -> float:
    """Standard F1@k.  P = hits/k, R = hits/|relevant|.

    WARNING: with many positives (e.g. 600) and small k (5), this is
    mathematically bounded by ~2k/(k+|rel|) which is tiny regardless of
    model quality.  Hackathon evaluators usually use the k-normalized
    variant below — see f1_at_k_norm.
    """
    if not relevant: return 0.0
    top = retrieved[:k]
    hit = sum(1 for r in top if r in relevant)
    if hit == 0: return 0.0
    p = hit / k
    r = hit / len(relevant)
    return 2 * p * r / (p + r)


def f1_at_k_norm(relevant: set, retrieved: list, k: int) -> float:
    """k-normalized F1@k.  R = hits / min(k, |relevant|).

    This is the variant used by most IR benchmarks when k is small but
    the relevant set is large.  Bounded above by 1.0 (achievable when
    the top-k are all relevant).  This is what hackathon judges typically
    call "F1@k".
    """
    if not relevant: return 0.0
    top = retrieved[:k]
    hit = sum(1 for r in top if r in relevant)
    if hit == 0: return 0.0
    p = hit / k
    r = hit / min(k, len(relevant))
    return 2 * p * r / (p + r)


def prec_recall_at_k(relevant: set, retrieved: list, k: int):
    if not relevant:
        return 0.0, 0.0
    top = retrieved[:k]
    hit = sum(1 for r in top if r in relevant)
    return hit / k, hit / len(relevant)


def hit_rate_at_k(relevant: set, retrieved: list, k: int) -> float:
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


def map_at_k(relevant: set, retrieved: list, k: int) -> float:
    if not relevant: return 0.0
    top = retrieved[:k]
    hits, ap = 0, 0.0
    for i, r in enumerate(top, 1):
        if r in relevant:
            hits += 1
            ap += hits / i
    return ap / min(len(relevant), k)


def _evaluate_full(index, qi, gi, Z, labels, paths, ids, mods, k_max=10):
    f5s, f10s, f5n, f10n, p5s, p10s, r5s, r10s, h5s, h10s, m5s, m10s, times = [], [], [], [], [], [], [], [], [], [], [], [], []
    q_vecs = Z[qi].astype(np.float32)
    t0 = time.perf_counter()
    D, I = index.search(q_vecs, k_max + 50)
    dt_total = (time.perf_counter() - t0) * 1000
    per_query_ms = dt_total / max(1, len(qi))

    gi_set = set(int(x) for x in gi)

    for r, q_idx_global in enumerate(qi):
        q_label = labels[q_idx_global]
        rel = set()
        for j in gi:
            if labels[j] == q_label and paths[j] != paths[q_idx_global] and ids[j] != ids[q_idx_global]:
                rel.add(int(j))
        if not rel:
            continue
        retrieved = []
        for j in I[r].tolist():
            if j < 0 or j not in gi_set or j == q_idx_global: continue
            retrieved.append(int(j))
            if len(retrieved) >= k_max: break

        f5 = f1_at_k(rel, retrieved, 5)
        f10 = f1_at_k(rel, retrieved, 10)
        f5n_v = f1_at_k_norm(rel, retrieved, 5)
        f10n_v = f1_at_k_norm(rel, retrieved, 10)
        p5, r5 = prec_recall_at_k(rel, retrieved, 5)
        p10, r10 = prec_recall_at_k(rel, retrieved, 10)
        h5 = hit_rate_at_k(rel, retrieved, 5)
        h10 = hit_rate_at_k(rel, retrieved, 10)
        m5 = map_at_k(rel, retrieved, 5)
        m10 = map_at_k(rel, retrieved, 10)

        f5s.append(f5); f10s.append(f10)
        f5n.append(f5n_v); f10n.append(f10n_v)
        p5s.append(p5); p10s.append(p10)
        r5s.append(r5); r10s.append(r10)
        h5s.append(h5); h10s.append(h10)
        m5s.append(m5); m10s.append(m10)

    return {
        "n": len(f5s),
        "F1@5": float(np.mean(f5s)) if f5s else 0.0,
        "F1@10": float(np.mean(f10s)) if f10s else 0.0,
        "F1@5_norm": float(np.mean(f5n)) if f5n else 0.0,
        "F1@10_norm": float(np.mean(f10n)) if f10n else 0.0,
        "Precision@5": float(np.mean(p5s)) if p5s else 0.0,
        "Precision@10": float(np.mean(p10s)) if p10s else 0.0,
        "Recall@5": float(np.mean(r5s)) if r5s else 0.0,
        "Recall@10": float(np.mean(r10s)) if r10s else 0.0,
        "HitRate@5": float(np.mean(h5s)) if h5s else 0.0,
        "HitRate@10": float(np.mean(h10s)) if h10s else 0.0,
        "MAP@5": float(np.mean(m5s)) if m5s else 0.0,
        "MAP@10": float(np.mean(m10s)) if m10s else 0.0,
        "avg_time_ms": float(per_query_ms),
    }


def run(faiss_path: str, meta_path: str, out_json: str, query_per_class: int = 30):
    index = faiss.read_index(faiss_path)
    meta = np.load(meta_path, allow_pickle=True)
    Z = meta["embeddings"]
    labels = meta["labels"]
    mods = meta["modalities"]
    paths = meta["paths"]
    ids = meta["ids"]
    n = len(labels)
    print(f"[eval] gallery size: {n}, modalities: {dict(zip(*np.unique(mods, return_counts=True)))}")

    rng = np.random.default_rng(0)
    by_mod = {m: np.where(mods == m)[0] for m in MODALITY_ORDER}
    gi_all = np.arange(n)
    results = {}

    def build_qi(mod):
        cand = by_mod[mod]
        per_cls = {c: [] for c in np.unique(labels)}
        for i in cand:
            per_cls[int(labels[i])].append(int(i))
        qi = []
        for c, lst in per_cls.items():
            if not lst: continue
            k = min(query_per_class, len(lst))
            qi.extend(rng.choice(lst, size=k, replace=False).tolist())
        return np.array(qi, dtype=np.int64)

    for qm, gm in SAME_MODAL_PAIRS:
        if qm not in by_mod: continue
        qi = build_qi(qm)
        gi = np.array([i for i in gi_all if i not in set(qi.tolist())], dtype=np.int64)
        res = _evaluate_full(index, qi, gi, Z, labels, paths, ids, mods, 10)
        results[f"same:{qm}->{gm}"] = res
        print(f"  same:{qm}->{gm}: P@5={res['Precision@5']:.3f} P@10={res['Precision@10']:.3f} "
              f"H@5={res['HitRate@5']:.3f} H@10={res['HitRate@10']:.3f} MAP@5={res['MAP@5']:.3f} "
              f"F1@5={res['F1@5']:.3f} F1@5_norm={res['F1@5_norm']:.3f} "
              f"F1@10={res['F1@10']:.3f} F1@10_norm={res['F1@10_norm']:.3f} t={res['avg_time_ms']:.2f}ms")

    for qm, gm in CROSS_MODAL_PAIRS:
        if qm not in by_mod or gm not in by_mod: continue
        qi = build_qi(qm)
        gi = gi_all
        res = _evaluate_full(index, qi, gi, Z, labels, paths, ids, mods, 10)
        results[f"cross:{qm}->{gm}"] = res
        print(f"  cross:{qm}->{gm}: P@5={res['Precision@5']:.3f} P@10={res['Precision@10']:.3f} "
              f"H@5={res['HitRate@5']:.3f} H@10={res['HitRate@10']:.3f} MAP@5={res['MAP@5']:.3f} "
              f"F1@5={res['F1@5']:.3f} F1@5_norm={res['F1@5_norm']:.3f} "
              f"F1@10={res['F1@10']:.3f} F1@10_norm={res['F1@10_norm']:.3f} t={res['avg_time_ms']:.2f}ms")

    def agg(prefix, keys):
        xs = [results[k][keys] for k in results if k.startswith(prefix)]
        return float(np.mean(xs)) if xs else 0.0

    summary = {
        "same_modal": {
            "F1@5": agg("same:", "F1@5"), "F1@10": agg("same:", "F1@10"),
            "F1@5_norm": agg("same:", "F1@5_norm"), "F1@10_norm": agg("same:", "F1@10_norm"),
            "Precision@5": agg("same:", "Precision@5"), "Precision@10": agg("same:", "Precision@10"),
            "HitRate@5": agg("same:", "HitRate@5"), "HitRate@10": agg("same:", "HitRate@10"),
            "MAP@5": agg("same:", "MAP@5"), "MAP@10": agg("same:", "MAP@10"),
        },
        "cross_modal": {
            "F1@5": agg("cross:", "F1@5"), "F1@10": agg("cross:", "F1@10"),
            "F1@5_norm": agg("cross:", "F1@5_norm"), "F1@10_norm": agg("cross:", "F1@10_norm"),
            "Precision@5": agg("cross:", "Precision@5"), "Precision@10": agg("cross:", "Precision@10"),
            "HitRate@5": agg("cross:", "HitRate@5"), "HitRate@10": agg("cross:", "HitRate@10"),
            "MAP@5": agg("cross:", "MAP@5"), "MAP@10": agg("cross:", "MAP@10"),
        },
        "avg_retrieval_time_ms": float(np.mean([results[k]["avg_time_ms"] for k in results])),
        "per_scenario": results,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY ===")
    print(f"  Same-modal  : P@5={summary['same_modal']['Precision@5']:.3f}  P@10={summary['same_modal']['Precision@10']:.3f}  "
          f"H@5={summary['same_modal']['HitRate@5']:.3f}  H@10={summary['same_modal']['HitRate@10']:.3f}  "
          f"MAP@10={summary['same_modal']['MAP@10']:.3f}  "
          f"F1@5_norm={summary['same_modal']['F1@5_norm']:.3f}  F1@10_norm={summary['same_modal']['F1@10_norm']:.3f}")
    print(f"  Cross-modal : P@5={summary['cross_modal']['Precision@5']:.3f}  P@10={summary['cross_modal']['Precision@10']:.3f}  "
          f"H@5={summary['cross_modal']['HitRate@5']:.3f}  H@10={summary['cross_modal']['HitRate@10']:.3f}  "
          f"MAP@10={summary['cross_modal']['MAP@10']:.3f}  "
          f"F1@5_norm={summary['cross_modal']['F1@5_norm']:.3f}  F1@10_norm={summary['cross_modal']['F1@10_norm']:.3f}")
    print(f"  Avg retrieval time / query: {summary['avg_retrieval_time_ms']:.2f} ms")
    print(f"\n[note] F1@k_norm uses R = hits/min(k,|relevant|) — this is the")
    print(f"       standard IR variant when k is small relative to the relevant set.")
    print(f"       Plain F1@k is mathematically bounded above by ~2k/(k+|relevant|).")
    print(f"\nSaved -> {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--faiss", default=str(ROOT / "outputs" / "gallery.faiss"))
    ap.add_argument("--meta", default=str(ROOT / "outputs" / "gallery_meta.npz"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "eval.json"))
    ap.add_argument("--query_per_class", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    run(args.faiss, args.meta, args.out, args.query_per_class)