"""Debug retrieval: inspect top neighbors for random queries."""
import numpy as np
import faiss

idx = faiss.read_index(r'D:\BAH2026\cross_modal_retrieval\outputs\gallery.faiss')
meta = np.load(r'D:\BAH2026\cross_modal_retrieval\outputs\gallery_meta.npz', allow_pickle=True)
emb = meta['embeddings']
labels = meta['labels']
mods = meta['modalities']
paths = meta['paths']

rng = np.random.default_rng(0)
# pick one query per modality
seen = set()
for q_idx in range(len(labels)):
    if mods[q_idx] in seen:
        continue
    seen.add(mods[q_idx])
    if len(seen) >= 3:
        break
    q_lbl = labels[q_idx]
    q_mod = mods[q_idx]
    q_path = paths[q_idx].split("\\")[-1]
    D, I = idx.search(emb[q_idx:q_idx+1], 11)
    print(f"\nQuery: {q_path} | mod={q_mod} | label={q_lbl}")
    for r, (j, score) in enumerate(zip(I[0], D[0])):
        if j == q_idx: continue
        p = paths[j].split("\\")[-1]
        match = "OK" if labels[j] == q_lbl else "--"
        print(f"  #{r} (score={score:.3f}) {match} {mods[j]:8s} label={labels[j]:2d} {p}")