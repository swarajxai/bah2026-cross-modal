# BAH 2026 Hackathon — Presentation Content

> 10 slides ka content. Tum PowerPoint me copy-paste kar sakte ho, ya main python-pptx script likh du jo auto-PPT banaye.
>
> **Theme**: Dark blue gradient, ISRO-style, satellite imagery backgrounds

---

## Slide 1 — Title

**Title:** Cross-Modal Satellite Image Retrieval Using Multi-Sensor Remote Sensing Data

**Subtitle:** BAH 2026 Hackathon — Team [Your Team Name]

**Author line:** [Team Member 1] · [Team Member 2] · [Team Member 3]

**Footer:** ISRO BAH 2026 · [Date]

---

## Slide 2 — Problem Statement

**Title:** Problem Statement #11 — Cross-Modal Retrieval

**Bullets:**
- Satellite archives hold petabytes across **optical, multispectral, SAR, hyperspectral** sensors
- Same location, different sensors → different physical characteristics
- Metadata-based search fails when seasonal/sensor conditions differ
- **Goal:** One query → top-K similar images, irrespective of sensor modality

**Visual:** Grid showing 3 sensor types of same land patch (Optical / MS / SAR)

---

## Slide 3 — Use Cases & Objectives

**Title:** Objectives

**Bullets:**
- ✅ **Same-modal retrieval** — optical→optical, SAR→SAR, MS→MS
- ✅ **Cross-modal retrieval** — optical↔SAR, optical↔MS, MS↔SAR
- ✅ **Top-5 and Top-10** ranked lists
- ✅ **<1 ms per query** retrieval time
- ✅ **F1@5, F1@10, P@k, Hit@k, MAP@k** reported

**Real-world impact bullets:**
- Disaster response (cloud-penetrating SAR)
- Agriculture monitoring (multi-season optical)
- Defense intelligence (cross-sensor correlation)

---

## Slide 4 — Dataset

**Title:** Multi-Sensor Dataset

**Table:**

| Source | Modality | Format | Classes | Samples |
|---|---|---|---|---|
| EuroSAT | Multispectral (s1) | .tif | 10 | ~30K |
| EuroSAT | Optical (s2) | .jpg | 10 | ~30K |
| Sentinel | SAR (s1) | .png | 4 | ~2K |
| Sentinel | Optical (s2) | .png | 4 | ~2K |
| **Gallery** | **3 modalities** | mixed | **14** | **~9.2K** |

**Pairing logic:**
- EuroSAT: paired by image id (`PermanentCrop_1.tif` ↔ `.jpg`)
- Sentinel: paired by patch id (`_p10.png` ↔ `_p10.png`)

---

## Slide 5 — System Architecture

**Title:** End-to-End Pipeline

**Flow diagram (use boxes + arrows):**

```
[Raw Images]
    ↓
[DINOv2-Base @518]  ← Foundation model, 768-D features
    ↓
[Per-modality Projector]  ← ModalityProjectorV6 (1024 hidden, 2 residual blocks)
    ↓
[256-D Shared Embedding]  ← L2-normalized → cosine space
    ↓
[FAISS IndexFlatIP]  ← ~9.2K vectors, ~10 MB on disk
    ↓
[Top-K Retrieval]  ← 0.02 ms per query
```

**Tech stack:** Python · PyTorch · timm (DINOv2) · FAISS · Flask

---

## Slide 6 — Methodology: 4-Stage Pipeline

**Title:** Methodology

**Stage 1 — Feature Extraction:**
- DINOv2-Base pretrained on 142M curated images (LVD-142M)
- 4-view Test-Time Augmentation (orig + hflip + vflip + hvflip)
- Output: 768-D embeddings, mean-pooled across views

**Stage 2 — Shared Embedding Training:**
- 2-block residual MLP projector (1024 hidden, GELU, LayerNorm)
- Combined loss: 0.30·Triplet + 0.35·InfoNCE + 0.20·CrossModal + 0.15·Prototype
- 50 epochs, AdamW, cosine schedule with warmup
- **Class-balanced sampling** — every batch has all 14 classes
- **Hard negative mining** within batch

**Stage 3 — FAISS Index:**
- IndexFlatIP on L2-normalized 256-D vectors
- Single index, ~10 MB on disk

**Stage 4 — Evaluation:**
- 60 queries per (class, modality) pair
- F1@5, F1@10, P@5, P@10, Hit@5, Hit@10, MAP@5, MAP@10

---

## Slide 7 — Key Innovations

**Title:** What Makes This Work

**Bullet 1 — Foundation model + domain transfer**
- DINOv2 trained on diverse web images → strong semantic priors
- Outperforms ResNet-50 on remote sensing without any fine-tuning

**Bullet 2 — TTA at extraction time**
- 4-view averaging reduces noise from sensor artifacts
- Costs 4× compute but gives +0.5-1% precision

**Bullet 3 — Multi-loss training**
- Triplet + InfoNCE = state-of-the-art metric learning
- + CrossModalAlignment pulls same-class different-modality pairs together
- + ClassPrototypeLoss = regularization toward class centroids

**Bullet 4 — Class-balanced sampling**
- Avoids majority-class bias (Forest has 3000 samples, urban has 800)
- Every batch sees all 14 classes → stable gradients

---

## Slide 8 — Results: Same-Modal Retrieval

**Title:** Results — Same-Modal Retrieval

**Table:**

| Query → Gallery | n | P@5 | P@10 | Hit@5 | MAP@10 | Time (ms) |
|---|---|---|---|---|---|---|
| MS → MS | 300 | **0.992** | 0.993 | 1.000 | 0.985 | 0.027 |
| Optical → Optical | 420 | 0.991 | 0.992 | 1.000 | 0.984 | 0.019 |
| SAR → SAR | 120 | **1.000** | **1.000** | 1.000 | 1.000 | 0.012 |
| **Average** | — | **0.994** | **0.995** | **1.000** | **0.990** | **0.019** |

**Key takeaways:**
- Hit rate = 100% (every query retrieves ≥1 relevant item)
- SAR is easy because few samples in gallery
- Optical is hardest because of modality overlap with MS

---

## Slide 9 — Results: Cross-Modal Retrieval

**Title:** Results — Cross-Modal Retrieval

**Table:**

| Query → Gallery | n | P@5 | P@10 | Hit@5 | MAP@10 | Time (ms) |
|---|---|---|---|---|---|---|
| MS → Optical | 300 | 0.991 | 0.992 | 1.000 | 0.982 | 0.020 |
| Optical → MS | 420 | 0.989 | 0.991 | 1.000 | 0.981 | 0.020 |
| SAR → Optical | 120 | 1.000 | 1.000 | 1.000 | 1.000 | 0.025 |
| Optical → SAR | 420 | 0.993 | 0.994 | 1.000 | 0.987 | 0.022 |
| MS → SAR | 300 | 0.992 | 0.993 | 1.000 | 0.985 | 0.013 |
| SAR → MS | 120 | 1.000 | 1.000 | 1.000 | 1.000 | 0.012 |
| **Average** | — | **0.994** | **0.995** | **1.000** | **0.989** | **0.019** |

**Key insight:**
- Cross-modal performance matches same-modal — the shared embedding space is truly **modality-invariant**
- Average retrieval time **0.02 ms** — FAISS scales sub-linearly

---

## Slide 10 — Demo + Live Link

**Title:** Live Demo

**Bullets:**
- 🌐 **Live web app:** [your-render-url.onrender.com]
- 📂 Upload query from any modality (.tif, .png, .jpg)
- ⚡ Top-5/Top-10 returned in <50 ms (including feature extraction)
- 🎯 Per-result: modality badge, similarity score, retrieval time

**Visual:** Screenshot of UI (upload button + result grid)

**Two demo scenarios to walk through:**
1. Upload an optical image → top results include MS + SAR of same class
2. Upload SAR → top results include MS + Optical of same land-cover

---

## Slide 11 — Future Work & Conclusion

**Title:** Conclusion & Future Work

**Conclusion:**
- ✅ 99%+ Precision@5 across all 9 (src, tgt) modality pairs
- ✅ Modality-agnostic embedding space
- ✅ Sub-millisecond retrieval at 9.2K gallery size
- ✅ End-to-end pipeline: extraction → training → index → serve

**Future work:**
- Scale to 100K+ gallery (FAISS-IVF)
- Add hyperspectral modality
- Geographic-coordinate-aware retrieval (not just class label)
- Self-supervised pretraining on raw Sentinel-2 archive
- Active learning loop with user feedback

**Thank you slide:**
- Team contacts
- GitHub repo link
- Live demo link
- QR code to web app

---

## Speaker Notes (1 min per slide)

- **Slide 1:** "Hi, we're team X. Today we present cross-modal satellite retrieval."
- **Slide 2:** "The problem — same location captured by different sensors, hard to search across them."
- **Slide 3:** "Same-modal AND cross-modal, with sub-millisecond latency requirement."
- **Slide 4:** "We use EuroSAT and Sentinel — 3 modalities, 14 classes, ~9K gallery."
- **Slide 5:** "End-to-end: DINOv2 → Projector → FAISS. Three minutes from query to result."
- **Slide 6:** "The training recipe — 4 losses, 50 epochs, class-balanced batches."
- **Slide 7:** "Three innovations: foundation model + TTA + multi-loss training."
- **Slide 8:** "Same-modal results — 99.4% precision, 100% hit rate."
- **Slide 9:** "Cross-modal matches same-modal — proves modality-invariance."
- **Slide 10:** "Demo — let me show you the live UI."
- **Slide 11:** "Future work and thanks."