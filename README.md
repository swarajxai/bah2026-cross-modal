---
title: BAH 2026 Cross-Modal Satellite Image Retrieval
emoji: 🛰️
colorFrom: blue
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
---

# 🛰️ BAH 2026 — Cross-Modal Satellite Image Retrieval

Demo: Upload a satellite image (optical/SAR/MS) and retrieve the most similar
images from a 9,200-image gallery across **3 modalities** (MS, Optical, SAR)
and **14 land-use classes**.

Powered by:
- ResNet-50 backbone (2048-D features)
- Modality-specific projection heads
- FAISS retrieval (~0.02 ms / query)
- Same-modal **P@5 = 98.6%**, Cross-modal **P@5 = 98.4%**

## API
- `GET /` — upload UI
- `GET /api/health` — service info
- `POST /api/retrieve` — image upload → top-K results
