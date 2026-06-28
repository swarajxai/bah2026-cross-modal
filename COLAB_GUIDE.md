# 🚀 BAH 2026 — Colab DINOv2 Pipeline (Option B — Top-2 Push)

**TL;DR** — ek script hai jo Colab pe chala ke Top-2-grade DINOv2 model train karta hai. Tum bas paste karo, run karo, download karo. Bas.

---

## Step 1: Google Colab kholna

1. Browser me jao: **https://colab.research.google.com**
2. Google account se sign-in karo
3. **File → New Notebook**
4. **Runtime → Change runtime type → T4 GPU** select karo, Save karo

## Step 2: Code paste karna

1. Notebook me ek single cell banao (already ek default cell hota hai)
2. **Poora code paste karo** from this file: `colab/colab_pipeline.py`
   - File Explorer me `colab_pipeline.py` kholo
   - Ctrl+A → Ctrl+C → cell me Ctrl+V
3. Cell ke upar **`INSTALL = True`** karo (first line me hai)
4. **Shift+Enter** dabao

## Step 3: Data upload karo (Colab se)

Jab cell chalega, ~30 second baad ye print hoga:
```
[setup] data not found at /content/BAH2026
[setup] uploading data zip from local computer...
```

Tab ek **file picker** khulega. Tumhe ek zip banana hoga:

### Apne local computer pe:
1. `D:\BAH2026\` folder me jao
2. **EuroSAT** aur **Sentinel** folders ka zip banao:
   - Right-click → "Send to" → "Compressed (zipped) folder"
   - Naam do: `BAH2026_data.zip`
   - Ya PowerShell me: `Compress-Archive -Path D:\BAH2026\EuroSAT, D:\BAH2026\Sentinel -DestinationPath D:\BAH2026_data.zip`
3. **Zip ke andar `EuroSAT/` aur `Sentinel/` folders hone chahiye** (top-level pe)

### Zip ke andar kya hona chahiye:
```
BAH2026_data.zip
├── EuroSAT/
│   ├── AnnualCrop/
│   │   ├── s1/  (3000 .tif files)
│   │   └── s2/  (3000 .jpg files)
│   ├── Forest/
│   └── ... (10 classes total)
└── Sentinel/
    ├── agri/
    │   ├── s1/  (SAR)
    │   └── s2/  (optical)
    ├── barrenland/
    └── ... (4 classes total)
```

## Step 4: Wait for training (~45 min total)

Cell 6 features extract karega — **5 min** (DINOv2 GPU pe fast hai)
Cell 9 v6 projector train karega — **2-3 min** (T4 GPU pe)
Cell 10 FAISS index banayega — **5 sec**
Cell 11 evaluation — **30 sec**

Total: **~10-15 min** (baaki time download ka)

## Step 5: Download v6 artifacts

Cell 12 automatic download kar dega:
- `cross_modal_v6.zip` (~30 MB)

Ye tumhare local computer pe `Downloads/` folder me aa jayega.

## Step 6: Local pe activate karo

Apne local project folder me jao aur ye command run karo:

```powershell
cd D:\BAH2026\cross_modal_retrieval
python scripts\colab_transfer.py --zip D:\Downloads\cross_modal_v6.zip
```

Ye automatically:
- Current V5 weights ko `outputs/backup_v5/` me backup karega
- V6 weights ko `outputs/projector.pt`, `gallery.faiss`, `gallery_meta.npz` me copy karega
- Naya `features_dinov2.npz` bhi rakh dega

## Step 7: Server restart + verify

```powershell
# Purana Flask server band karo (agar chal raha hai)
# Task manager me python.exe dhundho aur end karo

# Naya server start karo
cd D:\BAH2026\cross_modal_retrieval\webapp
D:\BAH2026\.venv\Scripts\python.exe app.py
```

Browser me `http://127.0.0.1:5000` kholke test karo:
- Forest image upload karo
- Top-5 me Forest ke images aane chahiye (same + cross modality)

## ⚠️ Agar kuch galat ho jaye

| Problem | Solution |
|---|---|
| "No module named X" | Cell 1 me INSTALL=True set karo, re-run |
| GPU nahi mila | Runtime → Change runtime type → T4 GPU (free) |
| Data zip bohot bada hai | EuroSAT me 30,000 images hain, ~2 GB. Colab free me slow hoga but chalega. Alternative: sirf pehle 1000 files per class upload karo for testing. |
| Training bahut slow | Cell 9 me epochs=30 → epochs=15 kar do (half time) |
| Zip download nahi hua | Files panel (📁 left side) me jao, `cross_modal_v6.zip` pe right-click → Download |

## 📊 Expected metrics after V6

| Metric | V5 (current) | V6 (target) |
|---|---|---|
| Same-modal P@5 | 0.986 | **0.992-0.995** |
| Cross-modal P@5 | 0.984 | **0.990-0.995** |
| Same-modal H@5 | 1.000 | **1.000** |
| Cross-modal H@5 | 1.000 | **1.000** |
| Latency | 0.05 ms | 0.05 ms (same, FAISS index only) |

## 🎯 Kab tak karna hai?

Best case: **agle 2 ghante me v6 ready**. Colab pe 15 min + tumhara 5 min (paste/download).

Koi bhi step pe atak jaoge to mujhe bata dena — main turant fix kar dunga. Main monitoring kar raha hun. 👍