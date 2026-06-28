# 🚀 BAH 2026 — DINOv2 V6 Training Guide (Improved Edition)

**TL;DR**: Apne local computer pe 1 command chalao (data zip banane ke liye), fir Colab me 1 file paste karo, ~25 min wait, 1 zip download karo, locally activate karo.

---

## Step 1: Data zip banao (apne local computer pe)

PowerShell kholke ye command chalao:

```powershell
D:\BAH2026\.venv\Scripts\python.exe D:\BAH2026\cross_modal_retrieval\scripts\build_colab_datazip.py
```

**Expected output**:
```
[zip] will package ~30100 files into D:\BAH2026_data.zip
  ... 1000/30100  (3s)
  ...
[zip] done: 30100 files, 1850 MB, 180s
```

Zip ban jayega at `D:\BAH2026_data.zip` (~1.5-2 GB).

**Optional — testing ke liye sirf 50 files/class chahiye:**
```powershell
D:\BAH2026\.venv\Scripts\python.exe D:\BAH2026\cross_modal_retrieval\scripts\build_colab_datazip.py --max-per-class 50
```
(Yeh sirf test ke liye, production run me mat karna — accuracy girega.)

---

## Step 2: Colab notebook kholo

1. Browser: https://colab.research.google.com
2. Sign in (Google account)
3. **File → New Notebook**
4. **Runtime → Change runtime type → T4 GPU → Save**

---

## Step 3: Script paste karo

1. Naya cell banao (ya default cell me paste karo)
2. **Poori file kholo**: `D:\BAH2026\cross_modal_retrieval\colab\colab_pipeline_v6.py`
3. **Ctrl+A → Ctrl+C → Colab cell me Ctrl+V**
4. Pehli line me check karo: `INSTALL = False` (agar first run hai toh True kar do)
5. **Shift+Enter**

---

## Step 4: Data upload karo

~30 sec baad Colab print karega:
```
[setup] data not found at /content/BAH2026
[setup] upload BAH2026_data.zip ...
```

**File picker khulega** → `D:\BAH2026_data.zip` select karo → Upload.

Upload me **5-10 min lagenge** (1.5-2 GB).

**Tip**: Tab tak dusra kaam karo — upload background me hoti hai. Zip upload hone ke baad script auto-continue karegi.

---

## Step 5: Wait for training (~25-30 min total)

Ye phases dikhenge:

```
[extract]  500/30100  elapsed 25s
[extract] 3000/30100  elapsed 150s       (~5 min: feature extraction with 4-view TTA)
[extract] done  (300s)
[split] train=27090 val=3010
v6 epoch 01/50  train=4.2134  val=4.1012  lr=2.00e-03  (50s)        (~20 min: 50 epochs)
v6 epoch 02/50  train=3.8521  val=3.7902  ...
...
v6 epoch 50/50  train=2.4521  val=2.4612  lr=4.50e-04
[train-v6] saved best@47  val=2.4312

[build] ntotal=30100
[eval] saved -> /content/v6_eval.json
[done] v6 artifacts ready at /content/cross_modal_v6.zip
```

### Console log ko monitor karo — ye check karo:

| Log line | Matlab |
|---|---|
| `v6 epoch 01/50 train=4.21 ...` | Training shuru, normal hai |
| `train=` value ghat-ta ja raha hai | Learning ho rahi hai ✅ |
| `val=` bhi ghat raha hai | Generalize ho raha hai ✅ |
| `train=` kam ho raha but `val=` badh raha | **Overfitting** — early stop ho jayega, theek hai |
| `cuda=OOM error` | GPU out of memory — Step 6 me troubleshooting dekho |

### Ye numbers dikhne chahiye eval me:

```
=== V6 EVAL (60 queries/class, K=10) ===
  same   ms->ms       P@5=0.99XX  H@5=1.0000  F1@5=0.0XXX
  cross  ms->optical  P@5=0.99XX  H@5=1.0000  F1@5=0.0XXX
  ...
=== SUMMARY ===
  Same-modal   P@5=0.992-0.995  H@5=1.000
  Cross-modal  P@5=0.990-0.995  H@5=1.000
```

Agar P@5 < 0.97 dikhe toh training mein kuch gadbad hai — Step 6 dekho.

---

## Step 6: Download v6 artifacts

Script auto-download kar dega **`cross_modal_v6.zip`** (~100 MB) tumhare `Downloads/` folder me.

Manual download karna ho toh Colab ke left sidebar me 📁 icon click karo → `cross_modal_v6.zip` pe right-click → Download.

**Zip me ye 5 files honi chahiye**:
- `v6_projector.pt` — trained projection heads (~85 MB)
- `v6_gallery.faiss` — FAISS index (~10 MB)
- `v6_gallery_meta.npz` — gallery metadata (~9 MB)
- `v6_eval.json` — final eval metrics
- `features_dinov2.npz` — cached DINOv2 features (reuse ke liye)

---

## Step 7: Local pe activate karo

Apne local PowerShell me:

```powershell
# 1. Move zip to project folder
Move-Item D:\Downloads\cross_modal_v6.zip D:\BAH2026\cross_modal_retrieval\outputs\

# 2. Unzip + activate
cd D:\BAH2026\cross_modal_retrieval\outputs
Expand-Archive -Path cross_modal_v6.zip -DestinationPath . -Force

# 3. Backup current v5
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Path "backup_v6_$ts" | Out-Null
Copy-Item projector.pt, gallery.faiss, gallery_meta.npz -Destination "backup_v6_$ts\"

# 4. Activate v6 (rename v6_* to standard names)
Rename-Item v6_projector.pt      projector.pt       -Force
Rename-Item v6_gallery.faiss     gallery.faiss      -Force
Rename-Item v6_gallery_meta.npz  gallery_meta.npz   -Force

# 5. Clean up
Remove-Item cross_modal_v6.zip
```

---

## Step 8: Restart server + verify

```powershell
# Kill old python (Flask server)
Get-Process python* | Where-Object { $_.MainWindowTitle -eq "" -or $_.Path -like "*\.venv\*" } | Stop-Process -Force

# Start new server
cd D:\BAH2026\cross_modal_retrieval\webapp
D:\BAH2026\.venv\Scripts\python.exe app.py
```

Browser me `http://127.0.0.1:5000` kholo. **Forest image upload karo** — top-5 me Forest ke images aane chahiye (same + cross modality).

Quick sanity check:
- Query: `EuroSAT/Forest/s2/Forest_5.jpg` (optical) → top-5 me `Forest/s1/*.tif` AND `Forest/s2/*.jpg` AND `agri/s2/*.png` (Sentinel class ≈ Forest in semantics) aane chahiye.
- Retrieval time har result ke saath dikhega (< 1 ms hona chahiye).

---

## ⚠️ Troubleshooting

| Problem | Solution |
|---|---|
| `INSTALL` line ko True karna bhool gaye | First line edit karo: `INSTALL = True`, re-run cell |
| `CUDA OOM` error | Step 9 me — `BATCH_SIZE = 192` kar do, re-run |
| Data upload slow | Colab pe `Files → Upload` se direct upload karo (script ke pause hone ka wait mat karo) |
| Training bahut slow (1+ hr) | `EPOCHS = 30` kar do (50 ki jagah) — sirf 60% time lagega |
| `p@5 = 0.90 (low)` after eval | `LR = 1e-3` kar do (default 2e-3 se kam). Re-run from extraction. |
| Zip bahut bada (2GB+) | `--max-per-class 500` se sirf 500/class upload karo for first run, baad me full karoge |
| `BadZipFile` on upload | Zip corrupt hua — re-run `build_colab_datazip.py` |
| Server pe v6 load nahi ho raha | Check `outputs/projector.pt` size — 80 MB+ hona chahiye. Agar 30 MB hai toh v5 hi copy hua, v6 nahi |

---

## 📊 Expected vs current (v5) metrics

| Metric | V5 (current) | V6 target |
|---|---|---|
| Same-modal P@5 | 0.986 | **0.992-0.995** |
| Cross-modal P@5 | 0.984 | **0.990-0.995** |
| Same-modal F1@5 | 0.0153 | **0.0153-0.0155** (similar — F1 capped by dataset) |
| Cross-modal F1@5 | 0.0146 | **0.0153-0.0155** |
| Latency | 0.02 ms | 0.02 ms (same — FAISS unchanged) |

**Note on F1**: F1@5/F1@10 structurally low rehte hain kyunki har query ke against ~600 relevant images hain (5/600 = max 0.8% recall). Hackathon ka criterion F1 hai but judges Precision/HitRate/MAP bhi dekhte hain — woh 99%+ honge.

---

## 🕐 Time budget

| Phase | Time |
|---|---|
| Zip build (local) | 3-5 min |
| Colab upload (1.5-2 GB) | 5-10 min |
| DINOv2 feature extraction (TTA x4) | 5 min |
| Projector training (50 epochs) | 15-20 min |
| Gallery build + eval | 1 min |
| Download v6 zip | 2-3 min |
| **Total** | **~35-45 min** |

Colab free T4 max 90 min session deta hai — comfortably fit ho jayega.

---

## 🎯 Kab tak complete karna hai?

Best case: **agle 1-1.5 ghante me v6 ready**.

Agar kisi bhi step pe atak jaoge toh mujhe batao — main turant fix kar dunga. 👍