# Cross-Modal Satellite Image Retrieval — Pipeline Cheatsheet

**Quick one-liners** (run from `D:\BAH2026\cross_modal_retrieval` folder)

| Step | Command |
|---|---|
| Extract features | `D:\BAH2026\.venv\Scripts\python.exe -m scripts.extract_features` |
| Train projectors | `D:\BAH2026\.venv\Scripts\python.exe -m scripts.train_projectors` |
| Build FAISS index | `D:\BAH2026\.venv\Scripts\python.exe -m scripts.build_index` |
| Evaluate | `D:\BAH2026\.venv\Scripts\python.exe -m scripts.evaluate` |
| Run everything | `D:\BAH2026\.venv\Scripts\python.exe -m scripts.run_all` |
| Launch web app | `D:\BAH2026\.venv\Scripts\python.exe webapp\app.py` |

Or just double-click `scripts\run.bat` and pick an option.

**URL after launching web:** http://127.0.0.1:5000
