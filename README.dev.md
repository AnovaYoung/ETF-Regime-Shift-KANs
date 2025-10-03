# Developer Guide (README.dev)

This repo uses a single virtual environment at the project root (`.venv`) and a simple `src/` layout.

## Environment
- Python: 3.11
- One venv only: `.venv` at repo root.
- PyTorch: install GPU build for CUDA 12.4 when possible (see main README).

### Select interpreter in VS Code
1. `Ctrl+Shift+P` → **Python: Select Interpreter**
2. Pick `.venv` (e.g., `.venv\Scripts\python.exe` on Windows).

### Check Torch is visible
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Dependency management
- Add new deps to `requirements.txt`.
- After changing deps, run: `pip freeze > requirements.lock` (optional) to capture a reproducible snapshot.
- For minimal top-level packages, you can use `pip install pip-chill && pip-chill > requirements.txt`.

## Code style / QA
- **ruff** and **black** configured via `pyproject.toml`.
- Run format/lint:
```bash
ruff check src tests
black src tests
```

## Testing
```bash
pytest -q
```

## Data policy
- Everything under `data/` is **ignored by git** by default.
- Put raw downloads in `data/raw/`, intermediate artifacts in `data/interim/`, model-ready in `data/processed/`.
- If you need to version large files, prefer **Git LFS** or an object store.

## Typical workflow
1. Activate venv.
2. Confirm Torch (GPU if available).
3. Put your raw ETF fund-flow folders under `data/raw/fundflow/<TICKER>/...`
4. Point configs to those paths.
5. Run `python -m etf_kan.training.train --config configs/default.yaml`.

## Project hygiene
- No virtualenvs inside subfolders. Only `.venv` at the root (and it is `.gitignore`’d).
- Avoid committing notebooks with large outputs. Clear outputs before commit.
- Keep experiments in `notebooks/` and push only those worth sharing.
