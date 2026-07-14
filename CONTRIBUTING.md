# Development and validation

Use Python 3.11 and install the PyTorch wheel matching the CPU/CUDA runtime.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m compileall -q .
python -m pytest -q
(cd liquidez_2026_checkpoint_22_3_2b && python -m pytest -q --confcutdir=.)
```
