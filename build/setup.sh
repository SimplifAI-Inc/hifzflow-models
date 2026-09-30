#!/usr/bin/env bash
# Set up NVIDIA NeMo in WSL to rebuild the FastConformer model (test tooling only).
set -e
mkdir -p ~/fcq && cd ~/fcq
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"
[ -d .venv ] || uv venv -q --python 3.11 .venv
. .venv/bin/activate
uv pip install -q "nemo_toolkit[asr]==2.4.0" onnx onnxruntime soundfile 2>&1 | tail -5
python - <<'EOF'
import nemo, torch
print("nemo", nemo.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available())
EOF
