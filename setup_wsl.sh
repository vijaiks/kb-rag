#!/usr/bin/env bash
# One-time setup for kb-rag on WSL2 / Linux. Run from the repo root:  bash setup_wsl.sh
set -euo pipefail

echo "== 1. System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq git build-essential poppler-utils tesseract-ocr libgl1 >/dev/null

echo "== 2. uv"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "== 3. Python environment"
uv python install 3.12
uv sync

echo "== 4. Configuration"
if [ ! -f config.yaml ]; then
  cp config.example.yaml config.yaml
  echo "Created config.yaml from the template - EDIT IT: set source_dir to your document folder."
fi

echo "== 5. GPU check"
uv run python check_gpu.py || echo "(CPU mode also works - set embedding.device: cpu in config.yaml)"

cat <<'MSG'

Done. Next steps:
  1. Edit config.yaml (source_dir, data_dir)
  2. export ANTHROPIC_API_KEY="sk-ant-..."     # put it in ~/.bashrc
  3. uv run python -m ingest.pipeline --all
  4. uv run python -m query.ask "your first question"
MSG
