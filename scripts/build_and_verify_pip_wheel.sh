#!/usr/bin/env bash
set -euo pipefail

rm -rf /tmp/clean-env
python -m build
WHEEL=$(ls -t dist/*.whl | head -n1)

python -m venv /tmp/clean-env
source /tmp/clean-env/bin/activate
pip install "$WHEEL"
python scripts/verify_pip_installation.py
deactivate

echo "OK: clean install verified: $WHEEL"