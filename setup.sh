#!/usr/bin/env bash
set -e

python3 -m venv .venv
# shellcheck source=/dev/null
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Environment ready. To activate: 'source .venv/bin/activate'"
echo "Start the server: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

echo "Done. To run tests: pytest"