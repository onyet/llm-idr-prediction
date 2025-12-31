#!/usr/bin/env bash
set -euo pipefail
# run.sh — start the uvicorn server in background and save pid to uvicorn.pid
. .venv/bin/activate
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &
echo $! > uvicorn.pid
echo "Server started (pid $(cat uvicorn.pid)). Logs: uvicorn.log"
