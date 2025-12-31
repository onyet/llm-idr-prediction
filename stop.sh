#!/usr/bin/env bash
set -euo pipefail
# stop.sh — stop server started by run.sh
if [ -f uvicorn.pid ]; then
  PID=$(cat uvicorn.pid)
  kill "$PID" || true
  rm -f uvicorn.pid
  echo "Stopped pid $PID"
else
  echo "No uvicorn.pid found — server not running (or started differently)"
fi
