#!/usr/bin/env bash
set -euo pipefail
# restart.sh — stop the server if running, then start it again
echo "Stopping server..."
./stop.sh || true  # ignore errors if not running
echo "Starting server..."
./run.sh