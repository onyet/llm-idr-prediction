#!/usr/bin/env bash
set -euo pipefail
# restart.sh — stop the server if running, then start it again

echo "=== Restarting FastAPI Server ==="
echo "Stopping server..."
./stop.sh

echo ""
echo "Starting server..."
./run.sh

echo ""
echo "=== Restart Complete ==="