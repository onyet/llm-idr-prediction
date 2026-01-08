#!/usr/bin/env bash
set -euo pipefail
# stop.sh — stop server started by run.sh

PORT=8000

# Function to check if port is in use
is_port_in_use() {
    lsof -i :$PORT >/dev/null 2>&1
}

# Function to kill process by port
kill_by_port() {
    local pids=$(lsof -ti :$PORT 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "Found processes using port $PORT: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
}

if [ -f uvicorn.pid ]; then
    PID=$(cat uvicorn.pid)
    if ps -p "$PID" >/dev/null 2>&1; then
        echo "Stopping server (pid $PID)..."
        kill "$PID" 2>/dev/null || true
        sleep 2
        if ps -p "$PID" >/dev/null 2>&1; then
            echo "Process still running, force killing..."
            kill -9 "$PID" 2>/dev/null || true
            sleep 1
        fi
    else
        echo "Process $PID not found, cleaning up pid file..."
    fi
    rm -f uvicorn.pid
else
    echo "No uvicorn.pid found"
fi

# Check if port is still in use and clean up
if is_port_in_use; then
    echo "Port $PORT still in use, cleaning up..."
    kill_by_port
fi

if is_port_in_use; then
    echo "Warning: Port $PORT still in use after cleanup"
else
    # Remove cronjobs for update_data and update_trending
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SCRIPT_PATH="$PROJECT_DIR/update_data.py"
    TREND_SCRIPT="$PROJECT_DIR/update_trending.py"

    # Remove any cron entries that reference either script
    if crontab -l 2>/dev/null | grep -Fq -e "$SCRIPT_PATH" -e "$TREND_SCRIPT"; then
        echo "Removing cronjobs for update_data.py and/or update_trending.py..."
        (crontab -l 2>/dev/null | grep -Fv -e "$SCRIPT_PATH" -e "$TREND_SCRIPT" || true) | crontab -
        echo "Cronjobs removed."
    else
        echo "No cronjobs to remove."
    fi

    echo "Server stopped successfully"
fi
