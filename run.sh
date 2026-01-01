#!/usr/bin/env bash
set -euo pipefail
# run.sh — start the uvicorn server in background and save pid to uvicorn.pid

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

# Clean up any existing server
if [ -f uvicorn.pid ]; then
    OLD_PID=$(cat uvicorn.pid)
    if ps -p "$OLD_PID" >/dev/null 2>&1; then
        echo "Stopping existing server (pid $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
        if ps -p "$OLD_PID" >/dev/null 2>&1; then
            kill -9 "$OLD_PID" 2>/dev/null || true
            sleep 1
        fi
    fi
    rm -f uvicorn.pid
fi

# Check if port is still in use and clean up
if is_port_in_use; then
    echo "Port $PORT already in use, cleaning up..."
    kill_by_port
fi

# Double check port is free
if is_port_in_use; then
    echo "Error: Port $PORT still in use. Please check manually."
    exit 1
fi

# Activate virtual environment
. .venv/bin/activate

# Setup cronjob for data updates
./setup_cron.sh

# Start server
echo "Starting server on port $PORT..."
nohup uvicorn app.main:app --host 0.0.0.0 --port $PORT > uvicorn.log 2>&1 &
NEW_PID=$!
echo $NEW_PID > uvicorn.pid

# Wait a moment and check if process is still running
sleep 2
if ps -p $NEW_PID >/dev/null 2>&1; then
    echo "Server started (pid $NEW_PID). Logs: uvicorn.log"
else
    echo "Error: Server failed to start. Check uvicorn.log for details."
    rm -f uvicorn.pid
    exit 1
fi
