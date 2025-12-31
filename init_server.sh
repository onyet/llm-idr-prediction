#!/usr/bin/env bash
set -euo pipefail

# init_server.sh
# Usage: sudo ./init_server.sh
# Installs system deps, creates venv, installs python deps, runs tests, and creates easy run scripts.

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
SUDO=""
if [ "$EUID" -ne 0 ]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO=sudo
    echo "Using sudo to install system packages"
  else
    echo "This script must be run as root or you need sudo installed." >&2
    exit 1
  fi
fi

echo "Repository directory: $REPO_DIR"

# Update and install system packages
echo "Updating apt and installing system packages..."
$SUDO apt-get update -y
$SUDO apt-get install -y python3 python3-venv python3-dev python3-pip build-essential gcc gfortran git curl \
  libatlas-base-dev libopenblas-dev liblapack-dev pkg-config

# Create virtual environment
if [ ! -d "$REPO_DIR/.venv" ]; then
  echo "Creating virtual environment at $REPO_DIR/.venv"
  python3 -m venv "$REPO_DIR/.venv"
else
  echo "Virtualenv already exists at .venv, skipping creation"
fi

# Activate and install Python deps
echo "Activating venv and installing Python dependencies..."
# shellcheck source=/dev/null
. "$REPO_DIR"/.venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r "$REPO_DIR/requirements.txt"

# Run tests
echo "Running test suite..."
export PYTHONPATH="$REPO_DIR"
if pytest -q; then
  echo "All tests passed"
else
  echo "Tests failed — please check the output above" >&2
  exit 2
fi

# Create easy run/stop scripts
RUN_SH="$REPO_DIR/run.sh"
STOP_SH="$REPO_DIR/stop.sh"

cat > "$RUN_SH" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
# run.sh — start the uvicorn server in background and save pid to uvicorn.pid
. .venv/bin/activate
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &
echo $! > uvicorn.pid
echo "Server started (pid $(cat uvicorn.pid)). Logs: uvicorn.log"
EOF

cat > "$STOP_SH" <<'EOF'
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
EOF

chmod +x "$RUN_SH" "$STOP_SH"

# Create a sample systemd unit file (user should edit paths & user)
SYSTEMD_DIR="$REPO_DIR/systemd"
mkdir -p "$SYSTEMD_DIR"
cat > "$SYSTEMD_DIR/llm-exchange.service" <<'EOF'
[Unit]
Description=LLM-EXHANGE FastAPI app (uvicorn)
After=network.target

[Service]
# Adjust User and Group to match deployment environment
User=www-data
Group=www-data
WorkingDirectory=/path/to/repo  # <- change this to the repository path
ExecStart=/path/to/repo/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

chmod -R 644 "$SYSTEMD_DIR"

# Basic health check
echo "Starting temporary dev server to run a health check (background)..."
# start server in background for quick check
./run.sh
sleep 2
if command -v curl >/dev/null 2>&1; then
  if curl -sSf http://127.0.0.1:8000/healthz >/dev/null 2>&1; then
    echo "Health check OK"
  else
    echo "Health check failed — see uvicorn.log" >&2
    ./stop.sh || true
    exit 3
  fi
else
  echo "curl not available to do health check; please verify server manually"
fi

# Stop the temporary server started for health check
./stop.sh || true

cat <<'USAGE'

Initialization complete ✅

Usage:
 - To start server manually: ./run.sh
 - To stop server: ./stop.sh
 - To install systemd unit (optional): copy systemd/llm-exchange.service to /etc/systemd/system/, edit WorkingDirectory, ExecStart, User, Group, then: sudo systemctl daemon-reload && sudo systemctl enable --now llm-exchange

Notes:
 - Run this script as root (sudo) on a fresh Ubuntu server.
 - If installation of some Python packages fails (prophet can take time to build), check the uvicorn.log for details and consider installing additional OS libraries.

USAGE

echo "Init script finished successfully"
