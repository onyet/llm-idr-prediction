#!/usr/bin/env bash
set -euo pipefail

# Skrip inisialisasi Ollama untuk VPS (siap produksi)
# Fitur utama:
# - Cek apakah Ollama sudah responsive di OLLAMA_BASE_URL (/api/tags)
# - Deploy via Docker dengan pull image terbaru dan volume persistensi
# - Opsi untuk membuat systemd unit (opsional, butuh sudo)
# - Opsi untuk memulai Ollama CLI jika binary tersedia
# - Logging dan healthchecks dengan retry/backoff

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"
LOG_DIR="$ROOT_DIR/logs"
OLLAMA_DATA_DIR="$ROOT_DIR/ollama_data"
DOCKER_IMAGE="ollama/ollama:latest"

# Load .env jika ada
if [ -f "$ENV_FILE" ]; then
  # ekspor baris non-komentar
  set -o allexport
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +o allexport
fi

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
API_CHECK_URL="$OLLAMA_BASE_URL/api/tags"

mkdir -p "$LOG_DIR"
mkdir -p "$OLLAMA_DATA_DIR"

LOG_FILE="$LOG_DIR/ollama-init.log"

echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') - Starting init_ollama.sh" | tee -a "$LOG_FILE"

# Fungsi cek apakah Ollama responsive
check_running() {
  if command -v curl >/dev/null 2>&1; then
    if curl -sSf --max-time 5 "$API_CHECK_URL" >/dev/null 2>&1; then
      return 0
    fi
  fi
  return 1
}

wait_for_ready() {
  local retries=${1:-30}
  local delay=${2:-2}
  local i=0
  while [ $i -lt $retries ]; do
    if check_running; then
      echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') - Ollama responsive" | tee -a "$LOG_FILE"
      return 0
    fi
    sleep $delay
    i=$((i+1))
  done
  return 1
}

if check_running; then
  echo "✅ Ollama sudah berjalan di $OLLAMA_BASE_URL. Tidak perlu inisialisasi." | tee -a "$LOG_FILE"
  exit 0
fi

echo "⚠️ Ollama tidak merespon di $OLLAMA_BASE_URL" | tee -a "$LOG_FILE"

# Show docker suggestion if docker available
if command -v docker >/dev/null 2>&1; then
  echo "Docker tersedia; Anda bisa menjalankan: docker run -d --restart unless-stopped --name ollama -p 11434:11434 -v $OLLAMA_DATA_DIR:/root/.ollama $DOCKER_IMAGE" | tee -a "$LOG_FILE"
else
  echo "Docker tidak tersedia di PATH. Jika ingin deploy otomatis via Docker, pasang Docker terlebih dahulu." | tee -a "$LOG_FILE" >&2
fi

# Deploy via Docker (pull, run, volume)
deploy_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker tidak terpasang. Tidak bisa melakukan deploy otomatis." | tee -a "$LOG_FILE" >&2
    return 1
  fi

  echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') - Menarik image $DOCKER_IMAGE" | tee -a "$LOG_FILE"
  docker pull "$DOCKER_IMAGE" | tee -a "$LOG_FILE"

  # If container exists, remove and recreate to ensure latest image, but try to be safe
  if docker ps -a --format '{{.Names}}' | grep -q '^ollama$'; then
    echo "Container 'ollama' sudah ada. Menghapus container lama untuk pembaruan..." | tee -a "$LOG_FILE"
    docker rm -f ollama | tee -a "$LOG_FILE" || true
  fi

  echo "Menjalankan container Ollama dengan volume: $OLLAMA_DATA_DIR" | tee -a "$LOG_FILE"
  docker run -d --restart unless-stopped --name ollama -p 11434:11434 -v "$OLLAMA_DATA_DIR":/root/.ollama "$DOCKER_IMAGE" | tee -a "$LOG_FILE"

  echo "Menunggu Ollama siap (max 60s)..." | tee -a "$LOG_FILE"
  if wait_for_ready 30 2; then
    echo "✅ Ollama siap di $OLLAMA_BASE_URL" | tee -a "$LOG_FILE"
    return 0
  else
    echo "❌ Ollama tidak responsif setelah deploy Docker" | tee -a "$LOG_FILE" >&2
    return 2
  fi
}

# Create a simple systemd unit to manage the docker container (optional)
create_systemd_unit() {
  local unit_file="/etc/systemd/system/ollama-docker.service"
  if [ ! -w /etc/systemd/system ] && [ "$EUID" -ne 0 ]; then
    echo "Butuh hak root untuk membuat systemd unit. Jalankan skrip ini dengan sudo --enable-systemd" | tee -a "$LOG_FILE" >&2
    return 1
  fi

  cat <<EOF | sudo tee "$unit_file" >/dev/null
[Unit]
Description=Ollama Docker container
After=docker.service
Requires=docker.service

[Service]
Restart=always
ExecStart=/usr/bin/docker start -a ollama
ExecStop=/usr/bin/docker stop -t 10 ollama

[Install]
WantedBy=multi-user.target
EOF

  sudo systemctl daemon-reload
  sudo systemctl enable --now ollama-docker.service
  echo "systemd unit created and started: $unit_file" | tee -a "$LOG_FILE"
}

# Start Ollama CLI if available
start_cli() {
  if command -v ollama >/dev/null 2>&1; then
    echo "Menjalankan Ollama daemon menggunakan binary 'ollama'..." | tee -a "$LOG_FILE"
    nohup ollama daemon > "$LOG_DIR/ollama.log" 2>&1 &
    if wait_for_ready 20 1; then
      echo "✅ Ollama daemon siap" | tee -a "$LOG_FILE"
      return 0
    else
      echo "⚠️ Ollama daemon dijalankan tetapi belum responsif" | tee -a "$LOG_FILE" >&2
      return 1
    fi
  else
    echo "Binary 'ollama' tidak ditemukan." | tee -a "$LOG_FILE" >&2
    return 2
  fi
}

# Parse flags: --deploy-docker, --start-cli, --enable-systemd
ENABLE_SYSTEMD=0
if [ "${1:-}" = "--deploy-docker" ]; then
  deploy_docker || exit 1
  if [ "${2:-}" = "--enable-systemd" ] || [ "${1:-}" = "--deploy-docker" ] && [ "${2:-}" = "--enable-systemd" ]; then
    create_systemd_unit || echo "Gagal membuat systemd unit" | tee -a "$LOG_FILE"
  fi
  exit 0
fi

if [ "${1:-}" = "--start-cli" ]; then
  start_cli || exit 1
  exit 0
fi

if [ "${1:-}" = "--deploy-docker--enable-systemd" ]; then
  deploy_docker || exit 1
  create_systemd_unit || exit 1
  exit 0
fi

# Jika tidak ada argumen, tampilkan bantuan singkat
cat <<EOF
Usage: $0 [--deploy-docker [--enable-systemd]] [--start-cli]

Contoh:
  $0 --deploy-docker
  $0 --deploy-docker --enable-systemd   # butuh sudo untuk membuat unit
  $0 --start-cli

Catatan:
  - Untuk opsi systemd, jalankan skrip dengan sudo
  - Script ini akan membuat folder $OLLAMA_DATA_DIR untuk persistensi data
  - Log file: $LOG_FILE
EOF

exit 1
