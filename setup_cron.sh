#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$PROJECT_DIR/update_data.py"
LOG_PATH="$PROJECT_DIR/update_data.log"
PYTHON_EXEC="$PROJECT_DIR/.venv/bin/python"

# Ensure the script is executable
chmod +x "$SCRIPT_PATH"

# The cron command to add
CRON_CMD="0 7 * * * $PYTHON_EXEC $SCRIPT_PATH >> $LOG_PATH 2>&1"

# Check if the job already exists in crontab
if crontab -l 2>/dev/null | grep -Fq "$SCRIPT_PATH"; then
    echo "Cronjob for update_data.py already exists. Skipping."
else
    echo "Adding cronjob for update_data.py to run daily at 07:00..."
    (crontab -l 2>/dev/null || true; echo "$CRON_CMD") | crontab -
    echo "Cronjob added successfully."
fi
