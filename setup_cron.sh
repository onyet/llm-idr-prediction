#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$PROJECT_DIR/update_data.py"
LOG_PATH="$PROJECT_DIR/update_data.log"
PYTHON_EXEC="$PROJECT_DIR/.venv/bin/python"

# Ensure the script is executable
chmod +x "$SCRIPT_PATH"

# The cron commands to add (morning and evening)
CRON_CMD_MORNING="0 7 * * * $PYTHON_EXEC $SCRIPT_PATH >> $LOG_PATH 2>&1"
CRON_CMD_EVENING="0 17 * * * $PYTHON_EXEC $SCRIPT_PATH >> $LOG_PATH 2>&1"

# Ensure the morning job exists
CRONTAB_CONTENT=$(crontab -l 2>/dev/null || true)
if echo "$CRONTAB_CONTENT" | grep -Fq "$CRON_CMD_MORNING"; then
    echo "Morning cronjob for update_data.py already exists."
else
    echo "Adding morning cronjob for update_data.py to run daily at 07:00..."
    (crontab -l 2>/dev/null || true; echo "$CRON_CMD_MORNING") | crontab -
    echo "Morning cronjob added successfully."
fi

# Ensure the evening job exists
CRONTAB_CONTENT=$(crontab -l 2>/dev/null || true)
if echo "$CRONTAB_CONTENT" | grep -Fq "$CRON_CMD_EVENING"; then
    echo "Evening cronjob for update_data.py already exists."
else
    echo "Adding evening cronjob for update_data.py to run daily at 17:00..."
    (crontab -l 2>/dev/null || true; echo "$CRON_CMD_EVENING") | crontab -
    echo "Evening cronjob added successfully."
fi
