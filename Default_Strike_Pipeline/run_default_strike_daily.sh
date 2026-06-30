#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${OPTIONS_DATASET_ROOT:-/srv/data/options_model_features}"
DEFAULT_STRIKE_ROOT="${OPTIONS_DEFAULT_STRIKE_ROOT:-$DATA_ROOT/Default_Strike}"
LOG_DIR="${OPTIONS_CRON_LOG_DIR:-$REPO_ROOT/logs}"
LOCK_FILE="${DEFAULT_STRIKE_LOCK_FILE:-$DATA_ROOT/default_strike_pipeline.lock}"

export OPTIONS_UPDATER_ROOT="$REPO_ROOT"
export OPTIONS_DATASET_ROOT="$DATA_ROOT"
export OPTIONS_DEFAULT_STRIKE_ROOT="$DEFAULT_STRIKE_ROOT"
export OPTIONS_CONSTITUENCY_ROOT="${OPTIONS_CONSTITUENCY_ROOT:-$DATA_ROOT/Constituency}"
export OPTIONS_ATM_NORMALIZED_ROOT="${OPTIONS_ATM_NORMALIZED_ROOT:-$DATA_ROOT/ATM_Normalized_Options}"
export OPTIONS_IV_ROOT="${OPTIONS_IV_ROOT:-$OPTIONS_ATM_NORMALIZED_ROOT/contracts}"
export OPTIONS_CLEAN_STOCK_ROOT="${OPTIONS_CLEAN_STOCK_ROOT:-/srv/data/stocks}"
export DEFAULT_STRIKE_WORKERS="${DEFAULT_STRIKE_WORKERS:-4}"

mkdir -p "$LOG_DIR" "$DEFAULT_STRIKE_ROOT/contracts" "$DEFAULT_STRIKE_ROOT/Features"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON="$PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/.venv/bin/python"
else
  echo "Missing project virtualenv: $REPO_ROOT/.venv" >&2
  exit 127
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="$LOG_DIR/default_strike_$timestamp.log"
latest_log="$LOG_DIR/default_strike_latest.log"
ln -sfn "$log_file" "$latest_log"

{
  echo "Started Default Strike update at $(date -u --iso-8601=seconds)"
  echo "Source IV: $OPTIONS_IV_ROOT"
  echo "Output: $DEFAULT_STRIKE_ROOT"
  echo "Constituency: $OPTIONS_CONSTITUENCY_ROOT"
  echo "Workers: $DEFAULT_STRIKE_WORKERS"
  cd "$REPO_ROOT"
  flock -n "$LOCK_FILE" "$PYTHON" Default_Strike_Pipeline/daily_update.py
  echo "Finished Default Strike update at $(date -u --iso-8601=seconds)"
} >>"$log_file" 2>&1
