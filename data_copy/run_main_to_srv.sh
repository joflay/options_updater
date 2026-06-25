#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${OPTIONS_DATASET_ROOT:-/srv/data/options_model_features}"
LOG_DIR="${OPTIONS_CRON_LOG_DIR:-$REPO_ROOT/logs}"
LOCK_FILE="${OPTIONS_CRON_LOCK_FILE:-$DATA_ROOT/main_pipeline.lock}"

export OPTIONS_DATASET_ROOT="$DATA_ROOT"
export OPTIONS_CONSTITUENCY_ROOT="${OPTIONS_CONSTITUENCY_ROOT:-$DATA_ROOT/Constituency}"
export OPTIONS_FLATFILES_ROOT="${OPTIONS_FLATFILES_ROOT:-$DATA_ROOT/FlatFiles/us_options_opra/day_aggs_v1}"
export OPTIONS_IV_ROOT="${OPTIONS_IV_ROOT:-$DATA_ROOT/options_data/iv/us_options_opra/day_aggs_v1}"
export OPTIONS_DAILY_FEATURES_ROOT="${OPTIONS_DAILY_FEATURES_ROOT:-$DATA_ROOT/options_data/features/day_aggs_v1}"
export OPTIONS_CLOSE_CACHE="${OPTIONS_CLOSE_CACHE:-$DATA_ROOT/options_data/features/underlying_close_cache.csv}"
export OPTIONS_FINAL_FEATURES_DIR="${OPTIONS_FINAL_FEATURES_DIR:-$DATA_ROOT/final_features}"
export OPTIONS_CLEAN_STOCK_ROOT="${OPTIONS_CLEAN_STOCK_ROOT:-$DATA_ROOT/clean stocks}"
export OPTIONS_LSEG_DEBUG_ROOT="${OPTIONS_LSEG_DEBUG_ROOT:-$DATA_ROOT/lseg_constituency_debug}"

mkdir -p \
  "$LOG_DIR" \
  "$OPTIONS_CONSTITUENCY_ROOT" \
  "$OPTIONS_FLATFILES_ROOT" \
  "$OPTIONS_IV_ROOT" \
  "$OPTIONS_DAILY_FEATURES_ROOT" \
  "$(dirname -- "$OPTIONS_CLOSE_CACHE")" \
  "$OPTIONS_FINAL_FEATURES_DIR" \
  "$OPTIONS_CLEAN_STOCK_ROOT" \
  "$OPTIONS_LSEG_DEBUG_ROOT"

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="$LOG_DIR/main_pipeline_$timestamp.log"
latest_log="$LOG_DIR/main_pipeline_latest.log"
ln -sfn "$log_file" "$latest_log"

{
  echo "Started Options_Model main pipeline at $(date -u --iso-8601=seconds)"
  echo "Repo root: $REPO_ROOT"
  echo "Dataset root: $DATA_ROOT"
  echo "Flatfiles: $OPTIONS_FLATFILES_ROOT"
  echo "IV output: $OPTIONS_IV_ROOT"
  echo "Daily features: $OPTIONS_DAILY_FEATURES_ROOT"
  echo "Final features: $OPTIONS_FINAL_FEATURES_DIR"
  echo "Clean stocks: $OPTIONS_CLEAN_STOCK_ROOT"
  echo
  cd "$REPO_ROOT"
  flock -n "$LOCK_FILE" "$PYTHON" Data_Pipeline/main.py
  echo
  echo "Finished Options_Model main pipeline at $(date -u --iso-8601=seconds)"
} >>"$log_file" 2>&1
