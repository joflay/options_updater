#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${OPTIONS_DATASET_ROOT:-/srv/data/options_model_features}"
LOG_DIR="${OPTIONS_CRON_LOG_DIR:-$REPO_ROOT/logs}"
LOCK_FILE="${OPTIONS_CRON_LOCK_FILE:-$DATA_ROOT/main_pipeline.lock}"

export OPTIONS_DATASET_ROOT="$DATA_ROOT"
export OPTIONS_ATM_NORMALIZED_ROOT="${OPTIONS_ATM_NORMALIZED_ROOT:-$DATA_ROOT/ATM_Normalized_Options}"
export OPTIONS_CONSTITUENCY_ROOT="${OPTIONS_CONSTITUENCY_ROOT:-$DATA_ROOT/Constituency}"
export OPTIONS_FLATFILES_ROOT="${OPTIONS_FLATFILES_ROOT:-$DATA_ROOT/FlatFiles/us_options_opra/day_aggs_v1}"
export OPTIONS_IV_ROOT="${OPTIONS_IV_ROOT:-$OPTIONS_ATM_NORMALIZED_ROOT/contracts}"
export OPTIONS_DAILY_FEATURES_ROOT="${OPTIONS_DAILY_FEATURES_ROOT:-$OPTIONS_ATM_NORMALIZED_ROOT/features/day_aggs_v1}"
export OPTIONS_CLOSE_CACHE="${OPTIONS_CLOSE_CACHE:-$DATA_ROOT/options_data/features/underlying_close_cache.csv}"
export OPTIONS_FINAL_FEATURES_DIR="${OPTIONS_FINAL_FEATURES_DIR:-$DATA_ROOT/final_features}"
export OPTIONS_CLEAN_STOCK_ROOT="${OPTIONS_CLEAN_STOCK_ROOT:-$DATA_ROOT/clean stocks}"
export OPTIONS_LSEG_DEBUG_ROOT="${OPTIONS_LSEG_DEBUG_ROOT:-$DATA_ROOT/lseg_constituency_debug}"
export OPTIONS_RISK_FREE_RATE_CSV="${OPTIONS_RISK_FREE_RATE_CSV:-/srv/data/risk_free_rate/DGS3MO_risk_free_rate.csv}"
export OPTIONS_WORKERS="${OPTIONS_WORKERS:-4}"

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

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON="$PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/.venv/bin/python"
else
  cat >&2 <<EOF
Missing project virtualenv: $REPO_ROOT/.venv

Create it before running the pipeline:
  cd "$REPO_ROOT"
  python3 -m venv .venv
  .venv/bin/python -m pip install -r Data_Pipeline/requirements.txt
EOF
  exit 127
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="$LOG_DIR/main_pipeline_$timestamp.log"
latest_log="$LOG_DIR/main_pipeline_latest.log"
ln -sfn "$log_file" "$latest_log"

{
  echo "Started Options_Model main pipeline at $(date -u --iso-8601=seconds)"
  echo "Repo root: $REPO_ROOT"
  echo "Dataset root: $DATA_ROOT"
  echo "ATM normalized root: $OPTIONS_ATM_NORMALIZED_ROOT"
  echo "Flatfiles: $OPTIONS_FLATFILES_ROOT"
  echo "IV output: $OPTIONS_IV_ROOT"
  echo "Daily features: $OPTIONS_DAILY_FEATURES_ROOT"
  echo "Final features: $OPTIONS_FINAL_FEATURES_DIR"
  echo "Clean stocks: $OPTIONS_CLEAN_STOCK_ROOT"
  echo "Risk-free rates: $OPTIONS_RISK_FREE_RATE_CSV"
  echo "Option workers: $OPTIONS_WORKERS"
  echo "Python: $PYTHON"
  "$PYTHON" --version
  echo
  cd "$REPO_ROOT"
  flock -n "$LOCK_FILE" "$PYTHON" Data_Pipeline/main.py
  echo
  echo "Finished Options_Model main pipeline at $(date -u --iso-8601=seconds)"
} >>"$log_file" 2>&1
