#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

"$REPO_ROOT/data_copy/run_main_to_srv.sh"
"$SCRIPT_DIR/run_default_strike_daily.sh"
