#!/usr/bin/env bash
# Run every EDA script in order. Reports land in eda/reports/*.txt.
# Override the interpreter with: PYTHON=/path/to/python3 eda/run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
for s in 0*_*.py; do
  echo ">>> $s"
  "$PY" "$s" > /dev/null
done
