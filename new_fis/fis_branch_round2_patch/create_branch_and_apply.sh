#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 /path/to/FIS_ENHANCE"
  exit 1
fi

REPO_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

git checkout main
git pull || true
git tag -a paper-fis-v1-stable -m "Stable version used for current manuscript" || true
git checkout -b feature/fis-channel-aware-controller-r2 || git checkout feature/fis-channel-aware-controller-r2

cp "$SCRIPT_DIR/patches/channel.py" .
cp "$SCRIPT_DIR/patches/fis_modules.py" .
cp "$SCRIPT_DIR/patches/model.py" .
cp "$SCRIPT_DIR/patches/train_baseline.py" .
cp "$SCRIPT_DIR/optional_tools/run_paper_sims.py" .
cp "$SCRIPT_DIR/optional_tools/diagnose_controller.py" .
cp "$SCRIPT_DIR/optional_tools/make_tables_from_json.py" .
cp "$SCRIPT_DIR/optional_tools/export_rule_table.py" .

python -m py_compile channel.py fis_modules.py model.py train_baseline.py run_paper_sims.py diagnose_controller.py make_tables_from_json.py export_rule_table.py

git status

echo "Patch applied on branch feature/fis-channel-aware-controller-r2"
