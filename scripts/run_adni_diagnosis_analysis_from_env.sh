#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="${1:-$PROJECT_ROOT/configs/experiment/adni_diagnosis_analysis.yaml}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-brainage-hcp-gpu}"

export PYTHONPATH="${PYTHONPATH:-src}"
cd "$PROJECT_ROOT"
conda run -n "$CONDA_ENV_NAME" python -m brainage.experiments.run_adni_diagnosis_analysis --config "$CONFIG_FILE"
