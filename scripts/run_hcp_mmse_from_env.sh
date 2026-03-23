#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${1:-$PROJECT_ROOT/configs/environment/ubuntu_data_layout.env}"
CONFIG_FILE="${2:-$PROJECT_ROOT/configs/experiment/hcp_mmse_baseline_linux.yaml}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-brainage-hcp-gpu}"

if [ ! -f "$ENV_FILE" ]; then
  echo "Environment file not found: $ENV_FILE"
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

mkdir -p "$HCP_MMSE_OUTPUT_DIR" "$HCP_MMSE_CACHE_DIR"
export PYTHONPATH="${PYTHONPATH:-src}"

cd "$PROJECT_ROOT"
conda run -n "$CONDA_ENV_NAME" python -m brainage.experiments.run_hcp_mmse --config "$CONFIG_FILE"
