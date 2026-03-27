#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ENV_FILE="${1:-$PROJECT_ROOT/configs/environment/ubuntu_data_layout.env}"
HCP_CONFIG="${2:-$PROJECT_ROOT/configs/experiment/hcp_mmse_baseline_linux.yaml}"
OASIS_CONFIGS_RAW="${3:-$PROJECT_ROOT/configs/experiment/oasis_mmse_transfer_full_ft.yaml}"
REMOTE_NAME="${REMOTE_NAME:-origin}"
BRANCH_NAME="${BRANCH_NAME:-main}"
UPDATE_CONDA_ENV="${UPDATE_CONDA_ENV:-0}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-brainage-hcp-gpu}"

if [ ! -f "$ENV_FILE" ]; then
  echo "Environment file not found: $ENV_FILE" >&2
  exit 1
fi

if [ ! -f "$HCP_CONFIG" ]; then
  echo "HCP config not found: $HCP_CONFIG" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

echo "==> Pull latest code from $REMOTE_NAME/$BRANCH_NAME"
git pull "$REMOTE_NAME" "$BRANCH_NAME"

if [ "$UPDATE_CONDA_ENV" = "1" ]; then
  echo "==> Update conda environment: $CONDA_ENV_NAME"
  conda env update -n "$CONDA_ENV_NAME" -f "$PROJECT_ROOT/configs/environment/linux_gpu_hcp_mmse.yml" --prune
fi

echo "==> Run HCP baseline"
bash "$PROJECT_ROOT/scripts/run_hcp_mmse_from_env.sh" "$ENV_FILE" "$HCP_CONFIG"

IFS=',' read -r -a OASIS_CONFIGS <<< "$OASIS_CONFIGS_RAW"

for raw_config in "${OASIS_CONFIGS[@]}"; do
  config="$(echo "$raw_config" | xargs)"
  if [ -z "$config" ]; then
    continue
  fi
  if [ ! -f "$config" ]; then
    echo "OASIS config not found: $config" >&2
    exit 1
  fi
  echo "==> Run OASIS transfer: $config"
  bash "$PROJECT_ROOT/scripts/run_oasis_transfer_from_env.sh" "$ENV_FILE" "$config"
done

echo "==> Completed HCP baseline and OASIS transfer reruns"
