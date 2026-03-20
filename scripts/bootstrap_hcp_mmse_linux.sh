#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/EunkuBae/transferlearning.git}"
REPO_DIR="${REPO_DIR:-transferlearning}"
ENV_NAME="${ENV_NAME:-brainage-hcp-gpu}"
ENV_FILE="configs/environment/linux_gpu_hcp_mmse.yml"
BASELINE_CONFIG="configs/experiment/hcp_mmse_baseline_linux.yaml"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  conda env create -f "$ENV_FILE"
fi

: "${HCP_MMSE_CSV:?Set HCP_MMSE_CSV to the MMSE CSV on the external SSD}"
: "${HCP_IMAGE_DIR:?Set HCP_IMAGE_DIR to the HCP NIfTI directory on the external SSD}"
: "${HCP_MMSE_OUTPUT_DIR:?Set HCP_MMSE_OUTPUT_DIR to an internal-disk output directory}"
: "${HCP_MMSE_CACHE_DIR:?Set HCP_MMSE_CACHE_DIR to an internal-disk cache directory}"

export PYTHONPATH="${PYTHONPATH:-src}"

conda run -n "$ENV_NAME" python -m brainage.experiments.run_hcp_mmse --config "$BASELINE_CONFIG"
