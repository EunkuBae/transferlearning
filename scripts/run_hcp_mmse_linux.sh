#!/usr/bin/env bash
set -euo pipefail

: "${HCP_MMSE_CSV:?Set HCP_MMSE_CSV to the MMSE CSV on the external SSD}"
: "${HCP_IMAGE_DIR:?Set HCP_IMAGE_DIR to the HCP NIfTI directory on the external SSD}"
: "${HCP_MMSE_OUTPUT_DIR:?Set HCP_MMSE_OUTPUT_DIR to an internal-disk output directory}"
: "${HCP_MMSE_CACHE_DIR:?Set HCP_MMSE_CACHE_DIR to an internal-disk cache directory}"

export PYTHONPATH="${PYTHONPATH:-src}"

python -m brainage.experiments.run_hcp_mmse --config configs/experiment/hcp_mmse_baseline.yaml
