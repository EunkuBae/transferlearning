#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <env_file> <config_path>"
  exit 1
fi

env_file="$1"
config_path="$2"

if [ -f "$env_file" ]; then
  set -a
  source "$env_file"
  set +a
fi

export PYTHONPATH="src:${PYTHONPATH:-}"
python -m brainage.experiments.run_mmse_pretraining --config "$config_path"
