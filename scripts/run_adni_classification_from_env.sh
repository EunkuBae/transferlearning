#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${1:-$PROJECT_ROOT/configs/environment/ubuntu_data_layout.env}"
CONFIG_FILE="${2:-$PROJECT_ROOT/configs/experiment/adni_cls_baseline_linux.yaml}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-brainage-hcp-gpu}"

if [ ! -f "$ENV_FILE" ]; then
  echo "Environment file not found: $ENV_FILE"
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

export PYTHONPATH="${PYTHONPATH:-src}"

if command -v python3 >/dev/null 2>&1; then
  METADATA_PYTHON=python3
else
  METADATA_PYTHON=python
fi

readarray -t RUN_METADATA < <(
  PROJECT_ROOT="$PROJECT_ROOT" CONFIG_FILE="$CONFIG_FILE" "$METADATA_PYTHON" - <<'PY'
import os
from pathlib import Path
import yaml

project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
config_path = Path(os.environ["CONFIG_FILE"]).expanduser()
if not config_path.is_absolute():
    config_path = (project_root / config_path).resolve()

config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
outputs = config.get("outputs", {})
experiment_name = str(config.get("experiment_name", config_path.stem))

def resolve_path(raw_value: str | None, default_value: str) -> Path:
    value = raw_value or default_value
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path

output_dir = resolve_path(outputs.get("run_dir"), f"outputs/{experiment_name}")
cache_dir = resolve_path(outputs.get("cache_dir"), f"outputs/cache/{experiment_name}")
print(str(config_path))
print(experiment_name)
print(str(output_dir))
print(str(cache_dir))
PY
)

CONFIG_FILE_ABS="${RUN_METADATA[0]}"
EXPERIMENT_NAME="${RUN_METADATA[1]}"
OUTPUT_DIR="${RUN_METADATA[2]}"
CACHE_DIR="${RUN_METADATA[3]}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_HISTORY_ROOT="$OUTPUT_DIR/run_history"
RUN_RECORD_DIR="$RUN_HISTORY_ROOT/$TIMESTAMP"
RUN_LOG="$RUN_RECORD_DIR/run.log"
RUN_REGISTRY="$OUTPUT_DIR/run_registry.jsonl"
STARTED_AT="${TIMESTAMP:0:4}-${TIMESTAMP:4:2}-${TIMESTAMP:6:2}T${TIMESTAMP:9:2}:${TIMESTAMP:11:2}:${TIMESTAMP:13:2}"
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR" "$RUN_RECORD_DIR"
cp "$CONFIG_FILE_ABS" "$RUN_RECORD_DIR/config_snapshot.yaml"
printf '%s\n' "$CONFIG_FILE_ABS" > "$RUN_RECORD_DIR/config_path.txt"
printf '%s\n' "$EXPERIMENT_NAME" > "$RUN_RECORD_DIR/experiment_name.txt"
printf '%s\n' "$OUTPUT_DIR" > "$RUN_RECORD_DIR/output_dir.txt"
printf '%s\n' "$CACHE_DIR" > "$RUN_RECORD_DIR/cache_dir.txt"
env | sort > "$RUN_RECORD_DIR/runtime_env.txt"
if git -C "$PROJECT_ROOT" rev-parse HEAD >/dev/null 2>&1; then
  git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_RECORD_DIR/git_commit.txt"
fi
printf '%s\n' "conda run -n $CONDA_ENV_NAME python -m brainage.experiments.run_adni_classification --config $CONFIG_FILE_ABS" > "$RUN_RECORD_DIR/command.txt"

RUN_STATUS=success
export BRAINAGE_RUN_RECORD_DIR="$RUN_RECORD_DIR"
export BRAINAGE_RUN_STARTED_AT="$STARTED_AT"
cd "$PROJECT_ROOT"
if ! env BRAINAGE_RUN_STATUS=success conda run -n "$CONDA_ENV_NAME" python -m brainage.experiments.run_adni_classification --config "$CONFIG_FILE_ABS" 2>&1 | tee "$RUN_LOG"; then
  RUN_STATUS=failed
fi

ENDED_AT="$(date +%Y-%m-%dT%H:%M:%S)"
RUN_REGISTRY="$RUN_REGISTRY" RUN_RECORD_DIR="$RUN_RECORD_DIR" RUN_STATUS="$RUN_STATUS" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" CONFIG_FILE_ABS="$CONFIG_FILE_ABS" OUTPUT_DIR="$OUTPUT_DIR" CACHE_DIR="$CACHE_DIR" \
STARTED_AT="$STARTED_AT" ENDED_AT="$ENDED_AT" "$METADATA_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

record = {
    "experiment_name": os.environ["EXPERIMENT_NAME"],
    "config_path": os.environ["CONFIG_FILE_ABS"],
    "output_dir": os.environ["OUTPUT_DIR"],
    "cache_dir": os.environ["CACHE_DIR"],
    "run_record_dir": os.environ["RUN_RECORD_DIR"],
    "status": os.environ["RUN_STATUS"],
    "started_at": os.environ["STARTED_AT"],
    "ended_at": os.environ["ENDED_AT"],
    "metrics_json": str(Path(os.environ["OUTPUT_DIR"]) / "metrics.json"),
    "history_json": str(Path(os.environ["OUTPUT_DIR"]) / "history.json"),
    "summary_txt": str(Path(os.environ["OUTPUT_DIR"]) / "training_summary.txt"),
}
registry_path = Path(os.environ["RUN_REGISTRY"])
registry_path.parent.mkdir(parents=True, exist_ok=True)
with registry_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record) + "\n")
(Path(os.environ["RUN_RECORD_DIR"]) / "run_record.json").write_text(
    json.dumps(record, indent=2) + "\n",
    encoding="utf-8",
)
PY

if [ "$RUN_STATUS" != "success" ]; then
  echo "ADNI classification run failed. See: $RUN_LOG" >&2
  exit 1
fi

echo "ADNI classification run completed successfully."
echo "Output dir: $OUTPUT_DIR"
echo "Run record: $RUN_RECORD_DIR"


