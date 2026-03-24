#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${1:-$PROJECT_ROOT/configs/environment/ubuntu_data_layout.env}"
CONFIG_FILE="${2:-$PROJECT_ROOT/configs/experiment/hcp_mmse_baseline_linux.yaml}"
SESSION_NAME="${SESSION_NAME:-$(basename "$CONFIG_FILE" .yaml)}"
REPO_DIR="${REPO_DIR:-$PROJECT_ROOT}"

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
print(str(output_dir / "tmux_logs"))
PY
)

LOG_DIR="${RUN_METADATA[0]}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${SESSION_NAME}_${TIMESTAMP}.log"
RUN_SCRIPT="$LOG_DIR/${SESSION_NAME}_${TIMESTAMP}_run.sh"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists."
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 1
fi

cat > "$RUN_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_DIR"
bash scripts/run_hcp_mmse_from_env.sh "$ENV_FILE" "$CONFIG_FILE" 2>&1 | tee -a "$LOG_FILE"
EOF

chmod +x "$RUN_SCRIPT"
tmux new-session -d -s "$SESSION_NAME" "bash '$RUN_SCRIPT'"

echo "Started tmux session: $SESSION_NAME"
echo "Config file: $CONFIG_FILE"
echo "Log file: $LOG_FILE"
echo "Runner script: $RUN_SCRIPT"
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "Detach with: Ctrl-b then d"
