#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-hcp_mmse}"
REPO_DIR="${REPO_DIR:-$PWD}"
LOG_DIR="${LOG_DIR:-${HCP_MMSE_OUTPUT_DIR:-$REPO_DIR/outputs/hcp_mmse_baseline}/tmux_logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${SESSION_NAME}_${TIMESTAMP}.log"
RUN_SCRIPT="$LOG_DIR/${SESSION_NAME}_${TIMESTAMP}_run.sh"

: "${HCP_MMSE_CSV:?Set HCP_MMSE_CSV to the MMSE CSV on the external SSD}"
: "${HCP_IMAGE_DIR:?Set HCP_IMAGE_DIR to the HCP NIfTI directory on the external SSD}"
: "${HCP_MMSE_OUTPUT_DIR:?Set HCP_MMSE_OUTPUT_DIR to an internal-disk output directory}"
: "${HCP_MMSE_CACHE_DIR:?Set HCP_MMSE_CACHE_DIR to an internal-disk cache directory}"

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
export HCP_MMSE_CSV="$HCP_MMSE_CSV"
export HCP_IMAGE_DIR="$HCP_IMAGE_DIR"
export HCP_MMSE_OUTPUT_DIR="$HCP_MMSE_OUTPUT_DIR"
export HCP_MMSE_CACHE_DIR="$HCP_MMSE_CACHE_DIR"
export PYTHONPATH="${PYTHONPATH:-src}"
bash scripts/bootstrap_hcp_mmse_linux.sh 2>&1 | tee -a "$LOG_FILE"
EOF

chmod +x "$RUN_SCRIPT"
tmux new-session -d -s "$SESSION_NAME" "bash '$RUN_SCRIPT'"

echo "Started tmux session: $SESSION_NAME"
echo "Log file: $LOG_FILE"
echo "Runner script: $RUN_SCRIPT"
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "Detach with: Ctrl-b then d"
