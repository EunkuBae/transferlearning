#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-hcp_mmse}"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  exec tmux attach -t "$SESSION_NAME"
fi

echo "tmux session '$SESSION_NAME' was not found."
echo "Existing sessions:"
tmux ls || true
exit 1
