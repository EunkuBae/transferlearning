#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/modeling}"
BRANCH="${BRANCH:-main}"
COMMIT_MSG="${COMMIT_MSG:-Update experiment results from Linux training run}"
REMOTE_NAME="${REMOTE_NAME:-origin}"

RESULT_FILENAMES=(
  metrics.json
  history.json
  resolved_paths.json
  test_predictions.csv
  training_summary.txt
)

cd "$REPO_DIR"

if [ ! -d .git ]; then
  echo "Not a git repository: $REPO_DIR" >&2
  exit 1
fi

collect_result_files() {
  local search_root="$1"
  if [ ! -d "$search_root" ]; then
    return 0
  fi

  find "$search_root" -type f \( \
    -name "metrics.json" -o \
    -name "history.json" -o \
    -name "resolved_paths.json" -o \
    -name "test_predictions.csv" -o \
    -name "training_summary.txt" \
  \) -print0
}

# Stage updates to already-tracked files first.
git add -u outputs configs/experiment/outputs scripts || true

# Stage canonical lightweight result artifacts from current output trees.
while IFS= read -r -d '' path; do
  git add -- "$path"
done < <(collect_result_files "outputs")

while IFS= read -r -d '' path; do
  git add -- "$path"
done < <(collect_result_files "configs/experiment/outputs")

# Force-add global experiment registries because outputs/metrics is gitignored.
if [ -f "outputs/metrics/experiment_runs.csv" ]; then
  git add -f -- "outputs/metrics/experiment_runs.csv"
fi
if [ -f "outputs/metrics/experiment_runs.jsonl" ]; then
  git add -f -- "outputs/metrics/experiment_runs.jsonl"
fi

# Keep optional seed summaries if they exist.
if [ -f "outputs/metrics/adni_classification_seed_summary.json" ]; then
  git add -f -- "outputs/metrics/adni_classification_seed_summary.json"
fi

if git diff --cached --quiet; then
  echo "No result changes to commit."
  exit 0
fi

echo "Staged result files:"
git diff --cached --name-only

git commit -m "$COMMIT_MSG"
git push "$REMOTE_NAME" "$BRANCH"

echo "Pushed result commit to $REMOTE_NAME/$BRANCH"
