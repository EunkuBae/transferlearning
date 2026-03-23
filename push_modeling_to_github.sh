#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/modeling}"
REMOTE_URL="${REMOTE_URL:-https://github.com/EunkuBae/transferlearning.git}"
BRANCH="${BRANCH:-main}"
COMMIT_MSG="${COMMIT_MSG:-Update modeling from Ubuntu server}"

cd "$REPO_DIR"

if [ ! -d .git ]; then
  git init
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin "$REMOTE_URL"
else
  git remote set-url origin "$REMOTE_URL"
fi

git fetch origin || true

if ! git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
  git checkout -b "$BRANCH"
else
  git checkout "$BRANCH"
fi

git add .

if git diff --cached --quiet; then
  echo "No changes to commit."
  exit 0
fi

git commit -m "$COMMIT_MSG"
git push -u origin "$BRANCH"
