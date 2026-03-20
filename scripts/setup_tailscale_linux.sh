#!/usr/bin/env bash
set -euo pipefail

if ! command -v tailscale >/dev/null 2>&1; then
  curl -fsSL https://tailscale.com/install.sh | sh
fi

if ! command -v tmux >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y tmux
  else
    echo "tmux is not installed. Install it with your package manager and rerun this script."
    exit 1
  fi
fi

sudo tailscale up --ssh --accept-routes

echo ""
echo "Tailscale is up. Useful checks:"
echo "  tailscale status"
echo "  tailscale ip -4"
echo ""
echo "Connect from your local machine with one of these:"
echo "  ssh <linux-user>@<tailscale-ip>"
echo "  ssh <linux-user>@<tailscale-device-name>"
