#!/usr/bin/env bash
set -euo pipefail

if ! command -v firewall-cmd >/dev/null 2>&1; then
  echo "firewall-cmd not found. Install or enable firewalld first." >&2
  exit 1
fi

sudo systemctl enable --now firewalld
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --permanent --remove-port=8000/tcp
sudo firewall-cmd --reload
sudo firewall-cmd --list-services
