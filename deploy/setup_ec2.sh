#!/usr/bin/env bash
# One-shot Blackbox KI setup for a fresh Ubuntu EC2 box.
# Run it FROM the cloned repo dir as the `ubuntu` user:
#   cd ~/blackbox-ki && bash deploy/setup_ec2.sh
set -euo pipefail

APP_DIR="$HOME/blackbox-ki"
if [ "$(pwd)" != "$APP_DIR" ]; then
  echo "!! Run this from $APP_DIR (git clone the repo to exactly that path first)."; exit 1
fi

echo "==> 0/6  2GB swap (t2.micro has only 1GB RAM — prevents OOM during pip install)"
if [ ! -f /swapfile ]; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

echo "==> 1/6  System packages"
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv debian-keyring debian-archive-keyring apt-transport-https curl

echo "==> 2/6  Caddy (automatic HTTPS)"
if ! command -v caddy >/dev/null 2>&1; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y caddy
fi

echo "==> 3/6  Python dependencies (user install -> ~/.local/bin/gunicorn)"
pip3 install --user -r requirements.txt

echo "==> 4/6  Seed data stores"
python3 bootstrap.py

echo "==> 5/6  systemd services"
sudo cp deploy/blackbox-app.service /etc/systemd/system/
# Scanner is optional (needs GROQ for full value); enable if you want it:
# sudo cp deploy/trading-scanner.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now blackbox-app
# sudo systemctl enable --now trading-scanner

echo "==> 6/6  Caddy reverse proxy on a free sslip.io hostname"
# Cloud-agnostic public IP detection (works on AWS / Oracle / GCP / any VPS)
IP=$(curl -s --max-time 4 https://api.ipify.org || curl -s --max-time 4 http://169.254.169.254/latest/meta-data/public-ipv4)
DASHED=$(echo "$IP" | tr '.' '-')
HOST="app.${DASHED}.sslip.io"
sudo tee /etc/caddy/Caddyfile >/dev/null <<EOF
${HOST} {
    reverse_proxy localhost:8090
}
EOF
sudo systemctl restart caddy

echo
echo "DONE.  Your site is live (HTTPS, real cert) at:"
echo "    https://${HOST}"
echo
echo "Optional: add ~/blackbox-ki/.env with GROQ_API_KEY / OANDA_API_KEY, then:"
echo "    sudo systemctl restart blackbox-app"
