#!/usr/bin/env bash
# Set up an always-on Ubuntu server (x86 or ARM) to run:
#   btc-recorder.service  live spot + futures recorder, restarted on failure and at boot
#   btc-history.timer     daily update of the historical data (02:30 UTC)
#
# Usage, as a normal user with sudo:
#   curl -fsSL https://raw.githubusercontent.com/Zephyr-Howson/BTC-HFT/main/deploy/setup.sh | bash
# Re-run it any time to pull the latest code and restart.
#
# Overrides: REPO_URL, APP_DIR (default ~/BTC-HFT), DATA_DIR (default $APP_DIR/data),
#            HISTORY_ARGS (default "--trades-start 2026-01").
set -euo pipefail

REPO_URL=${REPO_URL:-https://github.com/Zephyr-Howson/BTC-HFT.git}
APP_DIR=${APP_DIR:-$HOME/BTC-HFT}
DATA_DIR=${DATA_DIR:-$APP_DIR/data}
HISTORY_ARGS=${HISTORY_ARGS:---trades-start 2026-01}
RUN_AS=$(id -un)

echo "== packages (git, python venv, chrony for NTP time sync)"
sudo apt-get update -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q git python3-venv python3-pip chrony
sudo systemctl enable --now chrony

echo "== code -> $APP_DIR"
if [ -d "$APP_DIR/.git" ]; then
    git -C "$APP_DIR" pull --ff-only
else
    git clone "$REPO_URL" "$APP_DIR"
fi
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install -q --upgrade pip
"$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"
mkdir -p "$DATA_DIR"

echo "== systemd units"
sudo tee /etc/systemd/system/btc-recorder.service >/dev/null <<EOF
[Unit]
Description=BTC market-data recorder (Binance spot + futures)
After=network-online.target
Wants=network-online.target

[Service]
User=$RUN_AS
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/.venv/bin/python -m recorder --out $DATA_DIR
Restart=always
RestartSec=10
# SIGTERM makes the recorder flush its in-memory buffers before exiting.
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/systemd/system/btc-history.service >/dev/null <<EOF
[Unit]
Description=Update BTC historical data from data.binance.vision
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=$RUN_AS
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/.venv/bin/python -m history --out $DATA_DIR/history $HISTORY_ARGS
Nice=10
EOF

sudo tee /etc/systemd/system/btc-history.timer >/dev/null <<EOF
[Unit]
Description=Daily BTC historical data update

[Timer]
OnCalendar=*-*-* 02:30:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable btc-recorder.service btc-history.timer
sudo systemctl restart btc-recorder.service
sudo systemctl start btc-history.timer

echo
echo "Done. Recorder status:"
systemctl --no-pager --lines=5 status btc-recorder.service || true
cat <<EOF

Useful commands:
  journalctl -u btc-recorder -f                      # live recorder log
  $APP_DIR/.venv/bin/python $APP_DIR/scripts/inspect_data.py $DATA_DIR
  sudo systemctl start btc-history.service           # run the history update now
  journalctl -u btc-history -n 50                    # last history update log
  df -h $DATA_DIR                                    # disk space
EOF
