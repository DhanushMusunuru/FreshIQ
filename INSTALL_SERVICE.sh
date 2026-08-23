#!/usr/bin/env bash
#
#   ./INSTALL_SERVICE.sh
#
# Makes FreshIQ start on its own whenever the board powers up, and restart
# itself if it ever crashes.
#
# Undo with:  sudo systemctl disable --now freshiq

set -eu
cd "$(dirname "${BASH_SOURCE[0]}")"
HERE="$PWD"
B=$'\033[1m'; G=$'\033[32m'; O=$'\033[0m'

if ! command -v systemctl >/dev/null 2>&1; then
  echo "systemd not found - nothing to install. Just run ./START.sh instead."
  exit 1
fi

echo "${B}Installing the FreshIQ service${O}   ($HERE, user $USER)"

sudo tee /etc/systemd/system/freshiq.service >/dev/null <<EOF
[Unit]
Description=FreshIQ food inspection station
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HERE
ExecStart=$HERE/START.sh
# The camera and the serial port live in these groups. Without them the service
# starts and then fails at the first frame or the first tray move.
SupplementaryGroups=dialout video
Restart=always
RestartSec=5
# Five crashes in a minute is a real fault, not a transient - stop retrying so
# it does not crash-loop forever and flood the journal.
StartLimitBurst=5
StartLimitIntervalSec=60
Environment=PYTHONUNBUFFERED=1
StandardOutput=journal
StandardError=journal
SyslogIdentifier=freshiq

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now freshiq

sleep 3
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo
if systemctl is-active --quiet freshiq; then
  echo "  ${G}FreshIQ is running and will start on every boot.${O}"
  [ -n "${IP:-}" ] && echo "  open  ${B}http://$IP:8000${O}"
else
  echo "  It did not come up. See why:"
  echo "     journalctl -u freshiq -n 50 --no-pager"
fi
echo
echo "  watch the log:  journalctl -u freshiq -f"
echo "  stop:           sudo systemctl stop freshiq"
