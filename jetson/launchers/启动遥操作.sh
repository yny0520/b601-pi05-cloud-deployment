#!/usr/bin/env bash
set -euo pipefail
cd /home/jetson/Documents/ChatGPT/b601-rs
exec /usr/bin/python3 -u tools/b601_rs_teleop.py "$@"
