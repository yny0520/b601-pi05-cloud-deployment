#!/bin/bash
set -euo pipefail
cd /home/jetson/Documents/ChatGPT/b601-rs
sudo -v
sudo -n /usr/sbin/visudo -cf tools/b601-web.sudoers
sudo -n install -o root -g root -m 755 tools/b601_io_helper.sh /usr/local/sbin/b601-io
sudo -n install -o root -g root -m 440 tools/b601-web.sudoers /etc/sudoers.d/b601-web-io
sudo -n /usr/sbin/visudo -cf /etc/sudoers.d/b601-web-io
