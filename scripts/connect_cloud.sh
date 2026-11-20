#!/usr/bin/env bash
set -euo pipefail
: "${B601_SSH_HOST:?Set B601_SSH_HOST}"
: "${B601_SSH_USER:?Set B601_SSH_USER}"
exec ssh -f -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=20 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=yes \
 -L "127.0.0.1:${B601_LOCAL_PORT:-18015}:127.0.0.1:${B601_REMOTE_PORT:-8015}" \
 -p "${B601_SSH_PORT:-22}" "${B601_SSH_USER}@${B601_SSH_HOST}" \
 'systemctl start b601-pi05-inference.service && exec tail -f /dev/null'
