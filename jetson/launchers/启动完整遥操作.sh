#!/usr/bin/env bash
set -euo pipefail

# B601 Star Arm -> B601-RS：七关节完整软件范围跟随。
# 自动识别PEAK USB-CAN；不加 --span-deg，夹爪使用配置中的固定开合端点。
# 显式传入 --channel 时由用户参数覆盖默认值。
exec /home/jetson/Desktop/b601-rs/启动遥操作.sh --live --joint all --full-range --channel auto "$@"
