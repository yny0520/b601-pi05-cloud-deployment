#!/bin/sh
# Install root-owned as /usr/local/sbin/b601-io. Accept only fixed B601 I/O actions.
set -eu
if [ "$#" -ne 1 ]; then exit 2; fi
case "$1" in
  serial-stop) exec /usr/bin/systemctl stop uxrce-agent.service ;;
  serial-start) exec /usr/bin/systemctl start uxrce-agent.service ;;
  can-config|can-up)
    can_channel=$(/usr/bin/awk '$2 == "usb" && $3 ~ /^can[0-9]+$/ {name=$3; count++} END {if (count != 1) exit 3; print name}' /proc/pcan)
    # PEAK netdevs here are virtual; /proc/pcan identifies the USB interface.
    case "$(/usr/bin/readlink -f "/sys/class/net/$can_channel/device" 2>/dev/null || true)" in
      *mttcan*) echo 'refusing platform CAN interface' >&2; exit 3 ;;
    esac
    flags=$(/usr/bin/cat "/sys/class/net/$can_channel/flags")
    if [ "$((flags & 1))" -ne 0 ]; then
      echo "$can_channel already UP; no reset performed"
      exit 0
    fi
    if [ "$1" = can-config ]; then
      exec /usr/sbin/ip link set dev "$can_channel" type can bitrate 1000000 restart-ms 100 loopback off
    fi
    exec /usr/sbin/ip link set dev "$can_channel" up
    ;;
  *) exit 2 ;;
esac
