#!/usr/bin/env bash
set -eo pipefail
cd /home/jetson/Documents/ChatGPT/b601-rs
source /opt/ros/humble/setup.bash
source /home/jetson/orbbec_ws/install/setup.bash
set -u

# Keep one launcher per user; do not open the cameras twice.
exec 9>"/tmp/b601-insertion-launch-${UID}.lock"
if ! flock -n 9; then
    echo '采集程序已经启动，请打开 http://127.0.0.1:8766/'
    exit 0
fi
if /usr/bin/python3 -c 'import socket; s=socket.create_connection(("127.0.0.1",8766),timeout=1); s.close()' 2>/dev/null; then
    echo '8766端口已有服务，请打开 http://127.0.0.1:8766/'
    exit 0
fi

camera_pid=''
cleanup() {
    if [[ -n "$camera_pid" ]] && kill -0 "$camera_pid" 2>/dev/null; then
        kill -INT "$camera_pid" 2>/dev/null || true
        wait "$camera_pid" || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! pgrep -f '^/usr/bin/python3 /opt/ros/humble/bin/ros2 launch orbbec_camera gemini_330_series.launch.py .*camera_name:=camera_preview' >/dev/null; then
    camera_log="/tmp/b601-camera-preview-driver-$(date +%Y%m%d-%H%M%S).log"
    echo "启动336L彩色与深度相机，日志：$camera_log"
    ros2 launch orbbec_camera gemini_330_series.launch.py \
        camera_name:=camera_preview enable_color:=true enable_depth:=true \
        depth_registration:=false enable_point_cloud:=false \
        enable_left_ir:=false enable_right_ir:=false \
        enable_accel:=false enable_gyro:=false enable_frame_sync:=false \
        color_width:=640 color_height:=480 color_fps:=30 \
        depth_width:=848 depth_height:=480 depth_fps:=30 \
        >"$camera_log" 2>&1 &
    camera_pid=$!
fi
echo '打开 http://127.0.0.1:8766/；点击页面“一键启动并跟随”后机械臂才开始跟随。'
/usr/bin/python3 -u tools/b601_insertion_collector.py --port 8766 --camera-hz 15 --hand-camera-hz 30
