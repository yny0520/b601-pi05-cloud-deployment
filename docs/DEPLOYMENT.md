# 部署说明

## 平台与依赖

| 位置 | 环境 |
|---|---|
| Jetson Orin NX | Linux / aarch64，Python 3.10，ROS2 Humble，MotorBridge 0.5.6，PEAK netdev 驱动 8.19.0 |
| 相机 | Orbbec 驱动、`cv_bridge`、`sensor_msgs` |
| 云端 GPU | Python 3.11，OpenPI / Pi0.5；版本见 [`cloud-environment.json`](../server/environment/cloud-environment.json) |
| Windows | 源码阅读、文件校验和 Git 管理 |

MotorBridge 和 PCAN 驱动由厂商渠道安装。云端需配置完整 OpenPI 源码、基础模型、tokenizer、checkpoint assets 及 B601 checkpoint。

## Jetson 路径

| 仓库内容 | 部署位置 |
|---|---|
| `jetson/tools/`、`jetson/experiments/` | `/home/jetson/Documents/ChatGPT/b601-rs/` 下对应目录 |
| `jetson/calibration/` | `/home/jetson/Desktop/b601-rs/calibration/` |
| `jetson/launchers/` | `/home/jetson/Desktop/b601-rs/` |
| `third_party/seeed_b601/` | 桌面 `B601-RS-lerobot/lerobot_robot_seeed_b601/` 下对应目录 |

迁移设备时统一配置上述路径。模型选择使用 `pi05_remote`。

## CAN 与本地控制

CAN ID 1～3 对应 RS06，4～7 对应 RS00，HOST 为 `0xFD`。`--channel auto` 按外置 PEAK USB 接口识别 CAN 通道。

自主模式读取从臂状态；人工主臂使用 FashionStar UART。控制进程通过 MotorBridge 发送 MIT 指令，通过 SocketCAN 批量读取七轴 `mechPos` 及 J7 速度，循环约 14 Hz。

Pi0.5 运动配置为 [`vla_pi05_pick_place_motion_config.json`](../jetson/calibration/vla_pi05_pick_place_motion_config.json)：

- `speed_deg_s=null`：J1–J6（含 J2）不设独立速度上限。
- `accel_deg_s2=10`：六轴加减速度。
- J7 使用独立的速度、加速度及转矩控制参数。
- 跟踪误差、目标超前、关节范围和超时由本地控制循环检查。

运动 JSON 与 `b601_vla_motion.py`、`b601_rs_teleop.py` 配套部署。J2 的 `slew_nm_s` 表示重力补偿转矩变化率，单位为 N·m/s。

运行消息写入 `/dev/shm/b601-rs-<uid>/`。日志采用有界异步写入，遥测采用非阻塞 Unix datagram；PID、epoch 和取消状态用于区分控制会话。

## 网页与策略接口

网页端口 `8766` 提供采集、双 RGB、遥测及 VLA 控制；`8767` 为标定服务。

状态接口：`/api/teleop/status`、`/api/vla/status`、`/api/status`。POST 令牌在运行时生成。新帧与遥测有效期为 300 ms，策略预测有效期为 3 秒。

策略通道为 JSON-lines worker → MessagePack ndarray / WebSocket → SSH 本地 `18015` → 云端回环 `8015`。

在本机设置 [`connection.env.example`](../config/connection.env.example) 中的连接变量，再通过 [`connect_cloud.sh`](../scripts/connect_cloud.sh) 建立后台 SSH 隧道。实际连接配置放在 Git 忽略的 `connection.env` 中。

## 云端服务

OpenPI 路径为 `/root/experiments/pi05_b601/openpi-src`。将 `server/openpi_overlay/` 的适配文件合入匹配的 OpenPI 环境，文件来源和哈希见 [`CLOUD_SOURCE_PROVENANCE.json`](CLOUD_SOURCE_PROVENANCE.json)。

`server/inference/serve_b601.py` 在加载时初始化模型并启动服务，使用 validation 数据的首个样本预热。部署需提供对应的本地 LeRobot 数据路径。训练集划分与归一化统计位于 `training/`。

服务监听 `127.0.0.1:8015`，systemd 单元位于 `server/inference/b601-pi05-inference.service`，工作目录为 `/root/experiments/pi05_b601`。

## 检查与维护

源码校验：

```bash
python scripts/verify_source.py
```

该脚本检查文件哈希、Python 语法、JSON 与本地模块引用。

Linux 离线回归入口为 `jetson/tools/run_b601_offline_checks.py`，运行时获取控制锁。控制器停止、机械臂可靠支撑后再修改配置、安装驱动或执行回归。模型停止请求进入保持状态；退出控制器或收到 SIGTERM 时失能。
