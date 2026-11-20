# 部署与接口

## Windows用途

下载ZIP、校验SHA256、解压后阅读/静态校验/上传GitHub。硬件执行依赖Linux SocketCAN、ROS2、MotorBridge，Windows不会因安装Python就获得控制能力。

## 原成功版Jetson布局

源码放`/home/jetson/Documents/ChatGPT/b601-rs`，对应本包`jetson/`下tools及experiments；配置和入口位于`/home/jetson/Desktop/b601-rs`，分别来自`jetson/calibration`和`jetson/launchers`。重力依赖`third_party/seeed_b601`两文件对应桌面`B601-RS-lerobot/lerobot_robot_seeed_b601/`下路径。

这些个人路径约定暂保留以避免整理时改变成功版行为，移机需要统一调整路径；本仓库不声称开箱即用。原项目另含旧ACT/SmolVLA模型分支，其依赖/权重不在本包，使用本包仅选择pi05_remote。

系统Python3.10/aarch64、MotorBridge0.5.6、PEAK netdev驱动8.19.0、ROS2 Humble、cv_bridge/sensor_msgs/Orbbec相机驱动。MotorBridge/PCAN安装包需从厂商合法渠道取得，不包含在仓库。

CAN ID1～3 RS06，4～7 RS00，HOST0xFD；`--channel auto`识别唯一外置PEAK USB接口，不能永久把can0当身份。自主模式仅读从臂；人工主臂FashionStar UART与Pi自主执行分开。

网页8766：采集、双RGB和遥测/VLA控制。8767为原项目标定服务（未作为本次Pi部署主入口）。状态API `/api/teleop/status`、`/api/vla/status`、`/api/status`。POST令牌运行时生成，不归档。新帧和遥测均要求300ms内；预测按原观测3秒计算有效期。

本地边界：JSON-lines worker→MessagePack ndarray/WebSocket→SSH本地18015→云端回环8015。`config/connection.env.example`填写个人SSH参数后用`scripts/connect_cloud.sh`连接；不要把实际配置提交Git。认证后后台SSH，终端关闭不使隧道退出；不自动使能电机。

运行消息位于`/dev/shm/b601-rs-<uid>/`，日志有界异步，遥测非阻塞Unix datagram；PID/epoch/取消检查保证旧请求不跨控制会话使用。控制进程MotorBridge发MIT、SocketCAN批读7×mechPos及J7速度，约14Hz。

## 云端

服务器Python3.11环境，已修改OpenPI位于`/root/experiments/pi05_b601/openpi-src`；实际版本见`server/environment/cloud-environment.json`。仓库只给overlay，不包含完整上游。把overlay对照匹配上游再使用；原服务器未保留可解析Git提交，不能保证将当前最新OpenPI直接覆盖就兼容。

云端服务和配置保留原`/root/experiments/pi05_b601`路径，checkpoint需要另外合法获取。`server/inference/serve_b601.py`导入即加载模型并开始服务，**不是离线导入检查目标**。它通过validation数据首样本预热，因此还需要对应本地LeRobot数据路径；本包只含split/归一化，不含原视频。

完整推理还依赖上游base、tokenizer、checkpoint assets以及OpenPI其余源码；这不是可独立加载的LoRA权重文件。服务仅监听127.0.0.1:8015。systemd单元为实际服务器采集副本；启动需要GPU资源，本次打包不启动服务。

## 操作边界

模型停止请求保持；退出控制器/SIGTERM会失能。修改源码、安装驱动或恢复配置前应可靠支撑并停止相关控制；不要重启正在使能的控制器。回归必须通过`jetson/tools/run_b601_offline_checks.py`持锁入口，只在正确原布局、控制器停止时运行。纯静态校验脚本不接硬件，可以Windows执行。
