# B601 Cloud Policy Deployment

**双相机观测、云端 Pi0.5 推理与 Jetson 七轴机械臂执行。**

本项目记录一套已完成螺丝刀抓取、盒内释放和回撤演示的部署方案。基于上游 OpenPI/Pi0.5 做 B601 数据适配与 LoRA 微调，工程工作包括远程推理接口、单位契约、本地轨迹执行、夹爪适配、J2 部分重力补偿及故障诊断。基础模型由 Physical Intelligence 提供，本仓库不包含其模型实现全集或权重。

This repository documents a dual-camera cloud-policy deployment for a Seeed B601-RS arm, with local execution on Jetson Orin NX. It contains B601 integration, training overlays and deployment code, not the complete Pi0.5 model or its weights.

## Architecture

```mermaid
flowchart LR
 A[Top RGB + Wrist RGB + Joint State] --> B[Jetson Observation / Worker]
 B --> C[SSH / WebSocket]
 C --> D[A100 / OpenPI Pi0.5]
 D --> E[16 x 7 Absolute Radians]
 E --> F[Jetson Trajectory / MIT / J2 Feedforward]
 F --> G[PEAK USB-CAN / B601-RS]
```

## Evidence and limits

- 第一轮33次预测/528完成点，用户现场确认成功；第二轮36次预测、35段560点完整执行，相机见抓取、入盒释放、回撤。
- 第二轮七轴最高反馈温度46°C。没有统计成功率、长期可靠性或24小时连续运行验收。
- 第二轮原180秒预算段内到期被软件记为error，额外监督回收未执行；Pi自身已回撤。已知问题保留。
- 按作者确认，当前 Pi0.5 配置取消 J1–J6（含 J2）独立速度上限，`speed_deg_s=null`；本次代码修订仅做静态校验，历史演示记录不作为本次修订的真机复测结果。
- 放置时手柄在盒内、杆端跨盒沿。视频暂未加入；演示录好后放入`media/`或README链接，不伪造视频/性能。

## Repository map

| Directory | Contents |
|---|---|
| `jetson/tools/` | 成功版控制/采集/远程worker及相关离线测试 |
| `jetson/calibration/` | 部署数值配置，Pi0.5 六轴速度上限已按作者确认取消 |
| `jetson/launchers/` | 原Jetson Linux入口，涉及真实硬件 |
| `server/inference/` | 本次从实际服务器采集的推理服务及unit |
| `server/openpi_overlay/` | B601适配及实际修改的上游文件、训练runner，不是完整OpenPI |
| `server/environment/` | 云端实测依赖版本、上游配置/锁文件参考 |
| `training/` | 数据划分、归一化统计、训练说明 |
| `third_party/seeed_b601/` | 带MIT许可证的重力计算与RS URDF |
| `docs/` | 架构、部署约定、成功基线、代码来源 |
| `scripts/` | 可配置隧道入口与纯静态校验 |

## Start here

1. 阅读[部署说明](docs/DEPLOYMENT.md)、[成功基线](docs/BASELINE.md)和[第三方说明](THIRD_PARTY_NOTICES.md)。
2. Windows仅做阅读、静态校验、Git管理；不要在Windows运行Linux电机启动入口。
3. Windows或Linux Python 3.10+可执行：`python scripts/verify_source.py`。它只校验文件和语法，不import控制代码、不联网、不操作硬件。
4. 真机代码保持原Linux路径布局。先按部署文档恢复环境和配置，不能把下载后直接执行等同于安装完成。

## Scope

以成功版导出为基础，按作者确认取消 Pi0.5 六轴独立速度上限，未加入平滑调优。当前文件哈希见 `FILE_MANIFEST.json`；原导出来源哈希见 `docs/SOURCE_PROVENANCE.json`。没有下载/发布checkpoint、训练视频、SDK二进制、内核驱动、密码或私人服务器地址。GitHub发布不包含下载过程用到的个人SSH信息。

OpenPI改动文件保留上游Apache-2.0许可证；Seeed依赖保留MIT许可证。自定义工程代码尚未指定开源许可证，上传GitHub不自动授予重许可；详见第三方说明。
