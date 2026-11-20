# 代码来源与许可证

项目归属：西安电子科技大学杭州研究院（西电杭研院）。工程内容包括 B601 数据适配、Pi0.5 LoRA 微调、云端策略服务与 Jetson 本地执行。

| 来源 | 使用内容 | 许可证 / 版本 |
|---|---|---|
| [Physical Intelligence / OpenPI](https://github.com/Physical-Intelligence/openpi) | Pi0.5 基础模型接口、训练配置、数据加载和训练代码 | Apache-2.0，见 [`server/openpi_overlay/LICENSE`](server/openpi_overlay/LICENSE) |
| [Seeed / B601-RS](https://github.com/Seeed-Projects/lerobot-robot-seeed-b601) | 重力计算与 RS URDF | MIT，来源 commit `1fa325dfd76493d4f8f25886078b96cc056528dd`，见 [`third_party/seeed_b601/LICENSE`](third_party/seeed_b601/LICENSE) |
| MotorBridge、PEAK、ROS2、Orbbec | 电机通信、CAN 驱动及相机环境 | 按各组件的许可证和发行渠道安装 |

OpenPI 适配文件通过逐文件 SHA256 记录版本。Jetson 来源记录见 [`docs/SOURCE_PROVENANCE.json`](docs/SOURCE_PROVENANCE.json)，云端来源记录见 [`docs/CLOUD_SOURCE_PROVENANCE.json`](docs/CLOUD_SOURCE_PROVENANCE.json)，当前文件校验清单见 [`FILE_MANIFEST.json`](FILE_MANIFEST.json)。

模型权重按上游权重许可单独配置。自定义工程代码暂未设置项目级许可证。
