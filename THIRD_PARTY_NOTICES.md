# 代码来源与许可证

- Physical Intelligence OpenPI：https://github.com/Physical-Intelligence/openpi 。`server/openpi_overlay`中的config/data_loader/train相关代码含Apache-2.0上游代码及本项目改动；原许可证在`server/openpi_overlay/LICENSE`。服务器源码目录没有可解析Git提交，本导出用逐文件SHA256固定实际修改文件，不能声称对应某个未核实上游commit。模型权重许可另行适用，本包不发布权重。
- Seeed B601-RS：https://github.com/Seeed-Projects/lerobot-robot-seeed-b601 ，B601-RS分支，本机来源commit `1fa325dfd76493d4f8f25886078b96cc056528dd`。重力计算/RS URDF及原MIT许可证保存在`third_party/seeed_b601`。
- MotorBridge0.5.6、PEAK驱动、ROS2/Orbbec等作为外部依赖；本包不再分发SDK二进制/安装包/内核模块。
- 项目自定义集成代码未另设统一开源许可证。第三方许可证只覆盖其适用文件，不给未授权文件重新许可。未来选择项目许可证时需由项目作者决定，并保留上述署名。

展示中准确描述为基于Pi0.5/OpenPI的微调与部署工程，不称从零开发Pi0.5或给出未经测量的成功率。
