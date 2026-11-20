# 训练交接

`server/openpi_overlay`包含实际训练配置、数据加载改动、训练日志插桩runner和启动代码。`training/episode_split.json`、`norm_stats.json`为该训练run数据约定，不是数据集本身。

模型Pi0.5 LoRA，batch8，6000更新，train episodes59/帧15,997，14Hz/horizon16；train-only z-score，PyAV，内部32D接口，B601用前7维。最终编号5999对应6000更新。训练约1小时58分；未报告独立validation/test任务成功率或best选择。

推理适配器允许没有actions；训练传actions时仍保留。实际推理版本适配器以本次服务器文件为准。此包不含完整Pi0.5源码、base、checkpoint、视频、tokenizer或全部环境。
