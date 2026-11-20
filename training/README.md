# Pi0.5 LoRA 微调

西安电子科技大学杭州研究院 · B601-Pi0.5 云端 VLA 操控系统

`server/openpi_overlay/` 包含 B601 训练配置、数据加载改动、日志插桩 runner 和启动代码。`training/episode_split.json` 记录数据划分，`norm_stats.json` 记录训练集归一化统计。

| 参数 | 配置 |
|---|---|
| 模型 | Pi0.5 LoRA |
| LoRA 模块 | `gemma_2b_lora` + `gemma_300m_lora` |
| Batch / 更新次数 | 8 / 6,000 |
| 训练数据 | 59 episodes，15,997 帧 |
| 采样频率 / horizon | 14 Hz / 16 |
| 归一化 | 训练集 z-score |
| 视频读取 | PyAV |
| 接口维度 | 模型内部 32 维，B601 使用前 7 维 |
| 最终 checkpoint | `5999`，对应 6,000 次更新 |
| 训练时长 | 约 1 小时 58 分 |

适配器在训练时传递 `actions`，在推理时使用观测输入。基础模型、tokenizer、checkpoint 和训练数据在训练环境中单独配置，环境版本见 [`server/environment/`](../server/environment/)。
