# 参数与运行记录

西安电子科技大学杭州研究院 · B601-Pi0.5 云端 VLA 操控系统

## 模型与输入输出

模型为 `pi05_b601_lora_6000`，checkpoint `formal_20261120_01/5999` 对应 6,000 次更新。

输入包括顶置相机 `camera1`（640 × 480 RGB）、腕部相机 `camera2`（640 × 360 RGB）、七维实测绝对弧度及英文任务指令。策略输出 `16 × 7` 绝对弧度动作序列，控制端统一转换为角度。

## 训练参数

- 14 Hz，horizon 16，batch 8，6,000 次更新。
- 59 个训练 episodes，共 15,997 帧；归一化使用训练集 z-score。
- LoRA：`gemma_2b_lora` + `gemma_300m_lora`。
- 模型内部接口为 32 维，B601 使用前 7 维；推理采用 10 步 flow。

## 执行参数

- `complete_chunk=true`、`continuous_plan=true`、`rolling_replan=false`；每 16 点完整执行后重新观测。
- J1–J6：`speed_deg_s=null`，不设独立速度上限；加速度 10°/s²。J7：60°/s、120°/s²。
- 执行预算为 40 轮 / 180 秒，单段 35 秒，连续 4 秒无推进停止。
- J1–J6：lead 8°、abort 10°、settle 3.5°。J7：lead 45°、abort 50°、settle 25°。
- 观测有效期 3 秒；新帧和遥测有效期 300 ms。
- `task_bounds_deg=null`、`grasp_gate=false`、`action_filter_enabled=false`；关节软限位由底层控制器执行。
- J1–J6 Kp：`[50,150,150,60,50,50]`；Kd：`[3,10,10,5,4,4]`。
- J2 重力前馈：scale 0.6、转矩上限 8 N·m、转矩变化率 1.5 N·m/s；其余轴前馈为 0。
- 夹爪：主机 Kp 12 / Kd 0.6，转矩指令 ±1.8 N·m，MIT kd 1.5。

配置文件：[`vla_pi05_pick_place_motion_config.json`](../jetson/calibration/vla_pi05_pick_place_motion_config.json)、[`star_arm_teleop_config.json`](../jetson/calibration/star_arm_teleop_config.json)。

## 2026-11-20 运行记录

| 轮次 | 预测与执行 | 时长 | 温度 | 过程 |
|---|---|---|---|---|
| 第一轮 | 33 次预测，528 点执行 | 约 182 秒 | J2/J3 最高 44°C | 完成抓取放置，监督器用 2.845 秒回到起始姿态 |
| 第二轮 | 36 次预测，35 段 / 560 点执行 | 约 183 秒 | 七轴最高 46°C | 抓起螺丝刀、盒内释放、空爪回撤 |

第二轮在动作段内达到 180 秒预算，日志状态为 `error`。停止时控制 epoch 更新，监督器取消额外的起点对齐。放置后手柄位于盒内，杆端跨在盒沿。

动作段末速度接近零，等待约 0.6 秒稳定后采集下一次观测。第二轮段间隔为 1.255～2.302 秒，中位数 1.401 秒。

[演示视频](../media/b601-pi05-demo.mp4)
