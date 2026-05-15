## Why

需要一个完整的 PPO 算法演示项目，涵盖训练、可视化、模型保存与推理全流程，作为强化学习学习的 end-to-end 示例。

## What Changes

- 新增 `train.py`：使用 SB3 的 PPO 在 GPU 上训练 LunarLander-v2，目标 mean reward 150+，训练日志通过 Monitor 自动记录到 `monitor.csv`
- 新增 `plot_rewards.py`：读取 `monitor.csv` 绘制 reward 变化曲线
- 新增 `visualize.py`：在 CPU 上加载训练好的模型，分别录制随机策略和训练后策略的游戏过程，保存为 GIF 动画
- 新增 `requirements.txt`：项目依赖声明

## Capabilities

### New Capabilities
- `ppo-training`: PPO 算法训练流程，包括环境创建、模型配置、训练循环、模型保存
- `reward-visualization`: 训练 reward 数据的后处理与可视化
- `model-inference`: 已训练模型的加载、推理与游戏动画录制

### Modified Capabilities
- (none)

## Impact

- 新增 Python 脚本和依赖，不影响现有代码
- 训练需要 GPU，推理可在 CPU 运行
