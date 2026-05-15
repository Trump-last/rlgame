## Purpose

PPO 算法训练流程，包括环境创建、模型配置、训练循环和模型保存。

## Requirements

### Requirement: 训练环境配置
系统 SHALL 使用 Gymnasium 的 LunarLander-v2 环境作为训练任务。

#### Scenario: 环境初始化
- **WHEN** 训练脚本启动
- **THEN** 成功创建 LunarLander-v2 环境，状态空间为 8 维连续向量，动作空间为 4 个离散动作

### Requirement: PPO 模型训练
系统 SHALL 使用 stable-baselines3 的 PPO 算法训练模型，超参数固定如下：
- n_steps=2048, batch_size=64, n_epochs=10
- learning_rate=2.5e-4, gae_lambda=0.95, gamma=0.99
- clip_range=0.2, ent_coef=0.01
- total_timesteps=1_000_000

#### Scenario: GPU 训练
- **WHEN** 在具备 CUDA 的 GPU 机器上运行训练
- **THEN** 模型自动在 GPU 上训练，训练日志输出到控制台

### Requirement: 训练日志记录
系统 SHALL 通过 SB3 的 Monitor wrapper 自动记录每个 episode 的 reward 和长度到 logs/monitor.csv。

#### Scenario: 日志文件生成
- **WHEN** 训练完成
- **THEN** logs/monitor.csv 文件存在，包含 episode、reward、length、timestep 列

### Requirement: 模型保存
系统 SHALL 在训练完成后将模型保存到 models/ppo_lunarlander.zip。

#### Scenario: 模型文件保存
- **WHEN** 训练达到 total_timesteps 步数
- **THEN** models/ppo_lunarlander.zip 文件存在且可加载
