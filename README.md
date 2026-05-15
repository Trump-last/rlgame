# PPO LunarLander-v3 Demo

基于 PPO（Proximal Policy Optimization）算法的强化学习演示项目，使用 [Gymnasium](https://gymnasium.farama.org/) 的 LunarLander-v3 环境。

> 在原始 reward 基础上增加了**居中着陆惩罚**：距离着陆台中心越远，每步扣分越多。

## 效果

| 随机策略（未训练） | PPO 训练后（~100万步） |
|:---:|:---:|
| 随意乱飞，通常坠毁 | 稳定控制推进器，成功着陆 |

## 项目结构

```
.
├── train.py                      # GPU 训练脚本
├── centered_reward_wrapper.py    # 自定义 reward：居中着陆惩罚
├── plot_rewards.py               # 训练后绘制 reward 曲线
├── visualize.py                  # CPU 推理 + 录制对比动画
├── requirements.txt              # 依赖
├── models/            # 模型保存目录
├── logs/              # 训练日志（monitor.csv）
├── plots/             # reward 曲线图
└── gifs/              # 动画输出
```

## 依赖安装

```bash
# Box2D 需要 swig（Ubuntu/Debian）
sudo apt-get install swig

pip install -r requirements.txt
```

## 使用流程

### 1. 训练（GPU）

```bash
python train.py
```

- 自动检测并使用 GPU
- 训练约 10-15 分钟（100 万步）
- 目标 mean reward：150+
- 模型保存至 `models/ppo_lunarlander.zip`
- 训练日志写入 `logs/monitor.csv`

PPO 固定超参：

| 参数 | 值 |
|------|-----|
| n_steps | 2048 |
| batch_size | 64 |
| n_epochs | 10 |
| learning_rate | 2.5e-4 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.01 |
| total_timesteps | 1,000,000 |

### 2. 绘制 Reward 曲线

```bash
python plot_rewards.py
```

输出 `plots/rewards.png`，包含：
- 每集 reward（半透明）
- 100 集移动平均线（实线）
- 150 目标线（虚线）

### 3. 录制对比动画（CPU）

```bash
python visualize.py
```

输出：
- `gifs/untrained.gif` — 随机策略（坠毁表演）
- `gifs/trained.gif` — 训练后策略（稳定着陆）

使用 `render_mode="rgb_array"` 渲染，**不依赖 GUI**，WSL / headless 环境也能运行。

## 自定义 Reward 设计

标准 LunarLander 的奖励函数不关心横向位置——只要能安全着陆（两腿接触、速度低、角度直），落在着陆台左边、右边或正中间得分完全一样。

本项目通过 `CenteredLandingWrapper` 增加了**居中惩罚**：

```python
每步额外惩罚 = -0.1 * |火箭x坐标 - 着陆台中心x坐标|
```

这样策略在优先"活着着陆"的同时，会尽量往两个旗子中间落。

## 技术栈

- [Gymnasium](https://gymnasium.farama.org/) — RL 环境
- [stable-baselines3](https://stable-baselines3.readthedocs.io/) — PPO 实现
- PyTorch — 后端（GPU 训练）
- matplotlib — 可视化
- imageio — GIF 合成
