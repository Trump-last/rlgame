## Purpose

训练 reward 数据的后处理与可视化。

## Requirements

### Requirement: Reward 曲线绘制
系统 SHALL 读取 logs/monitor.csv，用 matplotlib 绘制 episode reward 随时间变化的曲线，并保存为 plots/rewards.png。

#### Scenario: 训练后绘制
- **WHEN** plot_rewards.py 运行且 logs/monitor.csv 存在
- **THEN** 生成 plots/rewards.png，包含 episode reward 折线图及移动平均线

### Requirement: 移动平均线
系统 SHALL 在 reward 曲线上叠加 100-episode 移动平均线，帮助观察训练趋势。

#### Scenario: 趋势可视化
- **WHEN** reward 曲线图生成
- **THEN** 图中同时显示原始 reward（半透明）和 100-episode 移动平均（实线）
