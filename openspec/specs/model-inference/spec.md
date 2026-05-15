## Purpose

已训练模型的加载、推理与游戏动画录制。

## Requirements

### Requirement: 随机策略动画录制
系统 SHALL 录制随机动作策略在 LunarLander-v2 上运行 1 个 episode 的动画，保存为 gifs/untrained.gif。

#### Scenario: 未训练模型表现
- **WHEN** visualize.py 运行且使用随机动作
- **THEN** 生成 gifs/untrained.gif，展示随机策略的着陆过程（通常失败）

### Requirement: 训练后策略动画录制
系统 SHALL 加载 models/ppo_lunarlander.zip，录制训练后模型在 CPU 上运行 1 个 episode 的动画，保存为 gifs/trained.gif。

#### Scenario: 训练后模型表现
- **WHEN** visualize.py 运行且模型文件存在
- **THEN** 加载模型并在 CPU 上推理，生成 gifs/trained.gif，展示训练后策略的着陆过程

### Requirement: 渲染模式
系统 SHALL 使用 render_mode="rgb_array" 收集帧，通过 imageio 合成 GIF，不依赖 GUI 窗口。

#### Scenario: Headless 环境运行
- **WHEN** 在无图形界面的环境（如 WSL）中运行 visualize.py
- **THEN** 动画正常生成，不报错或弹出窗口
