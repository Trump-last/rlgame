## Context

本项目是一个独立的 PPO 强化学习演示，从零开始构建。使用 Gymnasium 的 LunarLander-v2 作为任务环境，stable-baselines3 作为 PPO 训练框架。

## Goals / Non-Goals

**Goals:**
- 训练一个 PPO 智能体，在 LunarLander-v2 上达到 mean reward 150+
- 训练完成后可绘制 reward 曲线
- 支持模型保存与加载
- 提供随机策略 vs 训练后策略的对比动画（GIF）

**Non-Goals:**
- 手写 PPO 算法（使用 SB3 封装）
- 支持多环境并行训练（SB3 内部已处理）
- 超参数调优接口（参数固定）
- 实时训练可视化（训练后统一绘图）

## Decisions

**1. 训练框架：stable-baselines3 而非手写 PPO**
- 理由：demo 导向，SB3 一行调用即可训练，减少 boilerplate，专注环境交互和可视化
- 替代方案：手写 PPO（~300 行），但学习价值对 "demo 就行" 的目标过高

**2. 环境：Gymnasium 而非 gym**
- 理由：gym 已停止维护，Gymnasium 是活跃 fork，API 兼容
- LunarLander-v2 提供足够的复杂度（8维状态，4个离散动作），训练效果直观

**3. Reward 记录：SB3 Monitor wrapper + 后处理绘图**
- 理由：Monitor 自动记录每个 episode 的 reward/length/timestep，训练后统一用 matplotlib 画图
- 替代方案：自定义 callback 实时绘图，但用户明确偏好 "更懒的方案"

**4. 动画录制：rgb_array 模式 + imageio**
- 理由：不依赖 GUI（X11），可在 headless WSL 环境运行，输出标准 GIF 文件
- 替代方案：render_mode="human" 弹出窗口，但 WSL 无图形支持时会失败

**5. 推理在 CPU 执行**
- 理由：推理不需要 GPU，且用户要求在 CPU 机器上运行对比动画
- 通过 `device="cpu"` 加载模型即可

## Risks / Trade-offs

- [训练时间长] LunarLander-v2 到 150+ reward 可能需要 50-100 万步，在 GPU 上约 10-15 分钟 → 可接受，用户已明确目标
- [Box2D 安装问题] `gymnasium[box2d]` 依赖 Box2D，某些环境可能需要额外系统库 → 已在 requirements 中声明，安装失败时提示 `swig`
- [GIF 文件较大] 每个 episode 最长 1000 帧，30fps GIF 约几秒 → 可接受
