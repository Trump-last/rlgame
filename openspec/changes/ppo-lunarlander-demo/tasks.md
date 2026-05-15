## 1. 项目结构与依赖

- [x] 1.1 创建 requirements.txt，声明 gymnasium[box2d]、stable-baselines3、matplotlib、imageio、numpy 依赖
- [x] 1.2 创建目录结构：models/、logs/、plots/、gifs/

## 2. PPO 训练脚本

- [x] 2.1 创建 train.py，使用 Gymnasium 创建 LunarLander-v2 环境
- [x] 2.2 包装 Monitor wrapper，设置日志输出到 logs/monitor.csv
- [x] 2.3 配置 PPO 模型，固定超参数（n_steps=2048, batch_size=64 等）
- [x] 2.4 执行训练（total_timesteps=1_000_000），GPU 自动运行
- [x] 2.5 训练完成后保存模型到 models/ppo_lunarlander.zip

## 3. Reward 可视化

- [x] 3.1 创建 plot_rewards.py，读取 logs/monitor.csv
- [x] 3.2 绘制 episode reward 折线图 + 100-episode 移动平均线
- [x] 3.3 保存图表到 plots/rewards.png

## 4. 模型推理与动画录制

- [x] 4.1 创建 visualize.py，初始化 LunarLander-v2（render_mode="rgb_array"）
- [x] 4.2 实现 record_episode 函数：给定策略，运行 1 个 episode，收集帧序列
- [x] 4.3 录制随机策略动画，保存为 gifs/untrained.gif
- [x] 4.4 加载 models/ppo_lunarlander.zip（device="cpu"），录制训练后策略动画，保存为 gifs/trained.gif
