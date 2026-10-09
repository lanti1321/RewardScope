# CartPole 示例

[返回项目说明](../README.md)

这是保留的 CPU 入门示例与回归验证环境，入口为 `/cartpole`。

## 启动

当前工作区已经安装依赖：

```bash
cd /path/to/rl-workbench
bash start.sh
```

打开 <http://127.0.0.1:8765/cartpole>。Ctrl+C 会请求停止训练并保存模型。服务仅监听本机，不支持直接通过局域网共享同一服务。朋友可复制项目后在自己的电脑运行。

新机器推荐 Python 3.11–3.13（已在 Python 3.13 测试）：

```bash
bash setup.sh
bash start.sh
```

安装脚本复用系统已安装的 PyTorch，缺失时由 pip 解析安装。无需 GPU；首次安装 PyTorch 可能较大。可用 `bash start.sh --port 8766 --data-dir runs-other` 指定端口和数据目录。Windows 可手动创建虚拟环境、安装 requirements.txt，然后运行 `python -m rl_workbench.server`。

## 第一次使用

1. 设置名称、训练步数和随机种子，保留默认奖励权重，点击「开始训练」。
2. 观察训练回合奖励、固定种子的评估存活步数以及最近一次 PPO 更新指标。
3. 切换「评估回放」，选择训练节点和评估种子，播放或拖动进度条，核对画面、动作、单步奖励及奖励分项。
4. 点击「复制配置」，修改奖励权重。在回放模式下可先对已有轨迹试算新分数；开始训练会创建全新的模型和实验。
5. 选择对比实验，使用固定尺度的「评估存活步数」比较效果。可以导出实验摘要 JSON、最终模型和最佳评估模型。

同时只运行一个实验。暂停和继续仅适用于当前进程内的训练；停止或服务重启后不能从界面续训。复制配置会从头训练。刷新浏览器不影响后台训练。

## 数据含义

- **训练采样**：界面每秒显示最近一次环境转移，不逐帧直播全部训练步骤。画面由真实状态绘制，是二维示意图。图中动作造成的是当前显示的动作后状态。
- **训练曲线**：每个完整训练回合的加权奖励，以及最近最多 20 回合均值。训练采用随机策略动作；评估使用确定性动作，两者表现可能不同。
- **评估回放**：开始训练前、每 2,048 个训练步骤完成优化后、正常结束时，分别使用独立环境和固定种子 1000/1001/1002 评估。保存三个回合的完整转移，含动作前后观测、奖励和终止信息。停止时不强制新增评估。
- **评估指标**：三个种子的平均存活步数，上限 500。环境原始奖励每步为 1，因此不受自定义奖励权重影响。三个固定种子仅用于日常调试，不代表广泛泛化能力。
- **奖励试算**：对当前选中的完整评估回合重新加权，不重新模拟状态、不重新训练，也不预测新权重下未来策略的表现。
- **训练速度**：训练步数除以墙钟用时，包含初始化、优化、评估和暂停。
- **最佳模型**：按已完成评估的平均存活步数选择；并列保留较早的模型。

### 奖励公式

```text
r = w_alive × raw_reward
  − w_angle × (theta / theta_threshold)²
  − w_position × (x / x_threshold)²
  − w_failure × terminated
```

`theta_threshold` 使用环境的精确阈值（12°）；`x_threshold` 为 2.4m。特征由动作后的状态计算，不做裁剪。原始奖励在失败步也是 +1，沿用 Gymnasium 默认语义。500 步时间上限属于 `truncated`，不算失败惩罚。奖励分项之和等于环境返回给算法的单步奖励；不启用奖励归一化或奖励裁剪。PPO 内部仍有优势归一化、价值估计和时间截断 bootstrap，这些不是界面所显示的奖励项。

PPO 使用 MlpPolicy、单环境、CPU、n_steps=256、batch_size=64、n_epochs=10；其余使用当前固定 SB3 版本默认参数。每次新训练保存主要参数和运行库版本。模型保存时包含 SB3 的模型配置与优化器状态。

## 保存位置

```text
runs/<实验 ID>/
  run.json          # 配置、状态、版本、完整回合摘要、评估摘要、策略更新指标
  eval-0.json       # 初始策略在三个评估种子上的完整轨迹
  eval-1.json       # 后续评估；含 before / after / action / features / components
  ...
  best_model.zip    # 已评估节点中表现最好的策略
  model.zip         # 正常完成或主动停止时保存的策略
```

界面「导出实验 JSON」只导出 `run.json`；完整备份请复制整个实验目录。训练期间每秒及每次评估后原子保存摘要。意外终止后，已写入的记录可恢复；重启将未结束的实验标记为「已中断」，不声称自动续训。训练途中最佳模型也会保留。

加载已保存的模型（仅加载可信模型文件）：

```python
from stable_baselines3 import PPO
from rl_workbench.envs import RewardCartPole

model = PPO.load("runs/<实验 ID>/model.zip", device="cpu")
env = RewardCartPole()  # 评估加权分数时，应传入该实验 run.json 中的 weights
obs, _ = env.reset(seed=1000)
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))
    if terminated or truncated:
        break
env.close()
```

## 验证

```bash
.venv/bin/python -m unittest discover -s tests -v
node --check rl_workbench/static/app.js
```

测试包含真实 PPO 短训练、保存模型后复现完整评估轨迹、奖励分解、失败与时间上限区别、暂停/继续/停止、配置验证和 HTTP 来源检查。

## 代码入口与当前边界

- `rl_workbench/envs.py`：环境适配和奖励分解。
- `rl_workbench/training.py`：训练生命周期、采样、评估、模型与记录保存。
- `rl_workbench/server.py`：本地 HTTP API。
- `rl_workbench/static/`：无外部依赖的界面、绘图与交互。

CartPole 页面仅支持该入门任务；机器人 / MuJoCo 训练请使用上面的 MicroDuck 页面。通用环境插件、断点续训界面尚未实现。接入其他环境需要定义观测/动作含义、奖励分项、适合该任务的评价指标和画面渲染方式。工具不自动识别奖励漏洞，也不会把奖励构成当作策略行为的因果解释。
