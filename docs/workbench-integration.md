# MicroDuck 工作台接入接口参考

本文描述 MicroDuck 专用适配器。新项目请先使用根目录 agent.md 的 v1 规范、严格扫描器和命令行 runner；通用项目的高级网页监控仍需接入本文接口。

这是一份项目接入说明，不是通用 RL 标准。当前运行时适配的是 MicroDuck / mjlab / RSL-RL；把本项目复制到任意 RL 仓库后，并不会自动完成训练接入。静态扫描只能发现候选，运行时仍需适配训练入口、配置和奖励数据。

## 1. 先运行现有范本

需要 Linux、可用的 NVIDIA CUDA 驱动、Git、uv。工作台 Python 使用 3.11–3.13；上游使用独立的 Python 3.12 环境。安装和启动均在本项目根目录执行：

```bash
bash setup.sh
bash setup_microduck.sh
bash start.sh
```

访问 http://127.0.0.1:8765/ 。服务只监听本机。前台启动时关闭终端会停止服务；运行中关闭服务会请求停止并保存模型。

上游固定为 `pollen-robotics/microduck_rl` 提交 `cb70b792312d559a4da09064d92009079671815f`。安装脚本使用 `uv sync --frozen`，不要擅自升级上游依赖或改锁文件。换版本后必须重新验证运行时 API。

GPU 检查应包含实际运算，不能只看 nvidia-smi：

```bash
upstream/microduck_rl/.venv/bin/python -c "import torch; print(torch.ones(8, device='cuda').sum().item())"
```

新建训练，选择任务，设置迭代次数、并行环境、奖励数值和 MLP 隐藏层，再启动。默认 64 环境 × 5 轮是短配置，并非必须先跑的测试。支持暂停、继续、停止并保存。修改配置创建新训练，不自动继承原模型。

点击训练记录卡片右上角的垃圾桶按钮可删除已结束记录。确认后永久删除该记录整个目录（模型、日志、图片和统计）；需要的模型先下载。运行中、暂停中以及进程尚未退出时均禁止删除。当前该功能用于 MicroDuck 记录，CartPole 示例未提供删除入口。

## 2. 代码入口与责任

- `rl_workbench/discovery.py`：AST 静态扫描，不执行被扫描代码；它不是运行时适配器。
- `rl_workbench/microduck.py`：父进程配置验证、创建训练目录、启动独立子进程、暂停/继续/停止/删除。
- `rl_workbench/microduck_bridge.py`：在上游虚拟环境中读取任务配置、调用官方训练入口、采集奖励与日志、保存模型。
- `rl_workbench/reward_settings.py`：固定权重、屏蔽与旧倍率配置的优先级。
- `rl_workbench/diagnostics.py`：通用 PyTorch 奖励频率和回合分解统计，不依赖具体机器人。
- `rl_workbench/live_preview.py`：原子覆盖单份最新画面。
- `rl_workbench/server.py`：本机 HTTP 接口、来源检查、静态文件。
- `rl_workbench/static/microduck.{html,js,css}`：主工作台界面。

网络激活和探索图已移除，不应重新挂接激活 hooks。网络隐藏层配置功能仍保留。

## 3. 接入其他 RL 项目的步骤

1. 找到官方训练入口、环境创建入口、算法配置、奖励管理器、模型保存/加载入口。
2. 确认每轮采集多少环境步，以及向量环境的自动重置时机；区分正常时间截断和任务失败。
3. 在目标项目的独立 Python 环境运行桥接代码，保持原算法与仿真逻辑。以 `microduck_bridge.py` 为范本，不把新框架依赖装入工作台进程。
4. 生成下面的任务目录结构，填入真实算法配置。无法识别的功能应标为不支持，不能用猜测值冒充可用配置。
5. 实现配置映射与下述奖励数据契约。若只能读日志，先提供日志曲线，不伪造逐步分项、非零频率或精确回合分解。
6. 新增对应 manager/bridge 并接入 server；当前没有可插拔 adapter 注册接口，`microduck.py` 的路径常量和前端 MicroDuck 任务说明也需要针对新适配器调整。
7. 完成第 8 节验证，再开放给其他人使用。

扫描其他项目可先运行：

```bash
.venv/bin/python -m rl_workbench.discovery /path/to/rl-project
```

## 4. 任务目录和训练配置

`.runtime/microduck_catalog.json` 的顶层包括 `ready`、`source`、`gpu`、`default_task`、`tasks` 和可选扫描结果 `discovery`。`source` 保存仓库 URL、提交号、是否有改动、Python 与依赖版本。

每个 task 需要这些字段：

- `id`：官方注册任务 ID。
- `episode_seconds`、`step_dt`、`scale_by_dt`、`steps_per_iteration`：回合时长上限、控制步长、奖励是否乘 dt、每轮每环境采集步数。
- `default_iterations`、`default_num_envs`：官方默认预算；界面短训练默认值可以与它们不同。
- `rewards`：`[{name, weight, function, source}]`，其中 source 包括文件、行号、文档字符串，无法定位时可为 null。
- `network`：actor/critic 各包含 `class_name`、`hidden_dims`、`activation`。当前只允许编辑已识别的 `MLPModel`。
- `curricula`、`terminations`、`observation_groups`：名称数组。

提交训练的 JSON 示例：

```json
{
  "name": "速度跟踪训练",
  "task": "Mjlab-Velocity-Flat-MicroDuck",
  "num_envs": 64,
  "iterations": 2000,
  "seed": 42,
  "network": {"actor": [512, 256, 128], "critic": [512, 256, 128]},
  "reward_weights": {"track_linear_velocity": 2.0},
  "disabled_rewards": ["air_time"]
}
```

当前后端预算范围：环境数 16–4096、轮数 1–100000、seed 0–2147483647；前端提供 64/256/512/1024/2048/4096 个环境选项。隐藏层 1–6 层，每层 8–1024。权重须为有限数值且绝对值不超过 1000000。支持范围不等于显存能承受最大配置。

`reward_weights` 只填写需要固定覆盖的项。未填写项跟随官方课程变化；`disabled_rewards` 优先级最高，始终将对应权重置零。旧 `reward_scales` 仅用于兼容已保存配置，不建议新接入继续使用。不要编辑奖励触发条件或奖励函数源码。

## 5. 奖励数据契约（接入最重要的部分）

对于 N 个并行环境、K 个奖励项，采集最终加权贡献张量 `[N, K]`。贡献应与环境实际返回的奖励保持相同尺度：如果训练使用 `weight × raw × dt`，采集的也必须是这个值。

- MicroDuck 的 `_step_reward` 尚未包含 dt，桥接层按配置乘 dt；其他框架必须重新核对，不能照搬后重复乘 dt。
- 单环境单步分项之和应等于该步奖励。当前桥接层在采样时检查残差，容差为 `1e-4 × max(1, abs(total))`。
- 奖励列顺序必须与 `active_terms` 一致。环境维度与 done 标记必须对应同一批环境。
- 在自动重置清空缓冲前读取本步分项；done 回合的最后一步不能漏算或计入下一个回合。
- 存在额外奖励、包装器裁剪/缩放、内在奖励时，需要明确记录差额，不能假装基础分项必然等于学习器接收到的总奖励。

`RewardProbe.update(contributions, weights)` 的 weights 需要是与 contributions 同尺度的有效乘数（包含 dt）。它按所有环境的所有调用累计统计：

- `samples`：累计交互数，N × 调用次数。
- `nonzero_count`：贡献绝对值大于 1e-8 的次数。
- `frequency`：nonzero_count / samples，不是成功率或条件触发率。
- `evaluated_samples`：权重不为零时的交互数，不代表条件满足次数。
- `raw_mean_when_enabled`：启用期间去掉有效乘数后的平均原始评分；权重为零时不能恢复原始值。

`EpisodeRewardProbe` 累计每个环境的分项回报，并在 done 时按官方 logger 相同顺序加入最近最多 100 个回合窗口；重置相应环境累计值。只有窗口、顺序和奖励尺度一致，才能把它用作顶部平均回合奖励的分解。

三个不同口径不要混用：

1. 单步贡献：某个采样步的环境 0 分项，可加和为该步总奖励。
2. 官方 `Episode_Reward/*`：MicroDuck 在 reset 时将结束回合的分项累计均值除以回合时长上限，再由 logger 按迭代聚合；不是普通回合总分，也不保证采用最近 100 个回合的同一窗口。
3. 整体奖励构成：与顶部 `Train/mean_reward` 同批回合的平均累计分项。百分比为分项除以净总分，可为负或超过 100%；总分接近零时隐藏百分比。

## 6. 记录与预览格式

默认每条训练独立放在 `runs/microduck/<12位小写十六进制ID>/`：

```text
run.json          当前状态、配置、日志历史、奖励采样
control.json      {"action":"resume"}，也可为 pause 或 stop
train.log         官方进程 stdout/stderr
latest.json       最新预览，原子覆盖，不累积图片
frames/           兼容旧记录的图片目录；新训练不写历史 JPEG
official/         原生 checkpoint、ONNX、参数 YAML、TensorBoard event
```

`run.json` 必备字段：`id`、`created_at`（带时区 ISO 时间）、`config`、`status`、`iteration`、`step`、`elapsed`、`phase`、`error`、`history`、`metrics`、`frames`、`source`、`checkpoints`、`onnx`。

- `iteration`：完成的策略更新数，界面从 1 计数。官方 checkpoint 可从 0 编号，因此 500 轮最终文件可为 `model_499.pt`。
- `step`：已完成轮数 × 每轮每环境采样步数 × 环境数，不含正在采集的未完成轮次。
- `history`：`[{iteration, step, metrics}]`。metrics 使用官方标量名，例如 `Train/mean_reward`、`Train/mean_episode_length`、`Loss/value`、`Loss/surrogate`、`Perf/total_fps`。
- `frames`：稀疏奖励数据，包含 `index/env_step/env_id/episode_step/episode_id/reward/components/weights/sum_residual/step_dt/qpos/action/terminated/truncated/reward_stats/reward_batch_mean`。新训练没有 activations。
- `reward_breakdown`：包含 `episodes/window/components/component_total/logged_total/residual`。尚无结束回合时可为 null。
- `live_preview: true`：使用新的最新画面通道；旧记录无此字段时仍可回放旧图片。

`latest.json` 格式：

```json
{"env_step": 120, "captured_at": 1791511200.0, "image": "data:image/jpeg;base64,..."}
```

`captured_at` 为 Unix 秒。当前只渲染环境 0、640×400、最高 2 帧/秒；先写临时文件，再替换正式文件。浏览器每约 0.5 秒单请求轮询，隐藏页面不请求；训练端仍采集预览。优化、暂停期间可能没有新图，不能把旧图标作当前时刻。

最新画面与历史奖励时间轴独立。稀疏数据步距为 `max(4, ceil(iterations × steps_per_iteration / 400))`。只有预览图片存储保持固定数量；指标 history 仍随轮数增长，不宣称整个训练内存恒定。

## 7. 生命周期、保存与 HTTP

状态包括 starting、running、paused、stopped、completed、failed、interrupted。客户端还兼容 pausing/stopping。控制命令在安全边界读取；暂停后不推进训练，继续后恢复；停止应保存检查点后再退出。不要通过强杀实现正常停止。

停止保存使用 `.model_stopped.pending.pt` 临时文件，成功后替换 `model_stopped.pt`；失败清理临时文件并记录堆栈，避免发布残缺模型。已有定期检查点应保留。checkpoint 能恢复模型/优化器及实现保存的计数信息，不意味着完整仿真状态逐位重现。

主要接口（本机、同源 JSON）：

- `GET /api/microduck/info`：任务目录。
- `POST /api/microduck/scan`：重新读取项目配置。
- `GET/POST /api/microduck/runs`：列表/创建训练。
- `GET /api/microduck/runs/{id}`：完整记录。
- `POST /api/microduck/runs/{id}/{pause|resume|stop}`：控制，JSON 请求体 `{}`。
- `POST /api/microduck/runs/{id}/delete`：删除已结束记录，返回 `{"deleted":"id"}`。
- `GET /api/microduck/runs/{id}/latest`：最新图像 JSON，尚未生成时 404。
- `GET /api/microduck/runs/{id}/log`：日志末尾。
- `GET /api/microduck/runs/{id}/export`：导出完整记录。
- `GET /api/microduck/runs/{id}/download/{name}`：下载记录中列出的模型。
- `GET /api/microduck/runs/{id}/frame/{index}`：旧记录图片。

错误：400 配置无效，403 Host/Origin 不允许，404 记录不存在，409 状态冲突，415 请求不是 JSON。不要把服务直接暴露公网。删除须检查 ID、路径和进程状态；不能接受任意文件路径。

## 8. 接入验收与已知边界

```bash
.venv/bin/python -m unittest discover -s tests -v
node --check rl_workbench/static/microduck.js
node --check rl_workbench/static/app.js
```

新适配器至少检查：正常训练与结束、暂停不推进、继续推进、停止保存后加载、异常保存不发布残缺文件、权重/屏蔽与课程冲突、分项求和与回合窗口、预览覆盖、运行中禁止删除、删除只影响目标记录、刷新/重启后历史可读。测试使用临时目录，不清理用户历史数据。

2026-10-09 本次验收：18 项自动测试通过；64 环境、128/64 隐藏层的 5 轮 GPU 训练正常完成；另一轮实际验证暂停、继续、停止保存、checkpoint 读取、删除阻止与删除成功；奖励固定权重和屏蔽结果正确，回合分解残差小于 1e-4，新记录未采集激活，最新预览未累计历史 JPEG。

此前在同机用 16 环境、小网络完整运行 500 轮并读取最终模型，也验证过停止 checkpoint 经官方 CUDA runner 加载后继续训练两轮。该结果不等于当前版本所有任务、大网络、2048/4096 环境或睡眠恢复均已验证。

已知边界：仅 MicroDuck 运行时适配；只编辑 MLP 层宽与奖励数值；没有网页断点续训入口；没有激活/探索分析；未做长时间压力测试。GPU 驱动若报告 `Node Reboot Required`，需要按驱动建议恢复，应用不能保证修复驱动故障。训练中避免系统睡眠。
