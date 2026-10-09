# RL Workbench

**中文** | [English](README.en.md)

本地强化学习项目扫描与训练工作台。项目以 MicroDuck 官方代码作为首个完整接入实例，同时提供标准 RL 项目 v1 规范、可运行模板和严格扫描器。

当前有两条使用路径：

- **MicroDuck**：网页配置奖励权重、屏蔽奖励、修改 MLP 隐藏层、启动/暂停/停止训练，查看最新仿真画面、奖励曲线与构成，下载模型和删除记录。
- **标准 v1 项目**：扫描项目结构与代码入口，通过命令行覆盖配置、执行训练和校验指标/模型产物。通用项目尚未接入 MicroDuck 的网页训练控制与实时监控。

规范检查项目格式与运行链路，不评价奖励设计、策略效果或是否收敛。未遵循规范的任意项目不会自动获得完整接入。

## 1. 工作台与 RL 项目怎么存放

工作台目录布局示例：

```text
rl-workbench/                    工作台根目录
├── rl_workbench/               工具代码
├── agent.md                    交给其他 agent 的 RL 项目规范
├── upstream/microduck_rl/      MicroDuck 官方项目及其独立环境
├── .venv/                      工作台依赖环境
├── .runtime/                   MicroDuck 目录缓存、日志、编译缓存
└── runs/microduck/             MicroDuck 训练记录与模型
```

`upstream/microduck_rl/` 是当前 MicroDuck 专用适配器使用的固定位置。其他项目不必放进 upstream；放进去也不会自动变成 MicroDuck。

**新项目推荐：将工作台作为完整子目录放进 RL 项目。**

```text
my_rl/                         你开发的 RL 项目
├── rl-project.json            你的任务与配置清单
├── pyproject.toml             你的训练依赖
├── README.md                  你的任务使用说明
├── src/my_rl/                 你的环境、奖励和训练代码
├── .venv/                     你的训练环境
├── runs/                      你的命令行训练输出
└── rl-workbench/              本工具，保持独立目录
    ├── README.md
    ├── agent.md
    ├── schemas/
    ├── rl_workbench/
    └── .venv/                 工具自身环境
```

在 `rl-workbench/` 启动服务，标准项目扫描会向父目录查找最近的 `rl-project.json`。不要把两套 src、README、依赖环境混在一个目录里。

也可让工作台和 RL 项目并列存放，或放在完全不同的位置；扫描和训练时传入目标项目路径即可。只有向父目录查找是自动的，不会自动遍历 upstream 中所有子项目。

迁移或分享工具时携带源码、脚本、requirements.txt、schemas、templates、docs 和 agent.md；不要复制旧机器的 `.venv`、`.runtime`、runs 等生成内容作为安装包。需要 MicroDuck 时在新位置运行安装脚本重新准备 upstream。

## 2. 安装与启动工作台

在工作台根目录执行。工作台使用 Python 3.11–3.13：

```bash
bash setup.sh
bash start.sh
```

已安装依赖的环境只需运行 `bash start.sh`。若已有服务运行，不要重复启动相同端口。

界面侧栏提供“语言 / Language”切换，选择会保存在当前浏览器，并在三个页面之间共用；切换不会清空训练配置。训练名称、代码标识和原始日志保留原文。

打开 [工作台](http://127.0.0.1:8765/) 或 [标准 RL 项目扫描](http://127.0.0.1:8765/projects)。服务只监听本机；前台启动需保持终端打开。

```bash
bash start.sh --port 8766 --data-dir runs-other
```

`--data-dir` 调整网页管理的记录目录。标准项目命令行训练的输出由 `--output-dir` 单独指定。

仅使用标准项目扫描/模板时，无需安装 MicroDuck。使用 MicroDuck 训练时，需要 NVIDIA CUDA 驱动、uv 和独立的 Python 3.12 环境：

```bash
bash setup_microduck.sh
bash start.sh
```

该脚本下载固定版本的 MicroDuck 到 upstream，并安装其独立训练依赖。不要直接移动已安装的虚拟环境。

## 3. 从标准模板创建自己的 RL 项目

以下命令在工作台根目录执行，目标目录必须尚不存在：

```bash
.venv/bin/python -m rl_workbench.project init /absolute/path/to/my_rl
.venv/bin/python -m rl_workbench.project check /absolute/path/to/my_rl
.venv/bin/python -m rl_workbench.project train /absolute/path/to/my_rl \
  --task CartPole-Balance-v1 --output-dir /absolute/path/to/my_rl/runs/first
```

模板是可运行的 CPU PPO / CartPole 示例。开发自己的任务时，按 [agent.md](agent.md) 修改清单、环境、奖励与训练入口；具体文件修改清单也在该文档中。`init` 只复制模板，不会自动把工具移动到新项目中。若想采用嵌套布局，先创建目标项目，再将一份工具源码放入其 `rl-workbench/` 子目录并重新安装工作台环境。

已有 RL 项目不要运行 init 覆盖它。参考模板，在现有项目添加清单与符合签名的包装函数即可。

训练命令必须使用已安装目标框架的 Python；扫描本身只使用标准库。假设工具已放在 `my_rl/rl-workbench/`，在该工具目录执行：

```bash
# 不指定路径：向父目录寻找项目清单
.venv/bin/python -m rl_workbench.project check

# 使用父目录 RL 项目的独立 Python 训练
../.venv/bin/python -m rl_workbench.project train .. \
  --task Your-Task-ID --output-dir ../runs/first
```

`Your-Task-ID` 必须换成清单中的真实 ID；`../.venv` 必须先按目标项目的 pyproject.toml 安装依赖。模块命令的当前目录仍为工作台根目录。

使用 `--config /path/to/overrides.json` 可覆盖预算、种子、网络和奖励，例如：

```json
{"total_timesteps": 128, "reward_weights": {"alive": 2.0}, "disabled_rewards": ["angle"]}
```

输出目录不能已存在。产物包括 config.json、project.json、status.json、metrics.jsonl、checkpoint.bin；status 为 completed 表示训练函数完成并通过基本产物校验，模型实际加载与预测仍须按任务 README 验证。

## 4. MicroDuck 网页使用

1. 在“环境与奖励”选择官方任务，配置环境数、迭代数、奖励权重/屏蔽和网络层宽。
2. 直接启动训练，无需先完成测试轮次。默认 64 环境 × 5 迭代只是短配置。
3. 在“训练监控”查看指标、最新画面和奖励构成。时间轴回看稀疏奖励数据；新画面只保留最新一张，不提供历史视频回放。
4. 暂停后可继续，停止会请求保存模型。Ctrl+C 关闭服务也会请求停止训练并保存；驱动故障时不能保证保存成功。
5. 点击训练记录卡片右上角的垃圾桶按钮即可删除已结束记录。确认后永久删除该条模型、日志、画面和统计；运行中、暂停中或保存未结束时禁止删除。

非零贡献频率不是任务成功率。单步奖励、官方回合日志与整体回合奖励分解的口径见 [MicroDuck 说明](docs/microduck.md)。

## 5. 源码与数据目录

- `rl_workbench/`：扫描、协议校验、训练管理、统计和网页代码。
- `schemas/`：v1 JSON Schema；`templates/`：可运行的标准项目模板。
- `tests/`：回归测试；`docs/`：接口与使用参考。
- `upstream/`：官方训练项目与依赖；`.venv/`：工作台依赖。
- `runs/`：网页训练记录；`.runtime/`：缓存和日志。
- `.archive/`：本机遗留附件，不参与运行。

备份 MicroDuck 结果时复制完整训练目录。命令行项目结果位于用户指定的 output-dir，不一定在工作台 runs 下。训练依赖和项目源码不能按缓存随意删除。

## 6. 开发验证与文档

```bash
.venv/bin/python -m unittest discover -s tests -v
node --check rl_workbench/static/microduck.js
node --check rl_workbench/static/projects.js
node --check rl_workbench/static/app.js
node --test tests/i18n.test.cjs
```

测试包含 CPU 真实训练、模型加载和本机 HTTP 接口，不需要 GPU。v1 静态校验失败会返回非零退出码并说明原因；静态通过不代表依赖、硬件和训练逻辑已经验证。

- [给 agent 的 RL 项目规范与文件修改清单](agent.md)
- [v1 字段定义](schemas/rl-project-v1.schema.json)
- [MicroDuck 高级接入接口参考](docs/workbench-integration.md)
- [MicroDuck 指标说明](docs/microduck.md)
- [CartPole 网页示例](docs/cartpole.md)

## 7. 第三方许可与独立开发声明

RewardScope（界面中亦称 RL Workbench）为独立开发项目，与 Pollen Robotics 无隶属关系，亦未获得其官方背书。MicroDuck 和 Pollen Robotics 的名称仅用于说明兼容性与上游来源，不表示本工具为其官方产品。

本工具通过安装脚本获取 [MicroDuck 官方仓库](https://github.com/pollen-robotics/microduck_rl)，当前固定版本为 `cb70b792312d559a4da09064d92009079671815f`。`upstream/` 不作为本工具源码的一部分提交。

- MicroDuck 代码采用 [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)。如复制或分发上游代码，应遵守其许可证，保留适用的版权、归属与 NOTICE 声明，并标明文件修改。
- 该固定版本的 [上游 README](https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/README.md#license) 另将 3D 模型文件标为 **Creative Commons BY-SA-NC**，并未在该说明中给出版本号。模型资源不能一概视为 Apache-2.0；使用、修改或分发前应核对相应资源的完整许可，商业用途需确认授权。
- 其他依赖和资源保留各自的许可。本项目自身的许可（包括任何非商业限制）不替代或改变第三方许可，也不授予第三方商标使用权。
