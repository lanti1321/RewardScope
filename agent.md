# RL 项目规范 v1：给开发者与代码 agent

## 目标和交付判定

请按本文件创建 RL 项目：结构和代码入口可由 RL Workbench 精确扫描，训练可通过统一命令执行，真实奖励指标及模型产物可验证。开发者负责任务和奖励设计，规范不评价收敛、策略能力或奖励是否合理。

**合规依据是机器校验与真实训练结果，不是 agent 的文字声明。** 必须同时满足本文件、`schemas/rl-project-v1.schema.json` 和运行验收要求。不允许擅自改字段、换入口签名或跳过验证后声称完成。

参考范式来自 MicroDuck：任务注册、环境配置、独立奖励定义、算法/网络配置和训练器入口各有明确职责。v1 将这些信息收敛到一个机器可读清单。MicroDuck 原有适配器仍独立保留，无需改写上游代码。

本规范和模板随 RL Workbench 仓库提供：

- `schemas/rl-project-v1.schema.json`：字段定义与约束，唯一的结构校验依据。
- `templates/rl_project/`：可直接训练的 CPU PPO / CartPole 模板。
- `rl_workbench/project_contract.py`：静态验证器与配置解析器。
- `rl_workbench/project.py`：创建、检查与训练命令。

当前 v1 已支持**通用项目扫描、配置覆盖和命令行训练及产物检查**。通用项目的网页启动、暂停、实时预览尚未接入 MicroDuck 监控页面；不要将扫描通过描述成已具备这些扩展功能。已有 MicroDuck 高级采集接口另见 `docs/workbench-integration.md`。

## 1. 从模板创建项目

在 RL Workbench 根目录执行（路径换成真实位置）：

```bash
.venv/bin/python -m rl_workbench.project init /absolute/path/to/my_rl
.venv/bin/python -m rl_workbench.project check /absolute/path/to/my_rl
.venv/bin/python -m rl_workbench.project train /absolute/path/to/my_rl \
  --task CartPole-Balance-v1 --output-dir /absolute/path/to/new-run
```

`init` 复制模板，不覆盖已有目录；`check` 只静态读取文件，不导入项目、不初始化仿真、不占用 GPU；`train` 明确执行项目代码，必须使用装好目标项目依赖的 Python。最后一条不是静态扫描。

工作台现有 Python 环境已能运行模板。其他框架应使用目标项目的独立环境，在工作台根目录用该环境的 Python 执行同样命令，例如 `/path/to/my_rl/.venv/bin/python -m rl_workbench.project ...`。工作台模块本身须从当前目录可导入。

将 RL Workbench 文件夹放入目标 RL 项目根目录后，从工作台文件夹执行无路径的 `python -m rl_workbench.project check`，会从当前目录向父目录查找最近的 `rl-project.json`。网页“标准 RL 项目扫描”也默认查找服务启动目录的最近项目根目录。不得要求开发者手工填写奖励函数位置、网络层名等清单已有信息。

## 1.1 工作台和目标 RL 项目的存放约定

本文件中的“工作台根目录”指包含 `rl_workbench/`、`schemas/`、`templates/` 和本文件的目录；“RL 项目根目录”指包含 `rl-project.json`、`pyproject.toml`、任务代码的目录。agent 必须区分两者，不得把工作台的 README、配置或训练依赖覆盖成目标项目的文件。

推荐将完整工作台源码作为目标项目的一个子目录：

```text
my_rl/                              目标项目根目录
├── rl-project.json                 目标项目清单
├── pyproject.toml                   目标项目依赖
├── README.md                        目标项目运行与验收说明
├── src/my_rl/                       目标项目代码
├── .venv/                          目标训练环境
├── runs/                           目标训练产物
└── rl-workbench/                    工作台根目录
    ├── agent.md                     本规范
    ├── rl_workbench/                工具源码
    ├── schemas/                    工具协议定义
    ├── templates/                  创建项目的参考模板
    └── .venv/                      工具运行环境
```

在 `my_rl/rl-workbench/` 下执行 `.venv/bin/python -m rl_workbench.project check` 会发现父目录清单；`bash start.sh` 启动的网页扫描也默认定位父目录。扫描只向上找最近清单，不向下递归找任务仓库。

工作台与 RL 项目也可以并列或分别存放。此时命令显式传目标根目录，网页选择目标根目录。把项目放进 `upstream/` 本身不是注册操作；当前 `upstream/microduck_rl/` 仅是 MicroDuck 专用适配器的固定路径。

`init` 的目标必须不存在。新建嵌套布局时，先在现有工作台中 init 新项目，然后再复制工具源码到新项目子目录并安装其环境；不能先创建含工作台的目标目录，再用 init 覆盖它。已有项目直接添加规范文件与包装入口，不运行 init。

复制工具时保留源码、schemas、templates、文档、requirements.txt 和脚本，不携带旧 `.venv`、`.runtime`、runs 或已安装的 upstream 环境。训练依赖在目标项目环境中安装。目标与工具的 `.venv` 互相独立；从工作台目录用 `../.venv/bin/python -m rl_workbench.project train .. ...` 可调用目标 Python。不要修改 sys.path 到某位开发者机器上的硬编码路径。

## 1.2 开发 RL 项目时必须修改哪些文件

下列路径均以**目标 RL 项目根目录**为基准。模板使用 `src/example_rl/`；改名为自己的包后必须同步所有 import 和清单引用。

1. **`rl-project.json`：任务登记与可编辑默认值。** 修改项目名、任务 ID、算法/框架/仿真器说明、环境与训练函数引用、控制步长、奖励缩放、观测/终止/课程名称。添加每项奖励的函数位置、默认权重、启用状态和说明，设置环境数、预算、种子及 MLP 默认值。清单必须与真实代码一致。
2. **`src/<包名>/tasks/<任务>/env_cfg.py`：环境实现。** 实现 `make_env(config)`，配置机器人/场景、观测、动作、重置和终止，接入解析后的奖励及时间缩放。模板中的 CartPole 环境须替换为实际任务环境；不能只把清单改成走路或抓取而继续训练 CartPole。
3. **`src/<包名>/tasks/<任务>/rewards.py`：奖励公式。** 实现清单列出的 `函数名(transition)`，返回有限原始标量，说明输入和单位。新增/删除/改名奖励必须同时更新清单和环境中所需的 transition 字段。权重由 config 提供，不重复硬编码在函数里。
4. **`src/<包名>/train.py`：框架桥接与采集。** 实现 `train(config, output_dir)`，实际使用预算、并行数、种子、网络配置和奖励覆盖，调用正确的环境工厂和算法。模板当前直接导入 balance.make_env；换任务必须同步修改导入，或显式按 task 分派。采集真实指标、写标准产物、释放资源，不能只保留空入口。
5. **`pyproject.toml` 与依赖锁/环境清单：运行环境。** 修改 Python 约束、包名、打包路径和框架/仿真依赖。在目标环境验证安装，不能依赖工作台恰好安装了某个库。
6. **`README.md`：该 RL 项目的可执行说明。** 填写真实任务 ID、安装命令、从工作台目录执行的检查/训练命令、框架加载 checkpoint.bin 的命令，以及实际验收结果。不要把工作台 README 原样复制充当任务说明。
7. **按需新增资源和任务测试。** 机器人模型、场景、观测/终止/课程模块放在目标项目中；记录资源来源。测试配置独立性、奖励求和/屏蔽、模型加载及有效预测。不存在的能力不要创建占位实现冒充可用。

只修改权重、预算或受支持的 MLP 默认层宽时，修改清单 defaults/rewards 即可；单次实验可使用覆盖 JSON，无需改函数源码。改变奖励计算公式时修改 rewards.py；增加奖励项还要更新清单。改变模型类型超出 v1 的 MLP 能力时应明确报告不支持，不能改工具校验器放行后声称 v1 合规。

同一项目有多个任务时，为每个任务添加独立环境/奖励模块和清单条目；共享 train 入口须根据 config.task 正确分派，或分别提供训练函数。不要让所有任务 ID 都静默运行同一个模板环境。

## 1.3 哪些工具文件不需要修改

标准 v1 扫描与命令行训练通过清单定位项目。开发者不需要逐项目修改工作台的 `project_contract.py`、`project.py`、`server.py`、schemas 或前端；也不要在 `templates/rl_project/` 内直接开发目标项目，以免后续 init 复制了个人任务代码。

如果目标是额外接入网页启动/暂停、逐步统计、实时画面，当前仍需开发高级适配器；这属于工具扩展，不属于标准 v1 项目必须改的文件。具体参考 `docs/workbench-integration.md`，不得把 MicroDuck 路径常量简单改成新路径就宣称完成高级接入。

## 2. 必须存在的项目结构

```text
my_rl/
├── rl-project.json                  必须：v1 清单，项目唯一配置来源
├── pyproject.toml                   必须：Python 版本及安装依赖
├── README.md                        安装、训练和模型读取命令
├── src/                             清单 source_root 指向此目录
│   └── my_rl/
│       ├── __init__.py
│       ├── train.py                 train(config, output_dir)
│       └── tasks/
│           ├── __init__.py
│           └── balance/
│               ├── __init__.py
│               ├── env_cfg.py       make_env(config)
│               └── rewards.py       每项 reward(transition)
└── runs/                            生成数据，不提交版本库
```

包名和任务目录名可改，清单中引用必须同步修改。`source_root` 必须是项目内相对路径；代码引用必须定位到其下的 `.py` 文件。机器人资源、观测、终止和课程模块可按任务补充。算法实现复用现有框架，不要求重写 PPO。

依赖应固定经过测试的版本并保存依赖锁或精确环境清单。除清单加载外，普通模块导入不得启动训练或创建仿真环境。配置不可使用跨训练共享的可变对象。

## 3. 清单精确格式

文件名固定为 `rl-project.json`，UTF-8 JSON；禁止重复键、NaN、Infinity、未知字段。以下示例对应随附模板：

```json
{
  "schema_version": 1,
  "project": "example-rl",
  "source_root": "src",
  "tasks": [{
    "id": "CartPole-Balance-v1",
    "algorithm": "PPO",
    "framework": "stable-baselines3",
    "simulator": "Gymnasium CartPole-v1",
    "environment": "example_rl.tasks.balance.env_cfg:make_env",
    "train": "example_rl.train:train",
    "step_dt": 0.02,
    "reward_scale": "none",
    "observation_groups": ["actor", "critic"],
    "terminations": ["failure", "time_limit"],
    "curricula": [],
    "rewards": [
      {"name": "alive", "function": "example_rl.tasks.balance.rewards:alive", "weight": 1.0, "enabled": true, "description": "环境存活分数"},
      {"name": "angle", "function": "example_rl.tasks.balance.rewards:angle", "weight": -0.1, "enabled": true, "description": "杆角度弧度值的平方"}
    ],
    "defaults": {
      "num_envs": 2,
      "total_timesteps": 256,
      "seed": 42,
      "network": {"type": "mlp", "actor": [32, 32], "critic": [32, 32], "activation": "tanh"}
    }
  }]
}
```

所有示例字段均必填。任务和奖励各自名称不能重复；没有课程时使用空数组。任务至少一个，奖励至少一项。具体字段类型与取值以随附 JSON Schema 为准。

`step_dt` 是控制步长（秒），必须正数。`reward_scale` 只能为 `none` 或 `dt`。`num_envs`、`total_timesteps` 为正整数，seed 为非负整数；布尔值不是整数。预算统一为所有环境累计交互数。PPO 可向上取整至完整 rollout，必须报告真实交互数，不称为更新轮数。

v1 网络仅支持 MLP，actor/critic 为非空正整数层宽列表，激活为 tanh/relu。其他模型须后续扩展协议版本，不能塞入未经验证的字段。其他算法参数在任务代码/框架配置中定义并在 README 中说明；本版本不声明能编辑它们。

## 4. 必须实现的代码入口

清单的 `module:function` 必须指向 `source_root` 下的源文件与顶层同步函数。禁止指向外部路径、别名导出、类方法或被装饰器包装的入口。参数名及顺序固定，不得增加 *args、**kwargs、位置专用或关键字专用参数。静态扫描会验证这些条件。

```python
def make_env(config):
    # 返回单个环境，训练器据 num_envs 创建并行环境。
    ...

def alive(transition):
    # 返回本次单环境转移的有限原始标量奖励。
    ...

def train(config, output_dir):
    # 执行真实训练，将第 6 节产物写入已创建的 output_dir。
    ...
```

环境内部可以采用向量化实现，但上述外部入口保持固定。奖励函数输入 `transition` 为字典，必须提供 observation（本步结果观测）、action（本步动作）、base_reward（环境原始奖励）、terminated、truncated；任务可自行增加状态字段并在函数文档说明。结束步使用重置前的真实结果观测，不用自动重置后的下一回合初始观测。

原始标量、权重和贡献必须分离。贡献为 `raw × 实际权重 × scale`，其中 scale 为 1 或 step_dt。环境返回的奖励等于所有贡献之和。需要裁剪/归一化/内在奖励的项目必须确保输出指标注明并保持同一口径，不能混用不同尺度的总分与分项。

## 5. train 收到的解析配置

runner 从清单读取默认值并生成新字典，不修改清单。传给 train 的对象只包含：

```text
schema_version, task, num_envs, total_timesteps, seed, network,
step_dt, reward_scale, rewards
```

`rewards` 为清单奖励数组，每项增加 `mode`（default/fixed/disabled），weight 为解析后的权重，其他字段保留。实现必须使用解析结果，不重新读清单恢复旧权重。模式含义：

- default：使用默认值，允许已声明课程按项目逻辑调整。
- fixed：用户覆盖权重，课程不能再覆盖该项。
- disabled：贡献必须为零，包括课程更新后。

清单 enabled=false 或用户 disabled_rewards 包含该项时均为 disabled。即使同时传入固定权重，屏蔽仍优先。要重新启用清单原本禁用的项，先修改清单 enabled；当前覆盖接口没有独立的 enabled 开关。

训练可使用 `--config overrides.json`，仅允许以下六种顶层覆盖键：

```json
{
  "num_envs": 2,
  "total_timesteps": 128,
  "seed": 42,
  "network": {"type": "mlp", "actor": [32], "critic": [32], "activation": "relu"},
  "reward_weights": {"alive": 2.0},
  "disabled_rewards": ["angle"]
}
```

覆盖项均可省略，network 若填写必须完整。未知奖励名、重复屏蔽项、非有限权重和非法预算均拒绝；不要在训练器里再静默限制预算。运行中调整权重不属于 v1，修改后开启新训练。

## 6. 训练产物与指标契约

输出目录必须不存在，由 runner 创建。runner 保存 config.json（实际配置）、project.json（扫描清单与源码位置）、status.json（running/completed/failed）。训练函数负责下面两个文件：

- `checkpoint.bin`：非空、可由所属框架加载的最终检查点，扩展名不表示编码格式。保存格式与读取方式写入 README。优先先写临时文件，再原子替换。
- `metrics.jsonl`：UTF-8，每行独立 JSON，字段固定如下。

```json
{"step": 128, "reward_mean": 1.99, "components_mean": {"alive": 2.0, "angle": -0.01}}
```

step 为所有并行环境累计交互数，必须为严格递增的正整数。reward_mean 是该记录所代表的同一采样批次平均单步奖励；components_mean 是同批次、同权重尺度的各项平均贡献，键与清单奖励名完全一致，包括零贡献项。所有值有限，各项相加须等于 reward_mean，数值容差为 `1e-5 × max(1, abs(reward_mean))`。

可稀疏采样，但最后一条必须覆盖实际结束时刻，step 不小于预算。指标不是平均回合回报，不得用结束回合总分替代。此最小格式没有激活图、频率或回合分解，不得从它推导不存在的数据。

train 正常完成返回 None；失败抛异常，runner 标为 failed。只有产物检查成功才标为 completed。异常不能伪装正常完成。模型是否真正可加载还必须按第 7 节实测，文件非空检查不代替模型加载测试。

## 7. agent 必须执行的验收

1. 运行 `check`，退出码必须为 0，JSON 中 valid=true；修复所有错误，不能通过修改工具校验器绕过项目问题。
2. 在目标依赖环境执行一次小预算真实 train，退出码为 0，status.json 为 completed。预算取适合目标算法的最小量，不要求固定 128 或使用 CPU。
3. 验证固定权重与屏蔽在真实 metrics 中生效；分项和与总奖励一致。检查实际网络结构与所选配置一致。
4. 使用框架真实加载 checkpoint.bin，至少执行一次有限且符合动作空间的预测；记录具体命令及结果。不能只检查文件大小。
5. 给错误字段、未知奖励、错误函数签名做负例检查，确认静态校验或覆盖解析明确拒绝。
6. 将已验证的安装命令、依赖版本、训练命令、加载方式和输出位置写入项目 README；未验证硬件和规模如实列出。

工作台自带测试命令：`.venv/bin/python -m unittest discover -s tests -v`。其中模板测试会运行真实 PPO、读取奖励指标、加载检查点并预测。它验证模板和工具实现，不替代你为新环境做的运行验证。

静态报告 `validation=static`、`runtime_verified=false` 是刻意区分检查范围，不能由 agent 改成 true。训练验证结果记录在独立运行目录；依赖、硬件或代码变化后需要重新验证。静态格式无法证明任意代码都能训练。

## 8. 交给其他 agent 的指令

> 请严格按 RL Workbench 的 agent.md v1、随附 JSON Schema 和模板实现我的 RL 项目。任务为【开发者填写任务】。保持固定清单字段、代码入口签名、配置覆盖和输出格式；复用目标框架算法。完成静态 check、小预算真实 train、奖励覆盖/屏蔽验证和检查点加载预测后再交付。交付可复制运行的命令及实际结果。不要保证奖励设计或收敛，也不要把扫描通过说成已经具备尚未实现的网页训练功能。
