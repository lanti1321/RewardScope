# RL Workbench

[中文](README.md) | **English**

A local workbench for discovering reinforcement learning projects and inspecting training. MicroDuck is the first fully integrated upstream project. The repository also provides a standard v1 project contract, a runnable template, and a strict scanner.

There are two workflows:

- **MicroDuck:** configure reward weights and masks, edit MLP hidden layers, start/pause/stop training, inspect the latest simulation image and reward trends/composition, download models, and delete records in the browser.
- **Standard v1 projects:** validate structure and entry points, override configuration, run training, and check metrics/model artifacts through the CLI. Generic projects do not yet use MicroDuck's browser training controls or live monitor.

The contract checks project format and the execution workflow. It does not judge reward design, convergence, or policy performance. Arbitrary projects that do not follow the contract are not integrated automatically.

## 1. Where to put the workbench and RL project

Example workbench directory layout:

```text
rl-workbench/                    Workbench root
├── rl_workbench/               Application code
├── agent.md                    RL project instructions for other agents
├── upstream/microduck_rl/      Upstream MicroDuck and its independent environment
├── .venv/                      Workbench dependencies
├── .runtime/                   MicroDuck catalog, logs and compilation caches
└── runs/microduck/             MicroDuck training records and models
```

`upstream/microduck_rl/` is the fixed location used by the MicroDuck adapter. Other projects do not have to go in upstream, and placing them there does not make them MicroDuck projects.

**For new projects, put the complete workbench in a separate subdirectory of your RL project.**

```text
my_rl/                         Your RL project
├── rl-project.json            Task and configuration manifest
├── pyproject.toml             Training dependencies
├── README.md                  Instructions for your task
├── src/my_rl/                 Environment, rewards and training code
├── .venv/                     Training environment
├── runs/                      CLI training outputs
└── rl-workbench/              This tool, kept in its own directory
    ├── README.md
    ├── agent.md
    ├── schemas/
    ├── rl_workbench/
    └── .venv/                 Workbench environment
```

Start the server from `rl-workbench/`. The standard scanner searches upward for the nearest `rl-project.json`. Do not merge the two projects' source, README files or virtual environments.

The projects can also live beside each other or in unrelated directories: pass the target project path to scan/train. Only ancestor discovery is automatic; the tool does not register every subproject under upstream.

When distributing the tool, include its source, scripts, requirements.txt, schemas, templates, docs and agent.md. Do not distribute old `.venv`, `.runtime` or runs directories as an installation. Recreate the upstream environment with the installation script if MicroDuck is needed.

## 2. Install and start

Run these commands from the workbench root with Python 3.11–3.13:

```bash
bash setup.sh
bash start.sh
```

If dependencies are already installed, only `bash start.sh` is needed. Do not start another server on a port already in use.

Use **语言 / Language** in the sidebar to switch between Chinese and English. The preference is stored in your browser and shared across all three pages. Switching preserves training form values. Run names, code identifiers and original logs are kept as entered.

Open the [workbench](http://127.0.0.1:8765/) or [standard project scanner](http://127.0.0.1:8765/projects). The server listens on localhost only. Keep its terminal open when running in the foreground.

```bash
bash start.sh --port 8766 --data-dir runs-other
```

`--data-dir` sets the browser-managed record directory. Standard CLI training uses a separate `--output-dir` option.

MicroDuck is not required for standard project scanning or the CPU template. For MicroDuck training, install an NVIDIA CUDA driver and uv; its independent environment uses Python 3.12:

```bash
bash setup_microduck.sh
bash start.sh
```

The script downloads a pinned MicroDuck version into upstream and installs its training dependencies. Do not relocate an installed virtual environment.

## 3. Create a standard RL project

From the workbench root, use a target directory that does not exist yet:

```bash
.venv/bin/python -m rl_workbench.project init /absolute/path/to/my_rl
.venv/bin/python -m rl_workbench.project check /absolute/path/to/my_rl
.venv/bin/python -m rl_workbench.project train /absolute/path/to/my_rl \
  --task CartPole-Balance-v1 --output-dir /absolute/path/to/my_rl/runs/first
```

The template performs real CPU PPO training on CartPole. Follow [agent.md](agent.md) to edit the manifest, environment, rewards and training entry point for your task. That document lists which files to change. `init` copies the template; it does not move the workbench into the new project. For a nested layout, create the project first, then copy the tool's source into its `rl-workbench/` directory and install the workbench environment there.

For an existing RL project, add the manifest and wrappers matching the required signatures. Do not use init to overwrite it.

Training must use a Python environment with the target framework installed; scanning requires only the Python standard library. With the tool inside `my_rl/rl-workbench/`, run from that tool directory:

```bash
# Discover the manifest in an ancestor directory
.venv/bin/python -m rl_workbench.project check

# Train with the parent RL project's independent Python environment
../.venv/bin/python -m rl_workbench.project train .. \
  --task Your-Task-ID --output-dir ../runs/first
```

Replace `Your-Task-ID` with a real manifest ID. Install the target project's pyproject.toml dependencies into `../.venv` first. Keep the working directory at the workbench root so its module remains importable.

Add `--config /path/to/overrides.json` to override the budget, seed, network or rewards. For example:

```json
{"total_timesteps": 128, "reward_weights": {"alive": 2.0}, "disabled_rewards": ["angle"]}
```

The output directory must not exist. Outputs include config.json, project.json, status.json, metrics.jsonl and checkpoint.bin. A completed status means training returned and basic artifact validation passed. Actual model loading and prediction must still be verified using your task's README instructions.

## 4. MicroDuck browser workflow

1. In Environment & rewards, select an upstream task and configure environment count, iterations, reward weights/masks and hidden layers.
2. Start training directly; a preliminary test run is not required. The default 64 environments × 5 iterations is just a short configuration.
3. In Training monitor, inspect metrics, the latest image and reward composition. The timeline replays sparse reward samples. New runs retain only the latest image, not a historical video.
4. Pause and resume as needed. Stop requests a model save. Ctrl+C on the server also requests a stop and save, but saving cannot be guaranteed after a driver failure.
5. To delete a finished record, use the trash button in the upper-right corner of its record card. Confirmation permanently removes that run's models, logs, images and statistics. Active, paused or still-saving runs cannot be deleted.

Nonzero reward frequency is not task success rate. See the [MicroDuck notes](docs/microduck.md) for the differences between step contributions, upstream episode logs and episode reward breakdowns.

## 5. Source and data directories

- `rl_workbench/`: discovery, contract validation, training management, statistics and UI.
- `schemas/`: v1 JSON Schema. `templates/`: runnable standard project template.
- `tests/`: regression tests. `docs/`: usage and interface reference.
- `upstream/`: upstream training code and dependencies. `.venv/`: workbench dependencies.
- `runs/`: browser training records. `.runtime/`: caches and logs.
- `.archive/`: legacy local attachments, not used at runtime.

Back up complete MicroDuck run directories. CLI project results go to the specified output-dir, which may be outside the workbench runs folder. Training environments and source code are not disposable caches.

## 6. Development checks and documentation

```bash
.venv/bin/python -m unittest discover -s tests -v
node --check rl_workbench/static/microduck.js
node --check rl_workbench/static/projects.js
node --check rl_workbench/static/app.js
node --test tests/i18n.test.cjs
```

Tests include real CPU training, model loading and local HTTP endpoints; no GPU is required. Invalid v1 contracts produce a nonzero exit code with a reason. Static validation does not verify dependencies, hardware or training logic.

- [Agent instructions and project file checklist (Chinese)](agent.md)
- [v1 field definitions](schemas/rl-project-v1.schema.json)
- [Advanced MicroDuck integration reference (Chinese)](docs/workbench-integration.md)
- [MicroDuck metrics reference (Chinese)](docs/microduck.md)
- [CartPole browser example (Chinese)](docs/cartpole.md)

## 7. Third-party licensing and independent development

RewardScope (also labeled RL Workbench in the interface) is independently developed and is not affiliated with or endorsed by Pollen Robotics. MicroDuck and Pollen Robotics are named only to identify compatibility and upstream provenance; this tool is not an official product of Pollen Robotics.

The installation script retrieves the [official MicroDuck repository](https://github.com/pollen-robotics/microduck_rl), pinned to `cb70b792312d559a4da09064d92009079671815f`. The `upstream/` directory is excluded from this tool's source repository.

- MicroDuck code is licensed under [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0). When copying or redistributing upstream code, comply with that license, retain applicable copyright, attribution and NOTICE statements, and identify modified files.
- The pinned version's [upstream README](https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/README.md#license) separately labels 3D model files **Creative Commons BY-SA-NC**, without specifying a version in that statement. Do not assume those assets are Apache-2.0. Check the complete license applicable to each asset before use, modification or redistribution, and confirm authorization for commercial use.
- Other dependencies and assets retain their own licenses. This project's license, including any noncommercial restriction, does not replace or alter third-party licenses or grant rights to third-party trademarks.
