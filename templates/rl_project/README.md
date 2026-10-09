# RL Workbench v1 project template

This is a real CPU PPO/CartPole training example, not a locomotion recipe. Replace the task logic and rename `balance` to your task; update all manifest references together.

Install in your target Python environment with `python -m pip install -e .`.
From the RL Workbench repository (using that environment's Python), run:

```bash
python -m rl_workbench.project check /absolute/path/to/this/project
python -m rl_workbench.project train /absolute/path/to/this/project --task CartPole-Balance-v1 --output-dir /absolute/path/to/new-run
```

The output directory must not exist. Add `--config /path/to/overrides.json` to override budget, network or rewards. The template needs no editable installation when its dependencies are already installed: the runner adds `source_root` to the import path. Keep the working directory at RL Workbench so its module is importable.

`rl-project.json` is the source of truth. Static check does not import the project. Train explicitly executes the project and checks real output artifacts. It records `config.json`, `project.json`, `status.json`, `metrics.jsonl`, `checkpoint.bin`. PPO rounds the requested environment interactions up to a complete rollout. The minimum v1 contract does not include pause/resume, live preview or generic browser training controls; those remain richer adapter capabilities.
