"""Runs inside the pinned official MicroDuck environment, not the SB3 venv.

The official run_train + task runner own all learning and physics. This adapter
observes the logger and reward manager, with optional numeric reward weights and masking.
"""

import argparse
import importlib.metadata
import inspect
import sys
import json
import math
import os
import platform
import subprocess
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "microduck_rl"
TASK = "Mjlab-Velocity-Flat-MicroDuck"
sys.path.insert(0, str(ROOT))
from rl_workbench.discovery import scan_project
from rl_workbench.reward_settings import resolve_reward_weights
from rl_workbench.live_preview import write_preview


def source_location(function):
    try:
        return {"file": inspect.getsourcefile(function), "line": inspect.getsourcelines(function)[1], "doc": inspect.getdoc(function) or ""}
    except (TypeError, OSError):
        return None



def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def provenance():
    return {
        "repository": "https://github.com/pollen-robotics/microduck_rl",
        "commit": subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True).strip()),
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ("mjlab", "warp-lang", "torch", "rsl-rl-lib", "mujoco", "better-actuator-models")},
    }


def catalog(output):
    import mjlab.tasks  # noqa: F401 - official plugin registry
    import torch
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

    tasks = []
    for task_id in list_tasks():
        if "MicroDuck" not in task_id:
            continue
        cfg = load_env_cfg(task_id)
        agent = load_rl_cfg(task_id)
        tasks.append({"id": task_id, "episode_seconds": cfg.episode_length_s,
            "step_dt": cfg.sim.mujoco.timestep * cfg.decimation,
            "scale_by_dt": cfg.scale_rewards_by_dt, "steps_per_iteration": agent.num_steps_per_env,
            "default_iterations": agent.max_iterations, "default_num_envs": cfg.scene.num_envs,
            "rewards": [{"name": name, "weight": term.weight,
                         "source": source_location(term.func), "function": f"{term.func.__module__}.{getattr(term.func, '__name__', type(term.func).__name__)}"}
                        for name, term in cfg.rewards.items() if term is not None],
            "network": {role: {"hidden_dims": list(getattr(agent, role).hidden_dims), "activation": getattr(agent, role).activation, "class_name": getattr(agent, role).class_name} for role in ("actor", "critic")},
            "curricula": list(cfg.curriculum), "terminations": list(cfg.terminations),
            "observation_groups": list(cfg.observations)})
    write_json(output, {"ready": True, "source": provenance(), "tasks": tasks,
                       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                       "default_task": TASK, "discovery": scan_project(UPSTREAM)})


class StopTraining(Exception):
    pass


def save_stopped_checkpoint(runner, directory):
    """Publish only a fully saved checkpoint; CUDA errors may leave partial files."""
    target = Path(directory) / "model_stopped.pt"
    pending = target.with_name(".model_stopped.pending.pt")
    try:
        runner.save(str(pending))
        pending.replace(target)
    finally:
        pending.unlink(missing_ok=True)


def run_experiment(directory):
    directory = Path(directory).resolve()
    state_path = directory / "run.json"
    state = json.loads(state_path.read_text())
    config = state["config"]
    started = time.monotonic()
    runner_ref = None
    renderer = None
    environment = None
    last_publish = 0.0
    last_preview = 0.0
    def publish():
        state["elapsed"] = round(time.monotonic() - started, 2)
        write_json(state_path, state)

    def gate():
        while True:
            try:
                command = json.loads((directory / "control.json").read_text()).get("action", "resume")
            except (OSError, ValueError):
                command = "resume"
            if command == "stop":
                raise StopTraining()
            if command != "pause":
                if state["status"] == "paused":
                    state["status"] = "running"
                    publish()
                return
            if state["status"] != "paused":
                state["status"] = "paused"
                publish()
            time.sleep(.15)

    try:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ["WANDB_MODE"] = "disabled"
        import torch
        from rl_workbench.diagnostics import RewardProbe, EpisodeRewardProbe
        import mjlab.tasks  # noqa: F401
        import mjlab.scripts.train as official_train
        from mjlab.tasks.registry import load_runner_cls
        from mjlab.viewer.offscreen_renderer import OffscreenRenderer
        from PIL import Image

        if not torch.cuda.is_available():
            raise RuntimeError("MicroDuck 官方训练需要 CUDA GPU，当前进程无法访问 GPU")
        torch.set_num_threads(4)
        state["source"] = provenance()
        state["gpu"] = torch.cuda.get_device_name(0)
        cfg = official_train.TrainConfig.from_task(config["task"])
        cfg.env.scene.num_envs = config["num_envs"]
        cfg.agent.max_iterations = config["iterations"]
        cfg.agent.seed = config["seed"]
        for role, dims in config.get("network", {}).items():
            getattr(cfg.agent, role).hidden_dims = tuple(dims)
        cfg.agent.logger = "tensorboard"
        cfg.agent.upload_model = False
        cfg.agent.run_name = config["name"]
        cfg.env.viewer.width = 640
        cfg.env.viewer.height = 400
        cfg.env.viewer.env_idx = 0
        cfg.env.viewer.max_extra_envs = 0
        cfg.env.viewer.distance = 0.85
        cfg.env.viewer.elevation = -18.0
        cfg.env.viewer.azimuth = 135.0
        names = [name for name, term in cfg.env.rewards.items() if term is not None]
        unknown = set(config["reward_scales"]) - set(names)
        if unknown:
            raise ValueError(f"不存在的奖励项：{sorted(unknown)}")
        base_class = load_runner_cls(config["task"])
        state["steps_per_iteration"] = cfg.agent.num_steps_per_env
        state["episode_seconds"] = cfg.env.episode_length_s
        state["step_dt"] = cfg.env.sim.mujoco.timestep * cfg.env.decimation
        state["scale_by_dt"] = cfg.env.scale_rewards_by_dt
        state["phase"] = "创建官方环境与 CUDA 内核"
        publish()
        # At most ~400 sampled images per run, preserving exact sample timestamps.
        sample_stride = max(4, math.ceil(config["iterations"] * cfg.agent.num_steps_per_env / 400))

        class ObservedRunner(base_class):
            def __init__(self, env, train_cfg, log_dir=None, device="cpu", **kwargs):
                nonlocal runner_ref, renderer, environment
                super().__init__(env, train_cfg, log_dir, device, **kwargs)
                runner_ref = self
                base = env.unwrapped
                environment = base
                state["observation_dims"] = {key: list(value.shape[1:]) for key, value in env.get_observations().items()}
                state["action_dim"] = env.num_actions
                try:
                    renderer = OffscreenRenderer(base.sim.mj_model, base.cfg.viewer, base.scene)
                    renderer.initialize()
                except Exception as exc:
                    state["render_error"] = f"画面渲染不可用：{exc}"
                    renderer = None
                rm = base.reward_manager
                original_compute = rm.compute
                state["diagnostics_version"] = 2
                state["live_preview"] = True
                state["sampling"] = {"stride": sample_stride, "reward_frequency": "all environments, all steps; abs(weighted contribution)>1e-8"}
                reward_probe = RewardProbe(rm.active_terms, device)
                episode_ids = torch.zeros(env.num_envs, dtype=torch.long, device=device)

                def compute(dt):
                    nonlocal last_publish, last_preview
                    gate()
                    # Resolve fixed overrides and masks, then restore the manager
                    # weights before official reset/curriculum code runs.
                    current = {key: rm.get_term_cfg(key).weight for key in rm.active_terms}
                    effective = resolve_reward_weights(current, config)
                    for key in current:
                        rm.get_term_cfg(key).weight = effective[key]
                    try:
                        reward = original_compute(dt)
                        step = base.common_step_counter
                        scale = dt if base.cfg.scale_rewards_by_dt else 1.0
                        reward_probe.update(rm._step_reward * scale, {k: v * scale for k, v in effective.items()})
                        if renderer is not None and time.monotonic() - last_preview >= .5:
                            renderer.update(base.sim.data)
                            write_preview(directory, Image.fromarray(renderer.render()), step)
                            last_preview = time.monotonic()
                        if step % sample_stride == 0:
                            scale = dt if base.cfg.scale_rewards_by_dt else 1.0
                            components = {key: float(values[0]) * scale for key, values in rm.get_active_iterable_terms(0)}
                            total = float(reward[0].item())
                            residual = total - sum(components.values())
                            if not math.isfinite(total) or abs(residual) > 1e-4 * max(1, abs(total)):
                                raise RuntimeError(f"奖励分项与总奖励不一致：{residual}")
                            frame = {"index": len(state["frames"]), "env_step": int(step), "env_id": 0,
                                     "episode_step": int(base.episode_length_buf[0].item()),
                                     "reward": total, "components": components, "weights": effective,
                                     "sum_residual": residual, "step_dt": dt,
                                     "qpos": base.sim.data.qpos[0].cpu().tolist(),
                                     "action": base.action_manager.action[0].cpu().tolist(),
                                     "terminated": bool(base.reset_terminated[0].item()),
                                     "truncated": bool(base.reset_time_outs[0].item()),
                                     "episode_id": int(episode_ids[0]),
                                     "reward_stats": reward_probe.snapshot()}
                            frame["reward_batch_mean"] = {key: float((rm._step_reward[:, i] * scale).mean()) for i, key in enumerate(rm.active_terms)}
                            state["frames"].append(frame)
                            state["latest_weights"] = effective
                            if time.monotonic() - last_publish >= .5:
                                publish()
                                last_publish = time.monotonic()
                        episode_ids.add_((base.reset_terminated | base.reset_time_outs).long())
                        return reward
                    finally:
                        for key, weight in current.items():
                            rm.get_term_cfg(key).weight = weight

                rm.compute = compute
                episode_probe = EpisodeRewardProbe(rm.active_terms, env.num_envs, device, window=self.logger.rewbuffer.maxlen)
                original_process_step = self.logger.process_env_step
                def process_step(rewards, dones, extras, intrinsic_rewards=None):
                    if self.logger.writer is not None:
                        scale = base.step_dt if base.cfg.scale_rewards_by_dt else 1.0
                        episode_probe.update(rm._step_reward * scale, dones)
                    return original_process_step(rewards, dones, extras, intrinsic_rewards)
                self.logger.process_env_step = process_step
                original_init = self.logger.init_logging_writer
                scalars = {}

                def init_writer():
                    original_init()
                    original_scalar = self.logger.writer.add_scalar
                    def scalar(tag, value, global_step=None, *args, **kwargs):
                        value_float = float(value)
                        if math.isfinite(value_float) and not tag.endswith('/time'):
                            scalars[tag] = value_float
                        return original_scalar(tag, value, global_step, *args, **kwargs)
                    self.logger.writer.add_scalar = scalar

                self.logger.init_logging_writer = init_writer
                original_log = self.logger.log
                def log(*args, **kwargs):
                    scalars.clear()
                    original_log(*args, **kwargs)
                    iteration = int(kwargs.get("it", args[0] if args else 0)) + 1
                    state["iteration"] = iteration
                    state["step"] = iteration * cfg.agent.num_steps_per_env * config["num_envs"]
                    state["history"].append({"iteration": iteration, "step": state["step"], "metrics": dict(scalars)})
                    state["metrics"] = dict(scalars)
                    state["reward_breakdown"] = episode_probe.snapshot(scalars.get("Train/mean_reward"))
                    state["reward_stats"] = reward_probe.snapshot()
                    state["phase"] = "PPO 策略更新"
                    publish()
                    gate()
                self.logger.log = log
                state["status"] = "running"
                state["phase"] = "采集官方训练环境数据"
                publish()

        official_train.load_runner_cls = lambda task_id: ObservedRunner if task_id == config["task"] else load_runner_cls(task_id)
        official_train.run_train(config["task"], cfg, directory / "official")
        state["status"] = "completed"
        state["phase"] = "训练结束"
    except StopTraining:
        state["status"] = "stopped"
        state["phase"] = "已停止"
        if runner_ref is not None:
            try:
                save_stopped_checkpoint(runner_ref, directory / "official")
            except Exception as exc:
                traceback.print_exc()
                state["status"] = "failed"
                state["phase"] = "训练已停止，保存模型失败"
                state["error"] = f"停止后保存模型失败：{exc}"
    except Exception as exc:
        traceback.print_exc()
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["phase"] = "训练失败，请查看日志"
    finally:
        if renderer is not None:
            try:
                renderer.close()
            except Exception as exc:
                state["render_error"] = f"关闭渲染器时出错：{exc}"
        if environment is not None:
            try:
                environment.close()
            except Exception:
                pass
        checkpoints = list((directory / "official").glob("model_*.pt"))
        state["checkpoints"] = [p.name for p in sorted(checkpoints,
            key=lambda p: int(p.stem.split('_')[-1]) if p.stem.split('_')[-1].isdigit() else float('inf'))]
        state["onnx"] = sorted(p.name for p in (directory / "official").glob("*.onnx"))
        publish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["catalog", "train"])
    parser.add_argument("path")
    args = parser.parse_args()
    if args.command == "catalog":
        catalog(args.path)
    else:
        run_experiment(args.path)
