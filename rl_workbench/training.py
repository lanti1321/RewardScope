"""Single local training worker, immutable configs, atomic experiment snapshots."""

import copy
import json
import math
import numbers
import platform
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .envs import DEFAULT_WEIGHTS, RewardCartPole

ACTIVE = {"starting", "running", "pausing", "paused", "stopping"}


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def validate_config(data):
    if not isinstance(data, dict):
        raise ValueError("实验配置必须是对象")
    name = data.get("name", "我的实验")
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 60:
        raise ValueError("实验名称需为 1–60 个字符")

    def number(key, default, low, high, integer=False):
        value = data.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} 必须是有限数值")
        if not low <= value <= high or (integer and int(value) != value):
            raise ValueError(f"{key} 必须在 {low}–{high} 之间" + ("且为整数" if integer else ""))
        return int(value) if integer else float(value)

    steps = number("total_steps", 20480, 256, 200192, True)
    if steps % 256:
        raise ValueError("训练步数需为 256 的整数倍")
    weights = data.get("weights", DEFAULT_WEIGHTS)
    if not isinstance(weights, dict) or set(weights) != set(DEFAULT_WEIGHTS):
        raise ValueError("需要完整的四项奖励权重")
    for value in weights.values():
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 10:
            raise ValueError("奖励权重需在 0–10 之间")
    return {"name": name.strip(), "total_steps": steps,
            "learning_rate": number("learning_rate", 0.0003, 0.000001, 0.01),
            "seed": number("seed", 42, 0, 2147483647, True),
            "weights": {k: float(v) for k, v in weights.items()},
            "env": "CartPole-v1", "algorithm": "PPO", "eval_seeds": [1000, 1001, 1002]}


class TrainingStopped(Exception):
    pass


class RunManager:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.runs = {}
        self.worker = None
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        for path in self.root.glob("*/run.json"):
            try:
                run = json.loads(path.read_text())
                if run["status"] in ACTIVE:
                    run["status"] = "interrupted"
                    run["error"] = "服务已重启；上次训练中断。可复制配置重新训练。"
                    atomic_json(path, run)
                self.runs[run["id"]] = run
            except (OSError, ValueError, KeyError):
                continue

    def save(self, run):
        with self.lock:
            atomic_json(self.root / run["id"] / "run.json", run)

    def get(self, run_id):
        with self.lock:
            if run_id not in self.runs:
                raise KeyError("实验不存在")
            return copy.deepcopy(self.runs[run_id])

    def list(self):
        with self.lock:
            return [{k: copy.deepcopy(r[k]) for k in ("id", "config", "status", "step", "created_at", "best_eval", "error")}
                    for r in sorted(self.runs.values(), key=lambda r: r["created_at"], reverse=True)]

    def start(self, data):
        config = validate_config(data)
        with self.lock:
            if self.worker and self.worker.is_alive():
                raise RuntimeError("已有实验正在运行，请先停止当前训练")
            run_id = uuid.uuid4().hex[:12]
            run = {"id": run_id, "config": config, "created_at": datetime.now(timezone.utc).isoformat(),
                   "status": "starting", "step": 0, "elapsed": 0, "fps": 0,
                   "episodes": [], "evaluations": [], "updates": [], "latest": None,
                   "best_eval": None, "error": None, "model_saved": False, "best_model_saved": False}
            (self.root / run_id).mkdir()
            self.runs[run_id] = run
            self.save(run)
            self.stop_event.clear()
            self.pause_event.clear()
            self.worker = threading.Thread(target=self._train, args=(run,), daemon=True)
            self.worker.start()
            return self.get(run_id)

    def control(self, run_id, action):
        with self.lock:
            run = self.runs.get(run_id)
            if run is None:
                raise KeyError("实验不存在")
            if run["status"] not in ACTIVE:
                raise RuntimeError("此实验已结束；可以复制配置创建新实验")
            if action == "pause" and run["status"] in {"running", "starting"}:
                self.pause_event.set()
                run["status"] = "pausing"
            elif action == "resume" and run["status"] in {"paused", "pausing"}:
                self.pause_event.clear()
                run["status"] = "running"
            elif action == "stop":
                self.stop_event.set()
                self.pause_event.clear()
                run["status"] = "stopping"
            else:
                raise RuntimeError("当前状态不支持此操作")
            self.save(run)
            return self.get(run_id)

    def _gate(self, run):
        while True:
            with self.lock:
                if self.stop_event.is_set():
                    raise TrainingStopped()
                if not self.pause_event.is_set():
                    if run["status"] == "paused":
                        run["status"] = "running"
                    return
                run["status"] = "paused"
            if self.stop_event.wait(0.1):
                raise TrainingStopped()

    def _train(self, run):
        model = None
        env = None
        started = time.monotonic()
        try:
            import torch
            import numpy
            import gymnasium
            import stable_baselines3
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import BaseCallback
            torch.set_num_threads(1)
            manager = self
            config = run["config"]
            env = RewardCartPole(config["weights"])
            model = PPO("MlpPolicy", env, learning_rate=config["learning_rate"], n_steps=256,
                        batch_size=64, n_epochs=10, seed=config["seed"], device="cpu", verbose=0)
            with self.lock:
                run["runtime"] = {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": numpy.__version__, "gymnasium": gymnasium.__version__, "stable_baselines3": stable_baselines3.__version__}
                run["ppo"] = {"n_steps": 256, "batch_size": 64, "n_epochs": 10, "gamma": model.gamma,
                    "gae_lambda": model.gae_lambda, "clip_range": 0.2, "ent_coef": model.ent_coef,
                    "vf_coef": model.vf_coef, "max_grad_norm": model.max_grad_norm, "device": "cpu"}

            def capture_update():
                return {k.removeprefix("train/"): float(v) for k, v in model.logger.name_to_value.items()
                        if k.startswith("train/") and isinstance(v, numbers.Real) and math.isfinite(v)}

            class Recorder(BaseCallback):
                def __init__(self):
                    super().__init__()
                    self.length = 0
                    self.reward_sum = 0.0
                    self.component_sums = {k: 0.0 for k in DEFAULT_WEIGHTS}
                    self.last_save = 0
                    self.last_eval = -2048
                    self.last_update = -1

                def _on_rollout_start(self):
                    manager._gate(run)
                    # Here the preceding PPO optimization has completed.
                    update = capture_update()
                    if update and self.num_timesteps != self.last_update:
                        with manager.lock:
                            run["updates"].append({"step": self.num_timesteps, **update})
                        self.last_update = self.num_timesteps
                    if self.num_timesteps - self.last_eval >= 2048:
                        manager._evaluate(run, self.model)
                        self.last_eval = self.num_timesteps

                def _on_step(self):
                    manager._gate(run)
                    info = self.locals["infos"][0]
                    self.length += 1
                    self.reward_sum += info["reward"]
                    for key in self.component_sums:
                        self.component_sums[key] += info["components"][key]
                    with manager.lock:
                        run["step"] = self.num_timesteps
                        run["elapsed"] = round(time.monotonic() - started, 2)
                        run["fps"] = round(self.num_timesteps / max(run["elapsed"], 0.01))
                        run["latest"] = {k: copy.deepcopy(info[k]) for k in ("before", "after", "action", "raw_reward", "features", "components", "reward", "terminated", "truncated")}
                        if self.locals["dones"][0]:
                            run["episodes"].append({"episode": len(run["episodes"]) + 1, "step": self.num_timesteps,
                                "length": self.length, "reward": self.reward_sum,
                                "components": self.component_sums.copy(), "terminated": info["terminated"], "truncated": info["truncated"]})
                            self.length = 0
                            self.reward_sum = 0.0
                            self.component_sums = {k: 0.0 for k in DEFAULT_WEIGHTS}
                    if time.monotonic() - self.last_save > 1:
                        manager.save(run)
                        self.last_save = time.monotonic()
                    return True

            with self.lock:
                if run["status"] == "starting":
                    run["status"] = "running"
            model.learn(total_timesteps=config["total_steps"], callback=Recorder())
            with self.lock:
                run["updates"].append({"step": model.num_timesteps, **capture_update()})
            self._gate(run)
            self._evaluate(run, model)
            with self.lock:
                run["status"] = "completed"
        except TrainingStopped:
            with self.lock:
                run["status"] = "stopped"
        except Exception as exc:
            traceback.print_exc()
            with self.lock:
                run["status"] = "failed"
                run["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                if model is not None:
                    model.save(self.root / run["id"] / "model.zip")
                    with self.lock:
                        run["model_saved"] = True
            except Exception as exc:
                with self.lock:
                    run["error"] = f"模型保存失败：{exc}"
                    run["status"] = "failed"
            if env is not None:
                env.close()
            with self.lock:
                run["elapsed"] = round(time.monotonic() - started, 2)
            self.save(run)

    def _evaluate(self, run, model):
        env = RewardCartPole(run["config"]["weights"])
        episodes = []
        try:
            for seed in run["config"]["eval_seeds"]:
                obs, _ = env.reset(seed=seed)
                frames = []
                while True:
                    self._gate(run)
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(int(action))
                    frames.append(info)
                    if terminated or truncated:
                        break
                episodes.append({"seed": seed, "length": len(frames), "reward": sum(f["reward"] for f in frames), "frames": frames})
            index = len(run["evaluations"])
            atomic_json(self.root / run["id"] / f"eval-{index}.json", {"step": run["step"], "episodes": episodes})
            mean = sum(ep["length"] for ep in episodes) / len(episodes)
            entry = {"index": index, "step": run["step"], "mean_length": mean,
                     "min_length": min(ep["length"] for ep in episodes), "max_length": max(ep["length"] for ep in episodes),
                     "mean_reward": sum(ep["reward"] for ep in episodes) / len(episodes)}
            with self.lock:
                run["evaluations"].append(entry)
                is_best = run["best_eval"] is None or mean > run["best_eval"]
                if is_best:
                    run["best_eval"] = mean
            if is_best:
                model.save(self.root / run["id"] / "best_model.zip")
                with self.lock:
                    run["best_model_saved"] = True
            self.save(run)
        finally:
            env.close()
