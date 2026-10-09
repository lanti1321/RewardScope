"""Parent-side lifecycle; all official dependencies stay in their own venv."""

import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .training import ACTIVE, atomic_json

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "microduck_rl"
PYTHON = UPSTREAM / ".venv" / "bin" / "python"
CATALOG = ROOT / ".runtime" / "microduck_catalog.json"
BRIDGE = ROOT / "rl_workbench" / "microduck_bridge.py"


def validate_microduck(data, catalog):
    if not isinstance(data, dict):
        raise ValueError("配置必须为对象")
    tasks = {t["id"]: t for t in catalog.get("tasks", [])}
    task = data.get("task", "Mjlab-Velocity-Flat-MicroDuck")
    if task not in tasks:
        raise ValueError("请选择已注册的官方 MicroDuck 任务")
    name = data.get("name", "MicroDuck 实验")
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 60:
        raise ValueError("名称需要 1–60 个字符")
    def integer(key, default, low, high):
        v = data.get(key, default)
        if isinstance(v, bool) or not isinstance(v, int) or not low <= v <= high:
            raise ValueError(f"{key} 需要为 {low}–{high} 之间的整数")
        return v
    scales = data.get("reward_scales", {})
    reward_names = {r["name"] for r in tasks[task]["rewards"]}
    if not isinstance(scales, dict) or set(scales) - reward_names:
        raise ValueError("奖励名称与官方任务不匹配")
    for v in scales.values():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 5:
            raise ValueError("奖励倍率必须为 0–5 的有限数值")
    weights = data.get("reward_weights", {})
    disabled = data.get("disabled_rewards", [])
    if not isinstance(weights, dict) or set(weights) - reward_names:
        raise ValueError("奖励权重名称与官方任务不匹配")
    for value in weights.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1e6:
            raise ValueError("奖励权重必须为 -1000000 到 1000000 的有限数值")
    if not isinstance(disabled, list) or any(not isinstance(name, str) or name not in reward_names for name in disabled):
        raise ValueError("屏蔽奖励名称与官方任务不匹配")
    network = data.get("network", {})
    if not isinstance(network, dict) or set(network) - {"actor", "critic"}:
        raise ValueError("网络配置仅支持 actor / critic")
    for role, dims in network.items():
        if not isinstance(dims, list) or not 1 <= len(dims) <= 6 or any(type(v) is not int or not 8 <= v <= 1024 for v in dims):
            raise ValueError("网络隐藏层需要 1–6 层，每层为 8–1024 的整数")
        if tasks[task].get("network", {}).get(role, {}).get("class_name") != "MLPModel":
            raise ValueError("当前仅支持修改已识别的 MLP 隐藏层")
    return {"reward_weights": {key: float(value) for key, value in weights.items()}, "disabled_rewards": sorted(set(disabled)), "network": network, "name": name.strip(), "task": task, "num_envs": integer("num_envs", 64, 16, 4096),
            "iterations": integer("iterations", 5, 1, 100000), "seed": integer("seed", 42, 0, 2147483647),
            "reward_scales": {key: float(value) for key, value in scales.items()}}


class MicroduckManager:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.process = None
        self.current_id = None
        for path in self.root.glob("*/run.json"):
            try:
                run = json.loads(path.read_text())
                if run["status"] in ACTIVE:
                    run["status"] = "interrupted"
                    run["error"] = "服务重启，上次训练未正常结束；已有官方 checkpoint 可在实验目录中找到。"
                    atomic_json(path, run)
            except (OSError, ValueError, KeyError):
                pass

    def catalog(self):
        if not CATALOG.exists() or not PYTHON.exists():
            return {"ready": False, "tasks": [], "error": "请先运行 bash setup_microduck.sh 安装官方环境并生成任务目录"}
        return json.loads(CATALOG.read_text())

    def rescan(self):
        with self.lock:
            if self.is_active():
                raise RuntimeError("请在训练结束后重新扫描项目")
            try:
                result = subprocess.run([str(PYTHON), str(BRIDGE), "catalog", str(CATALOG)], cwd=ROOT,
                                        capture_output=True, text=True, timeout=120)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError(f"扫描失败：{exc}") from exc
            if result.returncode:
                raise RuntimeError("项目解析失败：" + result.stderr[-1500:])
            return self.catalog()

    def get(self, run_id):
        if not (self.root / run_id / "run.json").is_file():
            raise KeyError("实验不存在")
        with self.lock:
            run = json.loads((self.root / run_id / "run.json").read_text())
            if run_id == self.current_id and self.process and self.process.poll() is not None and run["status"] in ACTIVE:
                run["status"] = "failed"
                run["error"] = f"官方训练进程退出，代码 {self.process.returncode}；请查看训练日志。"
                atomic_json(self.root / run_id / "run.json", run)
            if run["status"] in ACTIVE:
                run["elapsed"] = round((datetime.now(timezone.utc) - datetime.fromisoformat(run["created_at"])).total_seconds(), 2)
            return run

    def list(self):
        result = []
        for path in self.root.glob("*/run.json"):
            run = self.get(path.parent.name)
            result.append({key: run.get(key) for key in ("id", "config", "status", "iteration", "step", "created_at", "error")})
        return sorted(result, key=lambda r: r["created_at"], reverse=True)

    def is_active(self):
        return self.process is not None and self.process.poll() is None

    def start(self, data):
        catalog = self.catalog()
        if not catalog.get("ready"):
            raise RuntimeError(catalog["error"])
        config = validate_microduck(data, catalog)
        with self.lock:
            if self.is_active():
                raise RuntimeError("已有 MicroDuck 实验运行中")
            run_id = uuid.uuid4().hex[:12]
            folder = self.root / run_id
            (folder / "frames").mkdir(parents=True)
            run = {"id": run_id, "created_at": datetime.now(timezone.utc).isoformat(), "config": config,
                   "status": "starting", "iteration": 0, "step": 0, "elapsed": 0,
                   "phase": "载入官方训练代码", "history": [], "metrics": {}, "frames": [],
                   "source": catalog["source"], "error": None, "checkpoints": [], "onnx": []}
            atomic_json(folder / "run.json", run)
            atomic_json(folder / "control.json", {"action": "resume"})
            env = os.environ.copy()
            env.update({"MUJOCO_GL": "egl", "CUDA_VISIBLE_DEVICES": "0", "WANDB_MODE": "disabled",
                        "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                        "XDG_CACHE_HOME": str(ROOT / ".runtime" / "cache")})
            with (folder / "train.log").open("w") as log:
                self.process = subprocess.Popen([str(PYTHON), str(BRIDGE), "train", str(folder)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            self.current_id = run_id
            return run

    def control(self, run_id, action):
        with self.lock:
            run = self.get(run_id)
            if run_id != self.current_id or not self.is_active() or run["status"] not in ACTIVE:
                raise RuntimeError("此实验已结束，不能继续控制")
            if action not in {"pause", "resume", "stop"}:
                raise ValueError("未知操作")
            atomic_json(self.root / run_id / "control.json", {"action": action})
            return run

    def delete(self, run_id):
        if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{12}", run_id):
            raise ValueError("训练记录 ID 无效")
        with self.lock:
            folder = self.root / run_id
            if folder.is_symlink() or folder.resolve().parent != self.root:
                raise ValueError("训练记录路径无效")
            run = self.get(run_id)
            if run["status"] in ACTIVE or (run_id == self.current_id and self.is_active()):
                raise RuntimeError("请先停止训练并等待保存结束，再删除记录")
            shutil.rmtree(folder)
            if run_id == self.current_id:
                self.current_id = None
                self.process = None
            return {"deleted": run_id}

    def close(self):
        if self.is_active():
            atomic_json(self.root / self.current_id / "control.json", {"action": "stop"})
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)
