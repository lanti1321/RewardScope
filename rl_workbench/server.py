"""Local-only HTTP API and static UI. No external services or frontend build needed."""

import argparse
import json
import mimetypes
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .training import RunManager
from .microduck import MicroduckManager
from .project_contract import inspect_project, find_project_root

STATIC = Path(__file__).parent / "static"


def create_server(port=8765, data_dir="runs"):
    manager = RunManager(data_dir)
    duck = MicroduckManager(Path(data_dir).resolve() / "microduck")
    launch_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if not args or str(args[1]) not in {"200", "304"}:
                super().log_message(fmt, *args)

        def send(self, data, status=200, content_type="application/json; charset=utf-8", filename=None):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def valid_host(self):
            return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

        def do_GET(self):
            if not self.valid_host():
                return self.send({"error": "仅接受本机访问"}, 403)
            path = urlsplit(self.path).path
            try:
                if path == "/api/projects/location":
                    return self.send({"directory": str(find_project_root(Path.cwd()))})
                if path == "/api/microduck/info":
                    return self.send(duck.catalog())
                if path == "/api/microduck/runs":
                    return self.send(duck.list())
                live = re.fullmatch(r"/api/microduck/runs/([a-f0-9]{12})/latest", path)
                if live:
                    # Do not parse the growing training history for each preview.
                    preview = duck.root / live.group(1) / "latest.json"
                    return self.send(preview.read_bytes(), content_type="application/json")
                match_duck = re.fullmatch(r"/api/microduck/runs/([a-f0-9]{12})(?:/(log|export|frame/\d+|download/[a-zA-Z0-9_.-]+))?", path)
                if match_duck:
                    run_id, action = match_duck.groups()
                    run = duck.get(run_id)
                    folder = duck.root / run_id
                    if not action:
                        return self.send(run)
                    if action == "export":
                        return self.send(run, filename=f"microduck-{run_id}.json")
                    if action == "log":
                        log_path = folder / "train.log"
                        if not log_path.exists():
                            return self.send({"text": "等待训练日志…"})
                        with log_path.open("rb") as log:
                            log.seek(max(0, log_path.stat().st_size - 24000))
                            return self.send({"text": log.read().decode("utf-8", errors="replace")})
                    if action.startswith("frame/"):
                        index = int(action.split("/")[1])
                        if not 0 <= index < len(run["frames"]):
                            raise KeyError("采样帧不存在")
                        return self.send((folder / "frames" / f"{index}.jpg").read_bytes(), content_type="image/jpeg")
                    if action.startswith("download/"):
                        name = action.split("/")[1]
                        if name not in run.get("checkpoints", []) + run.get("onnx", []):
                            raise KeyError("模型文件不存在")
                        return self.send((folder / "official" / name).read_bytes(), content_type="application/octet-stream", filename=f"{run_id}-{name}")
                if path == "/api/runs":
                    return self.send(manager.list())
                if path == "/api/health":
                    return self.send({"ok": True, "environment": "CartPole-v1", "algorithm": "PPO"})
                match = re.fullmatch(r"/api/runs/([a-f0-9]{12})(?:/(eval/\d+|export|model|best-model))?", path)
                if match:
                    run_id, action = match.groups()
                    run = manager.get(run_id)
                    if not action:
                        return self.send(run)
                    if action == "export":
                        return self.send(run, filename=f"rl-{run_id}.json")
                    if action.startswith("eval/"):
                        index = int(action.split("/")[1])
                        if not 0 <= index < len(run["evaluations"]):
                            raise KeyError("评估记录不存在")
                        return self.send((manager.root / run_id / f"eval-{index}.json").read_bytes())
                    name = "best_model.zip" if action == "best-model" else "model.zip"
                    flag = "best_model_saved" if action == "best-model" else "model_saved"
                    if not run[flag]:
                        raise KeyError("模型尚未保存")
                    return self.send((manager.root / run_id / name).read_bytes(), content_type="application/zip", filename=f"{run_id}-{name}")
                files = {"/": "microduck.html", "/microduck": "microduck.html", "/cartpole": "index.html",
                         "/i18n.js": "i18n.js", "/translations.js": "translations.js",
                         "/projects": "projects.html", "/projects.js": "projects.js",
                         "/microduck.js": "microduck.js", "/microduck.css": "microduck.css",
                         "/app.js": "app.js", "/style.css": "style.css", "/favicon.svg": "favicon.svg"}
                if path in files:
                    file = STATIC / files[path]
                    return self.send(file.read_bytes(), content_type=mimetypes.guess_type(file.name)[0] + "; charset=utf-8")
                self.send({"error": "页面不存在"}, 404)
            except (KeyError, FileNotFoundError):
                self.send({"error": "记录不存在"}, 404)

        def do_POST(self):
            if not self.valid_host():
                return self.send({"error": "仅接受本机访问"}, 403)
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                return self.send({"error": "不允许跨站操作"}, 403)
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.send({"error": "需要 JSON 请求"}, 415)
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 16384:
                    return self.send({"error": "请求大小无效"}, 400)
                data = json.loads(self.rfile.read(length))
                path = urlsplit(self.path).path
                if path == "/api/projects/scan":
                    if not isinstance(data, dict) or set(data) != {"directory"} or not isinstance(data["directory"], str) or not data["directory"].strip():
                        raise ValueError("需要项目根目录 directory")
                    return self.send(inspect_project(data["directory"]))
                if path == "/api/microduck/scan":
                    with launch_lock:
                        return self.send(duck.rescan())
                if path == "/api/microduck/runs":
                    with launch_lock:
                        if manager.worker and manager.worker.is_alive():
                            raise RuntimeError("请先停止正在运行的 CartPole 实验")
                        result = duck.start(data)
                    return self.send(result, 201)
                deletion = re.fullmatch(r"/api/microduck/runs/([a-f0-9]{12})/delete", path)
                if deletion:
                    return self.send(duck.delete(deletion.group(1)))
                match_duck = re.fullmatch(r"/api/microduck/runs/([a-f0-9]{12})/(pause|resume|stop)", path)
                if match_duck:
                    return self.send(duck.control(*match_duck.groups()))
                if path == "/api/runs":
                    with launch_lock:
                        if duck.is_active():
                            raise RuntimeError("请先停止正在运行的 MicroDuck 实验")
                        result = manager.start(data)
                    return self.send(result, 201)
                match = re.fullmatch(r"/api/runs/([a-f0-9]{12})/(pause|resume|stop)", path)
                if match:
                    return self.send(manager.control(*match.groups()))
                self.send({"error": "接口不存在"}, 404)
            except (ValueError, TypeError) as exc:
                self.send({"error": str(exc)}, 400)
            except KeyError:
                self.send({"error": "实验不存在"}, 404)
            except RuntimeError as exc:
                self.send({"error": str(exc)}, 409)
            except OSError:
                self.send({"error": "文件操作失败，请检查目录权限后重试"}, 500)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.manager = manager
    server.microduck = duck
    return server


def main():
    parser = argparse.ArgumentParser(description="RL Workbench — local PPO training")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", default="runs")
    args = parser.parse_args()
    server = create_server(args.port, args.data_dir)
    print(f"RL Workbench: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("正在保存模型并退出…", flush=True)
    finally:
        server.microduck.close()
        server.manager.stop_event.set()
        server.manager.pause_event.clear()
        if server.manager.worker:
            server.manager.worker.join(timeout=30)
        server.server_close()


if __name__ == "__main__":
    main()
