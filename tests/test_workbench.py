import json
import math
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from rl_workbench.envs import DEFAULT_WEIGHTS, RewardCartPole
from rl_workbench.server import create_server
from rl_workbench.training import RunManager, validate_config


class RewardTests(unittest.TestCase):
    def test_reward_and_transition_alignment(self):
        env = RewardCartPole()
        before, _ = env.reset(seed=2)
        after, reward, _, _, info = env.step(1)
        self.assertEqual(before.tolist(), info["before"])
        self.assertEqual(after.tolist(), info["after"])
        self.assertAlmostEqual(reward, sum(info["components"].values()))
        self.assertAlmostEqual(info["components"]["angle"], -DEFAULT_WEIGHTS["angle"] * (after[2] / env.unwrapped.theta_threshold_radians) ** 2)
        env.close()

    def test_failure_and_time_limit_differ(self):
        env = RewardCartPole()
        env.reset(seed=0)
        env.unwrapped.state = (2.5, 0, 0, 0)
        _, _, terminated, _, info = env.step(0)
        self.assertTrue(terminated)
        self.assertEqual(info["components"]["failure"], -1)
        env.reset(seed=0)
        env.env._elapsed_steps = 499
        _, _, terminated, truncated, info = env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["components"]["failure"], 0)
        env.close()

    def test_invalid_configs(self):
        for patch in ({"total_steps":257},{"seed":1.5},{"learning_rate":float('nan')},{"name":" "},{"weights":{}},{"weights":dict(DEFAULT_WEIGHTS,alive=float('inf'))}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate_config(patch)


class IntegrationTests(unittest.TestCase):
    def test_training_replay_checkpoint_restart_and_controls(self):
        from stable_baselines3 import PPO
        with tempfile.TemporaryDirectory() as root:
            manager = RunManager(root)
            run = manager.start({"name":"smoke", "total_steps":512})
            manager.worker.join(45)
            self.assertFalse(manager.worker.is_alive())
            final = manager.get(run["id"])
            self.assertEqual(final["status"], "completed", final["error"])
            self.assertEqual(final["step"], 512)
            self.assertTrue(final["episodes"])
            self.assertTrue(final["model_saved"])
            self.assertTrue(final["best_model_saved"])
            self.assertIn("approx_kl", final["updates"][-1])
            self.assertEqual(final["updates"][-1]["step"], 512)
            self.assertEqual([e["step"] for e in final["evaluations"]], [0, 512])
            trajectory = json.loads((Path(root)/run["id"]/"eval-1.json").read_text())
            model = PPO.load(Path(root)/run["id"]/"model.zip", device="cpu")
            env = RewardCartPole(final["config"]["weights"])
            obs, _ = env.reset(seed=1000)
            for frame in trajectory["episodes"][0]["frames"]:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, _, _, info = env.step(int(action))
                self.assertEqual(frame["action"], int(action))
                self.assertEqual(frame["after"], obs.tolist())
                self.assertAlmostEqual(frame["reward"], reward)
            env.close()
            recovered = RunManager(root)
            self.assertEqual(recovered.get(run["id"]), final)

            second = manager.start({"name":"control", "total_steps":51200})
            manager.control(second["id"], "pause")
            limit = time.monotonic()+10
            while manager.get(second["id"])["status"] != "paused" and time.monotonic()<limit:
                time.sleep(.02)
            self.assertEqual(manager.get(second["id"])["status"], "paused")
            step = manager.get(second["id"])["step"]
            time.sleep(.2)
            self.assertEqual(manager.get(second["id"])["step"], step)
            with self.assertRaises(RuntimeError):
                manager.start({"name":"conflict"})
            manager.control(second["id"], "resume")
            manager.control(second["id"], "stop")
            manager.worker.join(10)
            self.assertEqual(manager.get(second["id"])["status"], "stopped")
            self.assertTrue(manager.get(second["id"])["model_saved"])

    def test_http_validation_and_origin(self):
        with tempfile.TemporaryDirectory() as root:
            server=create_server(0, root)
            thread=threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url=f"http://127.0.0.1:{server.server_port}"
            try:
                with urllib.request.urlopen(url+'/api/health') as response:
                    self.assertTrue(json.load(response)["ok"])
                for body, origin, expected in (({"total_steps":1},None,400), ({},'http://evil.example',403)):
                    headers={"Content-Type":"application/json"}
                    if origin:headers["Origin"]=origin
                    request=urllib.request.Request(url+'/api/runs',data=json.dumps(body).encode(),headers=headers)
                    with self.assertRaises(urllib.error.HTTPError) as cm:
                        urllib.request.urlopen(request)
                    self.assertEqual(cm.exception.code, expected)
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    urllib.request.urlopen(url+'/../requirements.txt')
                self.assertEqual(cm.exception.code,404)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == '__main__':
    unittest.main()
