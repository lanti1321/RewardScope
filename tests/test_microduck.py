import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

from rl_workbench.microduck import MicroduckManager, validate_microduck
from rl_workbench.training import atomic_json
from rl_workbench.microduck_bridge import save_stopped_checkpoint

CATALOG = {"tasks": [{"id": "Mjlab-Velocity-Flat-MicroDuck", "rewards": [{"name": "action_rate_l2", "weight": -.1}]}]}


class MicroduckTests(unittest.TestCase):
    def test_failed_stop_save_does_not_publish_partial_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "model_stopped.pt"
            target.write_bytes(b"previous valid checkpoint")
            def fail(path):
                Path(path).write_bytes(b"partial")
                raise RuntimeError("CUDA error")
            with self.assertRaises(RuntimeError):
                save_stopped_checkpoint(Mock(save=fail), root)
            self.assertEqual(target.read_bytes(), b"previous valid checkpoint")
            self.assertEqual(list(Path(root).iterdir()), [target])
            save_stopped_checkpoint(Mock(save=lambda p: Path(p).write_bytes(b"complete")), root)
            self.assertEqual(target.read_bytes(), b"complete")

    def test_preserve_multiplier_and_budget(self):
        cfg = validate_microduck({"reward_scales": {"action_rate_l2": 2}}, CATALOG)
        self.assertEqual(cfg["num_envs"], 64)
        self.assertEqual(cfg["iterations"], 5)
        self.assertEqual(cfg["reward_scales"]["action_rate_l2"], 2)
        for data in ({"task": "unknown"}, {"iterations": 1.5}, {"num_envs": 4097},
                     {"reward_scales": {"typo": 1}}, {"reward_scales": {"action_rate_l2": -1}},
                     {"reward_scales": {"action_rate_l2": float('nan')}}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                validate_microduck(data, CATALOG)

    def test_restart_marks_interrupted_and_retains_data(self):
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root)/"123456abcdef"
            folder.mkdir()
            atomic_json(folder/"run.json", {"id": folder.name, "status": "running", "history": [{"iteration": 3}]})
            manager=MicroduckManager(root)
            result=manager.get(folder.name)
            self.assertEqual(result["status"], "interrupted")
            self.assertEqual(result["history"], [{"iteration": 3}])

    def test_large_run_can_start_without_previous_records(self):
        with tempfile.TemporaryDirectory() as root:
            manager = MicroduckManager(root)
            manager.catalog = lambda: {"ready": True, "source": {}, **CATALOG}
            process = Mock()
            process.poll.return_value = None
            with patch("rl_workbench.microduck.subprocess.Popen", return_value=process) as launch:
                run = manager.start({"iterations": 100000, "num_envs": 4096})
            launch.assert_called_once()
            self.assertEqual(run["config"]["iterations"], 100000)
            self.assertEqual(run["config"]["num_envs"], 4096)
            self.assertEqual(run["status"], "starting")


if __name__ == '__main__':
    unittest.main()
