import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen

from rl_workbench.project_contract import inspect_project, resolve_config, find_project_root
from rl_workbench.project import verify_outputs
from rl_workbench.discovery import scan_project
from rl_workbench.server import create_server

TEMPLATE = Path(__file__).resolve().parents[1] / 'templates/rl_project'


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'project'
        shutil.copytree(TEMPLATE, self.root, ignore=shutil.ignore_patterns('__pycache__'))
        self.manifest = json.loads((self.root/'rl-project.json').read_text())

    def write(self, value):
        (self.root/'rl-project.json').write_text(json.dumps(value))

    def test_valid_project_and_scan_never_executes(self):
        entry = self.root/'src/example_rl/train.py'
        entry.write_text('raise RuntimeError("must not import")\n'+ '\n'.join(entry.read_text().splitlines()))
        report = inspect_project(self.root)
        self.assertTrue(report['valid'], report)
        self.assertFalse(report['runtime_verified'])
        self.assertTrue(scan_project(self.root)['contract']['valid'])
        self.assertEqual(len(report['tasks'][0]['sources']['rewards']), 2)

    def test_strict_manifest_rejections(self):
        edits = [lambda m:m.update(schema_version=2), lambda m:m.update(unknown=True),
                 lambda m:m.update(source_root='../outside'),
                 lambda m:m['tasks'].append(copy.deepcopy(m['tasks'][0])),
                 lambda m:m['tasks'][0]['rewards'].append(copy.deepcopy(m['tasks'][0]['rewards'][0])),
                 lambda m:m['tasks'][0]['defaults'].update(num_envs=True),
                 lambda m:m['tasks'][0]['defaults']['network'].update(actor=[0]),
                 lambda m:m['tasks'][0]['rewards'][0].update(weight=float('nan')),
                 lambda m:m['tasks'][0].update(train='absent.module:train')]
        for edit in edits:
            with self.subTest(edit=edit):
                value=copy.deepcopy(self.manifest);edit(value);self.write(value)
                self.assertFalse(inspect_project(self.root)['valid'])
                self.assertFalse(scan_project(self.root)['contract']['valid'])
        (self.root/'rl-project.json').write_text('{"schema_version":1,"schema_version":1}')
        self.assertIn('duplicate', inspect_project(self.root)['errors'][0])

    def test_signature_and_symlink(self):
        entry=self.root/'src/example_rl/train.py'
        entry.write_text('def train(config):\n    pass\n')
        self.assertFalse(inspect_project(self.root)['valid'])
        outside=Path(self.temp.name)/'external.py'
        outside.write_text('def train(config, output_dir):\n    pass\n')
        entry.unlink();entry.symlink_to(outside)
        self.assertFalse(inspect_project(self.root)['valid'])

    def test_parent_discovery_and_required_package(self):
        nested = self.root/'tools/workbench'
        nested.mkdir(parents=True)
        self.assertEqual(find_project_root(nested), self.root)
        (self.root/'pyproject.toml').unlink()
        self.assertFalse(inspect_project(self.root)['valid'])

    def test_invalid_runtime_metrics_rejected(self):
        config=resolve_config(inspect_project(self.root),'CartPole-Balance-v1',{'total_timesteps':2,'disabled_rewards':['angle']})
        output=Path(self.temp.name)/'output';output.mkdir()
        (output/'checkpoint.bin').write_bytes(b'test artifact')
        for row in ({'step':2,'reward_mean':1,'components_mean':{'alive':2,'angle':-1}},
                    {'step':2,'reward_mean':3,'components_mean':{'alive':2,'angle':0}},
                    {'step':1,'reward_mean':2,'components_mean':{'alive':2,'angle':0}}):
            (output/'metrics.jsonl').write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):
                verify_outputs(output,config)

    def test_config_no_mutation_mask_and_unknowns(self):
        report=inspect_project(self.root)
        original=copy.deepcopy(report)
        config=resolve_config(report,'CartPole-Balance-v1',{'reward_weights':{'angle':-0.5},'disabled_rewards':['angle']})
        self.assertEqual(config['rewards'][1]['weight'],0)
        self.assertEqual(config['rewards'][1]['mode'],'disabled')
        self.assertEqual(report, original)
        for override in ({'iterations':5},{'reward_weights':{'missing':1}}, {'seed':True}, {'disabled_rewards':['angle','angle']}, {'reward_weights':{'alive':float('inf')}}):
            with self.assertRaises(ValueError):
                resolve_config(report,'CartPole-Balance-v1',override)

    def test_http_scan(self):
        server=create_server(0,Path(self.temp.name)/'runs')
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            request=Request(f'http://127.0.0.1:{server.server_port}/api/projects/scan',
                            data=json.dumps({'directory':str(self.root)}).encode(),headers={'Content-Type':'application/json'})
            with urlopen(request) as response:
                self.assertTrue(json.load(response)['valid'])
        finally:
            server.shutdown();server.server_close();thread.join()

    def test_real_training_and_checkpoint_load(self):
        output=Path(self.temp.name)/'new-run'
        override=Path(self.temp.name)/'config.json'
        override.write_text(json.dumps({'total_timesteps':128,'reward_weights':{'alive':2},'disabled_rewards':['angle']}))
        command=[sys.executable,'-m','rl_workbench.project','train',str(self.root),'--task','CartPole-Balance-v1','--config',str(override),'--output-dir',str(output)]
        result=subprocess.run(command,capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(json.loads((output/'status.json').read_text())['status'],'completed')
        rows=[json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
        self.assertEqual(rows[-1]['step'],128)
        self.assertTrue(all(row['components_mean']=={'alive':2.0,'angle':0.0} for row in rows))
        from stable_baselines3 import PPO
        import gymnasium as gym
        import numpy as np
        model=PPO.load(output/'checkpoint.bin',device='cpu')
        env=gym.make('CartPole-v1')
        try:
            obs,_=env.reset(seed=1)
            action,_=model.predict(obs,deterministic=True)
            self.assertTrue(np.isfinite(action).all())
            self.assertTrue(env.action_space.contains(action))
        finally:
            env.close()
        repeated=subprocess.run(command,capture_output=True,text=True,timeout=20)
        self.assertNotEqual(repeated.returncode,0)
        self.assertEqual(json.loads((output/'status.json').read_text())['status'],'completed')


if __name__ == '__main__':
    unittest.main()
