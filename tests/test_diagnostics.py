import tempfile
import unittest
from pathlib import Path
import torch
from rl_workbench.diagnostics import RewardProbe
from rl_workbench.discovery import scan_project
from rl_workbench.microduck import validate_microduck


class DiagnosticsTests(unittest.TestCase):
    def test_episode_components_share_window_and_reset_per_environment(self):
        from rl_workbench.diagnostics import EpisodeRewardProbe
        p=EpisodeRewardProbe(['reward','penalty'],2,'cpu',window=2)
        p.update(torch.tensor([[10.,-2.],[20.,-4.]]),torch.tensor([0,1]))
        self.assertEqual(p.snapshot(16.)['components'],{'reward':20.,'penalty':-4.})
        p.update(torch.tensor([[5.,-1.],[8.,-2.]]),torch.tensor([1,1]))
        result=p.snapshot(9.)
        self.assertEqual(result['components'],{'reward':11.5,'penalty':-2.5})
        self.assertEqual(result['residual'],0.)
        self.assertEqual(result['episodes'],2)

    def test_fixed_weights_mask_and_curriculum_precedence(self):
        from rl_workbench.reward_settings import resolve_reward_weights
        cfg = {'reward_weights': {'a': -.25, 'b': 2.}, 'disabled_rewards': ['b']}
        self.assertEqual(resolve_reward_weights({'a': 3., 'b': 1., 'c': .5}, cfg), {'a': -.25, 'b': 0., 'c': .5})
        self.assertEqual(resolve_reward_weights({'a': 7., 'b': 5., 'c': .9}, cfg), {'a': -.25, 'b': 0., 'c': .9})
        cat={'tasks':[{'id':'Mjlab-Velocity-Flat-MicroDuck','rewards':[{'name':'a'}]}]}
        for patch in ({'reward_weights': {'a':float('nan')}}, {'reward_weights':{'unknown':1}}, {'disabled_rewards':['unknown']}, {'disabled_rewards':[{}]}):
            with self.assertRaises(ValueError):validate_microduck(patch,cat)

    def test_reward_denominator_zero_weight_and_raw_value(self):
        p = RewardProbe(['enabled', 'disabled'], 'cpu')
        p.update(torch.tensor([[.2, 0.], [0., 0.]]), {'enabled': .1, 'disabled': 0.})
        s = p.snapshot()
        self.assertEqual(s['enabled']['frequency'], .5)
        self.assertAlmostEqual(s['enabled']['raw_mean_when_enabled'], 1.)
        self.assertIsNone(s['disabled']['raw_mean_when_enabled'])
        p.update(torch.tensor([[0., -.4], [0., 0.]]), {'enabled': 0., 'disabled': -.2})
        s = p.snapshot()
        self.assertEqual(s['disabled']['frequency'], .25)
        self.assertEqual(s['disabled']['evaluated_samples'], 2)
        self.assertAlmostEqual(s['disabled']['raw_mean_when_enabled'], 1.)

    def test_scanner_does_not_execute_and_skips_venv_and_symlinks(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'task.py').write_text('raise RuntimeError("do not execute")\ndef reward_goal(x):\n return x\n')
            (root/'.venv').mkdir(); (root/'.venv/x.py').write_text('def reward_fake(): pass')
            (root/'link.py').symlink_to(root/'task.py')
            result=scan_project(root)
            self.assertEqual(result['files'], 1)
            self.assertEqual(result['candidates'][0]['line'], 2)
            self.assertEqual(result['errors'], [])

    def test_network_validation(self):
        cat={'tasks':[{'id':'Mjlab-Velocity-Flat-MicroDuck','rewards':[], 'network':{'actor':{'class_name':'MLPModel'}}}]}
        self.assertEqual(validate_microduck({'network':{'actor':[64,32]}},cat)['network']['actor'],[64,32])
        for value in ([True],[],[4096],[32.5]):
            with self.assertRaises(ValueError):validate_microduck({'network':{'actor':value}},cat)
