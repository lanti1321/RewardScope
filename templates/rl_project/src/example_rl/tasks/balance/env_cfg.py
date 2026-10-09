"""Reward weights come only from the resolved configuration."""
import importlib
import gymnasium as gym


class RewardEnv(gym.Wrapper):
    def __init__(self, config):
        super().__init__(gym.make('CartPole-v1'))
        self.terms = []
        for reward in config['rewards']:
            module, name = reward['function'].split(':')
            fn = getattr(importlib.import_module(module), name)
            self.terms.append((reward['name'], fn, reward['weight']))
        self.scale = config['step_dt'] if config['reward_scale']=='dt' else 1.0

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        transition = {'observation':obs, 'action':action, 'base_reward':base_reward,
                      'terminated':terminated, 'truncated':truncated}
        components = {name: fn(transition)*weight*self.scale if weight else 0.0
                      for name,fn,weight in self.terms}
        info['reward_components'] = components
        return obs, sum(components.values()), terminated, truncated, info


def make_env(config):
    return RewardEnv(config)
