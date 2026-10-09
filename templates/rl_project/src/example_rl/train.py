"""Small real PPO example; replace task logic, retain the v1 entry signature."""
import json
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from .tasks.balance.env_cfg import make_env


class Metrics(BaseCallback):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

    def _on_step(self):
        infos = self.locals['infos']
        components = {name: sum(info['reward_components'][name] for info in infos)/len(infos)
                      for name in infos[0]['reward_components']}
        row = {'step':int(self.num_timesteps), 'reward_mean':float(self.locals['rewards'].mean()),
               'components_mean':components}
        self.stream.write(json.dumps(row, allow_nan=False)+'\n')
        self.stream.flush()
        return True


def train(config, output_dir):
    torch.set_num_threads(1)
    env = DummyVecEnv([lambda: make_env(config) for _ in range(config['num_envs'])])
    folder = Path(output_dir)
    try:
        network = config['network']
        model = PPO('MlpPolicy', env, n_steps=64, batch_size=32, n_epochs=2,
                    seed=config['seed'], device='cpu', verbose=0,
                    policy_kwargs={'net_arch':{'pi':network['actor'], 'vf':network['critic']},
                                   'activation_fn':{'tanh':torch.nn.Tanh,'relu':torch.nn.ReLU}[network['activation']]})
        with (folder/'metrics.jsonl').open('w',encoding='utf-8') as stream:
            model.learn(total_timesteps=config['total_timesteps'], callback=Metrics(stream))
        model.save(folder/'model.pending.zip')
        (folder/'model.pending.zip').replace(folder/'checkpoint.bin')
    finally:
        env.close()
