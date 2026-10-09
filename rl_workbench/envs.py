"""Reward instrumentation. All terms are computed on the resulting observation."""

import gymnasium as gym

DEFAULT_WEIGHTS = {"alive": 1.0, "angle": 0.1, "position": 0.05, "failure": 1.0}


class RewardCartPole(gym.Wrapper):
    def __init__(self, weights=None):
        super().__init__(gym.make("CartPole-v1"))
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.previous_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_obs = obs.copy()
        return obs, info

    def step(self, action):
        before = self.previous_obs.copy()
        obs, raw_reward, terminated, truncated, info = self.env.step(action)
        features = {
            "alive": float(raw_reward),
            "angle": -float(obs[2] / self.unwrapped.theta_threshold_radians) ** 2,
            "position": -float(obs[0] / self.unwrapped.x_threshold) ** 2,
            "failure": -float(terminated),
        }
        components = {key: value * self.weights[key] for key, value in features.items()}
        reward = sum(components.values())
        info.update({
            "before": before.tolist(), "after": obs.tolist(), "action": int(action),
            "raw_reward": float(raw_reward), "features": features,
            "components": components, "reward": reward,
            "terminated": bool(terminated), "truncated": bool(truncated),
        })
        self.previous_obs = obs.copy()
        return obs, reward, terminated, truncated, info
