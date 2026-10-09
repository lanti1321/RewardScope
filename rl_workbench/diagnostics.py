"""Framework-neutral PyTorch probes. No MicroDuck names or shapes are assumed."""
import torch


class RewardProbe:
    """Exact cumulative weighted nonzero counts across all supplied environment steps."""
    def __init__(self, names, device):
        self.names = list(names)
        n = len(names)
        self.count = 0
        self.nonzero = torch.zeros(n, device=device, dtype=torch.float64)
        self.enabled = torch.zeros_like(self.nonzero)
        self.total = torch.zeros_like(self.nonzero)
        self.raw_total = torch.zeros_like(self.nonzero)

    def update(self, contributions, weights):
        values = contributions.detach().double()
        w = values.new_tensor([weights[n] for n in self.names])
        active = w != 0
        self.count += values.shape[0]
        self.nonzero += (values.abs() > 1e-8).sum(0)
        self.enabled += active * values.shape[0]
        self.total += values.sum(0)
        # values are dt-scaled; caller passes weights including dt for raw recovery.
        self.raw_total += (values / torch.where(active, w, torch.ones_like(w))).sum(0) * active

    def snapshot(self):
        packed = torch.stack([self.nonzero, self.enabled, self.total, self.raw_total]).cpu().tolist()
        return {name: {'samples': self.count, 'nonzero_count': int(packed[0][i]), 'evaluated_samples': int(packed[1][i]), 'frequency': packed[0][i] / max(1, self.count), 'mean_contribution': packed[2][i] / max(1, self.count), 'raw_mean_when_enabled': packed[3][i] / packed[1][i] if packed[1][i] else None} for i, name in enumerate(self.names)}


class EpisodeRewardProbe:
    """Component returns for the same completed-episode window as the training logger."""
    def __init__(self, names, num_envs, device, window=100):
        from collections import deque
        self.names = list(names)
        self.sums = torch.zeros((num_envs, len(names)), device=device, dtype=torch.float64)
        self.episodes = deque(maxlen=window)
        self.window = window

    def update(self, contributions, dones):
        self.sums += contributions.detach().double()
        ids = (dones > 0).nonzero(as_tuple=False).flatten()
        if ids.numel():
            self.episodes.extend(self.sums[ids].cpu().tolist())
            self.sums[ids] = 0

    def snapshot(self, logged_total=None):
        if not self.episodes:
            return None
        count = len(self.episodes)
        means = {name: sum(row[i] for row in self.episodes) / count for i, name in enumerate(self.names)}
        total = sum(means.values())
        return {'episodes': count, 'window': self.window, 'components': means,
                'component_total': total, 'logged_total': logged_total,
                'residual': logged_total - total if logged_total is not None else None}
