"""Resolve numeric reward overrides without changing task reward functions."""

def resolve_reward_weights(current, config):
    overrides = config.get('reward_weights', {})
    disabled = set(config.get('disabled_rewards', []))
    scales = config.get('reward_scales', {})  # Existing saved configurations.
    return {name: 0.0 if name in disabled else overrides.get(name, weight * scales.get(name, 1.0))
            for name, weight in current.items()}
