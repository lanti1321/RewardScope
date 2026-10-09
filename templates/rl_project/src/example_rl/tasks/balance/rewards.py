"""Unweighted scalar rewards of one environment transition."""
def alive(transition):
    return float(transition['base_reward'])


def angle(transition):
    return float(transition['observation'][2]) ** 2
