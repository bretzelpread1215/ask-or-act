import numpy as np
from numpy.random import Generator


def replicate_streams(master_seed: int, replicate_index: int) -> tuple[Generator, Generator]:
    """return separate task and confidence-noise streams for one replicate."""
    if master_seed < 0 or replicate_index < 0:
        raise ValueError("master_seed and replicate_index must be nonnegative")

    replicate_seed = np.random.SeedSequence([master_seed, replicate_index])
    task_seed, noise_seed = replicate_seed.spawn(2)
    return np.random.default_rng(task_seed), np.random.default_rng(noise_seed)
