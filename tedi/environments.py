# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np

from .utils import make_env

class ShadowEnvPool:

    def __init__(self, env_name: str, count: int, seed_base: int) -> None:
        self.envs = [make_env(env_name, seed_base + i) for i in range(count)]
        self.count = count

    def reset(self, common_seed: int) -> np.ndarray:
        states = [env.reset(seed=common_seed)[0] for env in self.envs]
        return np.asarray(states, dtype=np.float32)

    def close(self) -> None:
        for env in self.envs:
            env.close()
