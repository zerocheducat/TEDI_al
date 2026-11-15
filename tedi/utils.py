# -*- coding: utf-8 -*-


from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import torch

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def deterministic_stream_seed(base_seed: int, stream_id: int, index: int) -> int:
    sequence = np.random.SeedSequence([int(base_seed), int(stream_id), int(index)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])

def make_env(env_name: str, seed: int) -> gym.Env:
    import gymnasium as gym
    try:
        env = gym.make(env_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create {env_name}. "
            "Check the installed Gymnasium, MuJoCo, or Box2D version."
        ) from exc
    env.action_space.seed(seed)
    env.reset(seed=seed)
    return env

def safe_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name).strip(".")
    return cleaned or "unnamed_run"

def write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
