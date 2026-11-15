# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

Transition = Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]

class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_size: int,
        storage_device: torch.device,
        train_device: torch.device,
    ) -> None:
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.storage_device = storage_device
        self.train_device = train_device
        self.state = torch.empty((max_size, state_dim), dtype=torch.float32, device=storage_device)
        self.action = torch.empty((max_size, action_dim), dtype=torch.float32, device=storage_device)
        self.reward = torch.empty((max_size, 1), dtype=torch.float32, device=storage_device)
        self.next_state = torch.empty((max_size, state_dim), dtype=torch.float32, device=storage_device)
        self.terminal = torch.empty((max_size, 1), dtype=torch.float32, device=storage_device)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        terminal: float,
    ) -> None:
        index = self.ptr
        self.state[index].copy_(torch.as_tensor(state, dtype=torch.float32, device=self.storage_device))
        self.action[index].copy_(torch.as_tensor(action, dtype=torch.float32, device=self.storage_device))
        self.reward[index, 0] = float(reward)
        self.next_state[index].copy_(torch.as_tensor(next_state, dtype=torch.float32, device=self.storage_device))
        self.terminal[index, 0] = float(terminal)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        index = torch.randint(0, self.size, (batch_size,), device=self.storage_device)
        tensors = (
            self.state[index],
            self.action[index],
            self.reward[index],
            self.next_state[index],
            self.terminal[index],
        )
        if self.storage_device == self.train_device:
            return tensors
        return tuple(item.to(self.train_device, non_blocking=True) for item in tensors)
