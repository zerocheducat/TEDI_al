# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustActor(nn.Module):
    """Actor network with optional layer normalization."""


    def __init__(self, state_dim: int, action_dim: int, max_action: float, use_layernorm: bool = True) -> None:
        super().__init__()
        self.use_layernorm = bool(use_layernorm)
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.ln1 = nn.LayerNorm(256) if self.use_layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(256) if self.use_layernorm else nn.Identity()
        self.max_action = float(max_action)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        action = F.relu(self.ln1(self.l1(state)))
        action = F.relu(self.ln2(self.l2(action)))
        return self.max_action * torch.tanh(self.l3(action))

class RobustCritic(nn.Module):
    """Twin critic networks with optional layer normalization."""


    def __init__(self, state_dim: int, action_dim: int, use_layernorm: bool = True) -> None:
        super().__init__()
        self.use_layernorm = bool(use_layernorm)
        input_dim = state_dim + action_dim
        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.l4 = nn.Linear(input_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
        self.ln1 = nn.LayerNorm(256) if self.use_layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(256) if self.use_layernorm else nn.Identity()
        self.ln4 = nn.LayerNorm(256) if self.use_layernorm else nn.Identity()
        self.ln5 = nn.LayerNorm(256) if self.use_layernorm else nn.Identity()

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state, action), self.q2(state, action)

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_action = torch.cat([state, action], dim=1)
        return self.l3(F.relu(self.ln2(self.l2(F.relu(self.ln1(self.l1(state_action)))))))

    def q2(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_action = torch.cat([state, action], dim=1)
        return self.l6(F.relu(self.ln5(self.l5(F.relu(self.ln4(self.l4(state_action)))))))
