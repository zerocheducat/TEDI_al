# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .config import EnvironmentConfig
from .networks import RobustActor, RobustCritic
from .replay import ReplayBuffer

class TD3:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        config: EnvironmentConfig,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        self.max_action = float(max_action)
        use_layernorm = bool(getattr(config, "use_layernorm", True))
        self.actor = RobustActor(state_dim, action_dim, max_action, use_layernorm=use_layernorm).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.ema_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.current_actor_lr = float(config.actor_lr)
        self.critic = RobustCritic(state_dim, action_dim, use_layernorm=use_layernorm).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.total_it = 0

    @torch.inference_mode()
    def select_action(
        self,
        state: np.ndarray,
        noise_std: float = 0.0,
        rng: Optional[np.random.Generator] = None,
        actor: Optional[nn.Module] = None,
    ) -> np.ndarray:
        policy = self.actor if actor is None else actor
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        action = policy(state_tensor).cpu().numpy().reshape(-1)
        if noise_std > 0.0:
            generator = rng if rng is not None else np.random.default_rng()
            action = action + generator.normal(0.0, noise_std, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    @staticmethod
    def _huber_from_error(error: torch.Tensor, delta: float) -> torch.Tensor:
        absolute_error = error.abs()
        quadratic = torch.minimum(absolute_error, torch.full_like(absolute_error, float(delta)))
        linear = absolute_error - quadratic
        return (0.5 * quadratic.square() + float(delta) * linear).mean()

    def train(self, replay_buffer: ReplayBuffer, main_step: int) -> Dict[str, float]:
        self.total_it += 1
        state, action, reward, next_state, terminal = replay_buffer.sample(self.config.batch_size)

        with torch.no_grad():
            noise = torch.randn_like(action) * self.config.policy_noise
            noise = noise.clamp(-self.config.noise_clip, self.config.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = reward + (1.0 - terminal) * self.config.gamma * torch.minimum(target_q1, target_q2)
            if self.config.target_q_min is not None:
                target_q = torch.maximum(target_q, torch.tensor(self.config.target_q_min, device=self.device))
            if self.config.target_q_max is not None:
                target_q = torch.minimum(target_q, torch.tensor(self.config.target_q_max, device=self.device))

        current_q1, current_q2 = self.critic(state, action)
        td1 = current_q1 - target_q
        td2 = current_q2 - target_q
        if self.config.critic_loss_mode == "huber":
            critic_loss = self._huber_from_error(td1, self.config.critic_huber_delta)
            critic_loss = critic_loss + self._huber_from_error(td2, self.config.critic_huber_delta)
        elif self.config.critic_loss_mode == "mse":
            critic_loss = td1.square().mean() + td2.square().mean()
        else:
            if self.config.td_error_clip is not None:
                clip = float(self.config.td_error_clip)
                td1 = td1.clamp(-clip, clip)
                td2 = td2.clamp(-clip, clip)
            critic_loss = td1.square().mean() + td2.square().mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.config.critic_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.critic_grad_clip)
        self.critic_optimizer.step()

        actor_loss_value = float("nan")
        if self.total_it % self.config.policy_freq == 0:
            actor_lr = self.config.actor_learning_rate(main_step)
            if actor_lr != self.current_actor_lr:
                for parameter_group in self.actor_optimizer.param_groups:
                    parameter_group["lr"] = actor_lr
                self.current_actor_lr = actor_lr
            policy_action = self.actor(state)
            actor_loss = -self.critic.q1(state, policy_action).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            if self.config.actor_grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.actor_grad_clip)
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.item())

            with torch.no_grad():
                for parameter, target_parameter in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_parameter.mul_(1.0 - self.config.tau).add_(parameter, alpha=self.config.tau)
                for parameter, target_parameter in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_parameter.mul_(1.0 - self.config.tau).add_(parameter, alpha=self.config.tau)
                if self.config.use_ema_actor:
                    for parameter, ema_parameter in zip(self.actor.parameters(), self.ema_actor.parameters()):
                        ema_parameter.mul_(self.config.ema_beta).add_(parameter, alpha=1.0 - self.config.ema_beta)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": actor_loss_value,
            "target_q_mean": float(target_q.mean().item()),
            "actor_learning_rate": self.current_actor_lr,
        }

    def checkpoint(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "ema_actor": self.ema_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_it": self.total_it,
            "current_actor_lr": self.current_actor_lr,
        }
