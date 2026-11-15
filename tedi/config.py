# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


# This backbone ablation disables EMA; it is not a plain TD3 implementation.
TD3_BASELINE_OVERRIDES: Dict[str, Any] = {
    "td3_baseline": True,
    "use_ema_actor": False,
    "eval_actor": "actor",
}

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


VANILLA_TD3_OVERRIDES: Dict[str, Any] = {
    "vanilla_td3": True,
    "use_layernorm": False,
    "use_ema_actor": False,
    "eval_actor": "actor",
    "target_q_min": None,
    "target_q_max": None,
    "td_error_clip": None,
    "critic_loss_mode": "mse",
    "critic_grad_clip": None,
    "actor_grad_clip": None,
    "bootstrap_on_truncation": False,
    "updates_per_step": 1,
    "start_steps": 25_000,
    "update_after": 25_000,
    "actor_lr": 3e-4,
    "actor_lr_final": None,
    "critic_lr": 3e-4,
    "noise_initial": 0.1,
    "noise_final": 0.1,
    "noise_decay_steps": 1,
    "replay_size": 1_000_000,
}

def _base_params(**overrides: Any) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "td3_baseline": False,
        "vanilla_td3": False,
        "use_layernorm": True,
        "use_ema_actor": True,
        "target_reward": 0.0,
        "gamma": 0.99,
        "tau": 0.005,
        "ema_beta": 0.99,
        "batch_size": 256,
        "replay_size": 1_000_000,
        "actor_lr": 3e-4,
        "actor_lr_final": None,
        "actor_lr_decay_start_step": 0,
        "actor_lr_decay_steps": 1,
        "critic_lr": 3e-4,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "policy_freq": 2,
        "start_steps": 0,
        "update_after": 256,
        "updates_per_step": 1,
        "bootstrap_on_truncation": True,
        "target_q_min": -1000.0,
        "target_q_max": 7000.0,
        "td_error_clip": 50.0,
        "critic_loss_mode": "clipped_mse",
        "critic_huber_delta": 50.0,
        "critic_grad_clip": 10.0,
        "actor_grad_clip": 5.0,
        "noise_initial": 0.30,
        "noise_final": 0.10,
        "noise_decay_steps": 300_000,
        "stagnation_patience_limit": 10,
        "dynamic_threshold_ratio": 0.05,
        "stagnation_detector": "legacy",
        "stagnation_progress_threshold": 0.0,
        "stagnation_variation_threshold": 1.0,
        "min_mean_reward_to_evolve": -1e9,
        "min_steps_before_evolution": 0,
        "evolution_cooldown_steps": 0,
        "evolution_stop_reward": None,
        "evolution_stop_survival_rate": None,
        "evolution_stop_survival_min_reward": None,
        "sigma_conservative": 0.01,
        "sigma_explorer": 0.05,
        "sigma_destroyer": 0.05,
        "max_inject_per_evo": 500,
        "evolution_horizon": 500,
        "trajectory_keep_quantile": 0.0,
        "trajectory_keep_quantile_conservative": None,
        "trajectory_keep_quantile_explorer": None,
        "trajectory_keep_quantile_destroyer": None,
        "use_parent_return_gate": False,
        "parent_return_tolerance_conservative": 0.0,
        "parent_return_tolerance_explorer": 0.0,
        "parent_return_tolerance_destroyer": 0.0,
        "transition_value_mode": "current_q",
        "mutation_parent": "actor",
        "mutation_style": "legacy",
        "mutation_fraction": 1.0,
        "force_nonzero_mutation": True,
        "penalty_initial": 0.0,
        "penalty_final": 0.0,
        "penalty_decay_steps": 1,
        "eval_actor": "ema",
        "eval_every_episodes": 1,
        "eval_every_steps": None,
        "eval_episodes": 1,
        "eval_seed_mode": "fixed",
        "eval_seed_stride": 1_000,
        "final_eval_episodes": 10,
        "recent_window": 5,
        "checkpoint_window": 5,
        "save_every_episodes": 25,
        "default_steps": 1_000_000,
        "fn_check_stagnation": lambda rm, om: rm <= om,
        "fn_qualify_conservative": lambda r, q, mr, bt, term: q > bt,
        "fn_qualify_explorer": lambda r, q, mr, bt, term: q > bt,
        "fn_qualify_destroyer": lambda r, q, mr, bt, term: q > bt,
    }
    params.update(overrides)
    return params

ENV_PARAMS: Dict[str, Dict[str, Any]] = {
    "Ant-v2": _base_params(
        eval_actor="actor",
        target_reward=7000.0,
        replay_size=1_000_000,
        stagnation_patience_limit=10,
        dynamic_threshold_ratio=0.05,
        min_mean_reward_to_evolve=-1200.0,
        sigma_conservative=0.01,
        sigma_explorer=0.05,
        sigma_destroyer=0.25,
        max_inject_per_evo=500,
        evolution_horizon=500,
        noise_initial=0.30,
        noise_final=0.10,
        noise_decay_steps=300_000,
        force_nonzero_mutation=False,
        penalty_initial=50.0,
        penalty_final=0.0,
        penalty_decay_steps=800_000,
        fn_check_stagnation=lambda rm, om: rm <= om + abs(om) * 0.02,
        fn_qualify_conservative=lambda r, q, mr, bt, term: q > bt and r >= mr,
        fn_qualify_explorer=lambda r, q, mr, bt, term: q > bt * 0.9 and r > 0 and np.random.rand() < 0.5,
        fn_qualify_destroyer=lambda r, q, mr, bt, term: r > mr * 2.5 or r > 5.0,
    ),

    "HalfCheetah-v2": _base_params(
        target_reward=14000.0,
        replay_size=350_000,
        gamma=0.99,
        actor_lr=3e-4,
        critic_lr=3e-4,
        start_steps=10_000,
        update_after=10_000,
        updates_per_step=2,
        target_q_min=None,
        target_q_max=None,
        td_error_clip=None,
        stagnation_patience_limit=3,
        dynamic_threshold_ratio=0.015,
        stagnation_detector="normalized_windows",
        stagnation_progress_threshold=0.025,
        stagnation_variation_threshold=0.015,
        min_mean_reward_to_evolve=5500.0,
        min_steps_before_evolution=180_000,
        evolution_cooldown_steps=75_000,
        evolution_stop_reward=13750.0,
        sigma_conservative=0.008,
        sigma_explorer=0.022,
        sigma_destroyer=0.030,
        max_inject_per_evo=200,
        evolution_horizon=1000,
        trajectory_keep_quantile=0.5,
        trajectory_keep_quantile_conservative=0.67,
        trajectory_keep_quantile_explorer=0.67,
        trajectory_keep_quantile_destroyer=0.67,
        use_parent_return_gate=True,
        parent_return_tolerance_conservative=0.0,
        parent_return_tolerance_explorer=50.0,
        parent_return_tolerance_destroyer=100.0,
        transition_value_mode="one_step_target",
        mutation_parent="ema",
        noise_initial=0.15,
        noise_final=0.08,
        noise_decay_steps=840_000,
        penalty_initial=0.0,
        penalty_final=0.0,
        penalty_decay_steps=1_000_000,
        eval_every_steps=5_000,
        eval_episodes=3,
        eval_seed_mode="rotating",
        final_eval_episodes=20,
        checkpoint_window=10,
        fn_check_stagnation=lambda rm, om: rm <= om,
        fn_qualify_conservative=lambda r, q, mr, bt, term: r >= mr and q >= bt,
        fn_qualify_explorer=lambda r, q, mr, bt, term: not term and r >= mr * 0.75 and q >= bt * 0.90,
        fn_qualify_destroyer=lambda r, q, mr, bt, term: not term and r >= mr * 0.60 and q >= bt * 0.80,
    ),
    "Hopper-v2": _base_params(
        target_reward=3650.0,
        replay_size=300_000,
        actor_lr=1e-4,
        actor_lr_final=None,
        actor_lr_decay_start_step=0,
        actor_lr_decay_steps=1,
        critic_lr=3e-4,
        ema_beta=0.995,
        start_steps=10_000,
        update_after=10_000,
        critic_loss_mode="huber",
        critic_huber_delta=50.0,
        stagnation_patience_limit=6,
        dynamic_threshold_ratio=0.04,
        min_mean_reward_to_evolve=500.0,
        min_steps_before_evolution=50_000,
        evolution_cooldown_steps=50_000,
        evolution_stop_reward=3650.0,
        evolution_stop_survival_rate=0.90,
        evolution_stop_survival_min_reward=3400.0,
        sigma_conservative=0.005,
        sigma_explorer=0.015,
        sigma_destroyer=0.04,
        max_inject_per_evo=80,
        evolution_horizon=1000,
        trajectory_keep_quantile=0.5,
        use_parent_return_gate=True,
        parent_return_tolerance_conservative=0.0,
        parent_return_tolerance_explorer=100.0,
        parent_return_tolerance_destroyer=250.0,
        mutation_parent="ema",
        mutation_style="output_masked",
        mutation_fraction=0.2,
        noise_initial=0.120,
        noise_final=0.020,
        noise_decay_steps=700_000,
        penalty_initial=10.0,
        penalty_final=0.5,
        penalty_decay_steps=600_000,
        eval_every_steps=5_000,
        eval_episodes=5,
        final_eval_episodes=20,
        checkpoint_window=5,
        fn_check_stagnation=lambda rm, om: rm <= om + 10.0,
        fn_qualify_conservative=lambda r, q, mr, bt, term: r >= mr and q > bt,
        fn_qualify_explorer=lambda r, q, mr, bt, term: r > mr * 0.8 and q > bt * 0.8,
        fn_qualify_destroyer=lambda r, q, mr, bt, term: r > mr * 0.9 and not term,
    ),
    "Swimmer-v2": _base_params(
        target_reward=120.0,
        gamma=0.999,
        replay_size=500_000,
        actor_lr=1e-4,
        critic_lr=3e-4,
        start_steps=10_000,
        update_after=10_000,
        target_q_min=None,
        target_q_max=None,
        td_error_clip=None,
        stagnation_patience_limit=15,
        dynamic_threshold_ratio=0.05,
        min_mean_reward_to_evolve=20.0,
        min_steps_before_evolution=50_000,
        evolution_cooldown_steps=25_000,
        evolution_stop_reward=1000000,
        sigma_conservative=0.002,
        sigma_explorer=0.01,
        sigma_destroyer=0.03,
        max_inject_per_evo=150,
        evolution_horizon=1000,
        trajectory_keep_quantile=0.5,
        mutation_parent="ema",
        noise_initial=0.124,
        noise_final=0.124,
        final_eval_episodes=50,
        noise_decay_steps=1,
        penalty_initial=1.0,
        penalty_final=0.05,
        penalty_decay_steps=500_000,
        fn_check_stagnation=lambda rm, om: rm <= om + 1.0,
        fn_qualify_conservative=lambda r, q, mr, bt, term: r >= mr,
        fn_qualify_explorer=lambda r, q, mr, bt, term: r > mr * 0.8 and q > bt * 0.5,
        fn_qualify_destroyer=lambda r, q, mr, bt, term: r > mr * 1.5 and not term,
    ),
}

@dataclass
class EnvironmentConfig:
    env_name: str
    values: Dict[str, Any]

    @classmethod
    def create(cls, env_name: str, overrides: Optional[Dict[str, Any]] = None) -> "EnvironmentConfig":
        if env_name not in ENV_PARAMS:
            choices = ", ".join(ENV_PARAMS.keys())
            raise ValueError(f"Unknown environment: {env_name}. Available environments: {choices}")
        values = copy.deepcopy(ENV_PARAMS[env_name])
        requested = dict(overrides or {})
        if requested.get("td3_baseline") and requested.get("vanilla_td3"):
            raise ValueError("--td3_baseline and --vanilla_td3 are mutually exclusive")
        if requested.get("td3_baseline"):
            values.update(copy.deepcopy(TD3_BASELINE_OVERRIDES))
        elif requested.get("vanilla_td3"):

            values.update(copy.deepcopy(VANILLA_TD3_OVERRIDES))
        if requested:
            for key, value in requested.items():
                if value is not None:
                    if key not in values:
                        raise KeyError(f"Parameter {key} is not defined in ENV_PARAMS[{env_name}]")
                    values[key] = value
        config = cls(env_name=env_name, values=values)
        config.validate()
        return config

    def __getattr__(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def validate(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        if not 0.0 <= self.ema_beta < 1.0:
            raise ValueError("ema_beta must be in [0, 1)")
        if self.batch_size <= 0 or self.replay_size < self.batch_size:
            raise ValueError("replay_size must be at least batch_size")
        actor_lr_final = self.values.get("actor_lr_final")
        if self.actor_lr <= 0.0 or (actor_lr_final is not None and actor_lr_final <= 0.0):
            raise ValueError("Actor learning rates must be positive")
        if self.actor_lr_decay_start_step < 0:
            raise ValueError("actor_lr_decay_start_step must be non-negative")
        if actor_lr_final is not None and self.actor_lr_decay_steps <= 0:
            raise ValueError("actor_lr_decay_steps must be positive when actor_lr_final is enabled")
        if self.start_steps < 0 or self.update_after < 0:
            raise ValueError("start_steps and update_after must be non-negative")
        if self.policy_freq <= 0 or self.updates_per_step <= 0:
            raise ValueError("Update frequencies must be positive integers")
        if self.evolution_horizon <= 0 or self.max_inject_per_evo < 0:
            raise ValueError("evolution_horizon must be positive and max_inject_per_evo non-negative")
        if not 0.0 <= self.trajectory_keep_quantile < 1.0:
            raise ValueError("trajectory_keep_quantile must be in [0, 1)")
        for role in ("conservative", "explorer", "destroyer"):
            role_quantile = self.values[f"trajectory_keep_quantile_{role}"]
            if role_quantile is not None and not 0.0 <= role_quantile < 1.0:
                raise ValueError(f"trajectory_keep_quantile_{role} must be in [0, 1) or None")
            if self.values[f"parent_return_tolerance_{role}"] < 0.0:
                raise ValueError(f"parent_return_tolerance_{role} must be non-negative")
        if self.stagnation_detector not in {"legacy", "normalized_windows"}:
            raise ValueError("stagnation_detector must be legacy or normalized_windows")
        if self.stagnation_progress_threshold < 0.0:
            raise ValueError("stagnation_progress_threshold must be non-negative")
        if self.stagnation_variation_threshold < 0.0:
            raise ValueError("stagnation_variation_threshold must be non-negative")
        if self.transition_value_mode not in {"current_q", "one_step_target"}:
            raise ValueError("transition_value_mode must be current_q or one_step_target")
        if self.mutation_parent not in {"actor", "ema"}:
            raise ValueError("mutation_parent must be either actor or ema")
        if self.eval_actor not in {"actor", "ema"}:
            raise ValueError("eval_actor must be either actor or ema")
        if self.vanilla_td3:
            if self.td3_baseline:
                raise ValueError("td3_baseline and vanilla_td3 are mutually exclusive")
            if self.use_layernorm or self.use_ema_actor:
                raise ValueError("vanilla_td3 requires use_layernorm=False and use_ema_actor=False")
            if self.eval_actor != "actor":
                raise ValueError("vanilla_td3 must evaluate the current actor")
            if self.target_q_min is not None or self.target_q_max is not None:
                raise ValueError("vanilla_td3 must not clamp the target Q")
            if self.td_error_clip is not None:
                raise ValueError("vanilla_td3 must not clip TD errors")
            if self.critic_grad_clip is not None or self.actor_grad_clip is not None:
                raise ValueError("vanilla_td3 must not clip gradients")
            if self.critic_loss_mode != "mse":
                raise ValueError("vanilla_td3 must use the plain MSE critic loss")
        if self.td3_baseline:
            if self.use_ema_actor:
                raise ValueError("td3_baseline must disable the EMA actor")
            if self.eval_actor != "actor":
                raise ValueError("td3_baseline must evaluate the current actor")
            if not self.use_layernorm:
                raise ValueError("td3_baseline keeps the LayerNorm backbone")
        if self.critic_loss_mode not in {"clipped_mse", "mse", "huber"}:
            raise ValueError("critic_loss_mode must be clipped_mse, mse, or huber")
        if self.critic_huber_delta <= 0.0:
            raise ValueError("critic_huber_delta must be positive")
        if self.mutation_style not in {"legacy", "output_masked"}:
            raise ValueError("mutation_style must be legacy or output_masked")
        if not 0.0 < self.mutation_fraction <= 1.0:
            raise ValueError("mutation_fraction must be in (0, 1]")
        if not isinstance(self.force_nonzero_mutation, bool):
            raise ValueError("force_nonzero_mutation must be a bool")
        if self.eval_every_steps is not None and self.eval_every_steps <= 0:
            raise ValueError("eval_every_steps must be positive or None")
        if self.eval_every_episodes <= 0 or self.eval_episodes <= 0 or self.final_eval_episodes <= 0:
            raise ValueError("Evaluation frequencies and episode counts must be positive")
        if self.eval_seed_mode not in {"fixed", "rotating"}:
            raise ValueError("eval_seed_mode must be fixed or rotating")
        if self.eval_seed_stride <= 0:
            raise ValueError("eval_seed_stride must be positive")
        if self.eval_seed_mode == "rotating" and self.eval_seed_stride <= self.eval_episodes:
            raise ValueError("eval_seed_stride must exceed eval_episodes in rotating mode")
        if self.recent_window <= 0 or self.checkpoint_window <= 0:
            raise ValueError("recent_window and checkpoint_window must be positive")
        if self.evolution_stop_survival_rate is not None:
            if not 0.0 <= self.evolution_stop_survival_rate <= 1.0:
                raise ValueError("evolution_stop_survival_rate must be in [0, 1]")
            if self.evolution_stop_survival_min_reward is None:
                raise ValueError("evolution_stop_survival_min_reward is required with a survival gate")

    def noise_std(self, step: int) -> float:
        return linear_value(self.noise_initial, self.noise_final, step, self.noise_decay_steps)

    def actor_learning_rate(self, step: int) -> float:
        if self.actor_lr_final is None or step < self.actor_lr_decay_start_step:
            return float(self.actor_lr)
        return linear_value(
            self.actor_lr,
            self.actor_lr_final,
            step - self.actor_lr_decay_start_step,
            self.actor_lr_decay_steps,
        )

    def penalty_weight(self, step: int) -> float:
        return linear_value(self.penalty_initial, self.penalty_final, step, self.penalty_decay_steps)

    def check_stagnation(self, recent_mean: float, older_mean: float) -> bool:
        return bool(self.fn_check_stagnation(recent_mean, older_mean))

    def trajectory_quantile(self, role: str) -> float:
        value = self.values[f"trajectory_keep_quantile_{role}"]
        return float(self.trajectory_keep_quantile if value is None else value)

    def parent_return_tolerance(self, role: str) -> float:
        return float(self.values[f"parent_return_tolerance_{role}"])

    def is_qualified(
        self,
        role: str,
        reward: float,
        q_value: float,
        mean_reward: float,
        base_threshold: float,
        terminated: bool,
    ) -> bool:
        fn = self.values[f"fn_qualify_{role}"]
        return bool(fn(reward, q_value, mean_reward, base_threshold, terminated))

    def serializable(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"env_name": self.env_name}
        for key, value in self.values.items():
            if callable(value):
                result[key] = getattr(value, "__name__", "callable")
            else:
                result[key] = value
        return result

def linear_value(initial: float, final: float, step: int, decay_steps: int) -> float:
    if decay_steps <= 0:
        return float(final)
    ratio = min(max(step / float(decay_steps), 0.0), 1.0)
    return float(initial + ratio * (final - initial))
