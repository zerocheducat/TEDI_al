# -*- coding: utf-8 -*-


from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import EnvironmentConfig
from .environments import ShadowEnvPool
from .evolution import TEDIEvolutionMixin
from .replay import ReplayBuffer
from .td3 import TD3
from .utils import deterministic_stream_seed, make_env, write_json


class TEDIFramework(TEDIEvolutionMixin):
    def __init__(
        self,
        config: EnvironmentConfig,
        total_steps: int,
        pop_size: int,
        seed: int,
        run_dir: Path,
        device: torch.device,
        replay_on_cpu: bool,
        disable_evolution: bool,
        disable_trigger: bool = False,
        disable_filtering: bool = False,
        trigger_interval: int = 50000,
        trigger_mode: str = "stagnation",
        trigger_jitter: float = 0.5,
        unlimited_injection: bool = False,
        log_candidates: bool = False,
        matched_budget=None,
    ) -> None:
        self.config = config
        self.matched_budget = matched_budget
        self.evaluation_main_steps = []
        self.total_injected_transitions = 0
        self.total_steps = int(total_steps)
        self.pop_size = int(pop_size)
        self.seed = int(seed)
        self.run_dir = run_dir
        self.device = device
        self.disable_evolution = disable_evolution
        self.disable_trigger = bool(disable_trigger)
        self.disable_filtering = bool(disable_filtering)
        self.trigger_interval = int(trigger_interval)
        if trigger_mode not in {"stagnation", "periodic", "random"}:
            raise ValueError("trigger_mode must be stagnation, periodic, or random")
        self.trigger_mode = str(trigger_mode)
        if not 0.0 <= trigger_jitter < 1.0:
            raise ValueError("trigger_jitter must be in [0, 1)")
        self.trigger_jitter = float(trigger_jitter)
        self.trigger_rng = np.random.default_rng(seed + 77)
        if self.trigger_mode == "periodic":
            self._trigger_next = float(config.min_steps_before_evolution) + float(self.trigger_interval)
        elif self.trigger_mode == "random":
            self._trigger_next = float(config.min_steps_before_evolution) + self._draw_trigger_gap()
        else:
            self._trigger_next = None
        self.unlimited_injection = bool(unlimited_injection)
        self.log_candidates = bool(log_candidates)
        if self.log_candidates:
            (self.run_dir / "candidates").mkdir(parents=True, exist_ok=True)

        self.train_env = make_env(config.env_name, seed)
        self.eval_env = make_env(config.env_name, seed + 100_000)
        self.shadow_pool = ShadowEnvPool(config.env_name, pop_size, seed + 200_000)
        self.parent_shadow_env = (
            make_env(config.env_name, seed + 250_000)
            if config.use_parent_return_gate
            else None
        )

        observation_space = self.train_env.observation_space
        action_space = self.train_env.action_space
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError("TEDI supports only continuous Box action spaces")
        self.state_dim = int(np.prod(observation_space.shape))
        self.action_dim = int(np.prod(action_space.shape))
        high = np.asarray(action_space.high, dtype=np.float32)
        low = np.asarray(action_space.low, dtype=np.float32)
        if not np.allclose(high, high[0]) or not np.allclose(low, -high):
            raise ValueError("The current network requires identical symmetric bounds for all action dimensions")
        self.max_action = float(high[0])

        self.td3 = TD3(self.state_dim, self.action_dim, self.max_action, config, device)
        storage_device = torch.device("cpu") if replay_on_cpu else device
        self.buffer = ReplayBuffer(
            self.state_dim,
            self.action_dim,
            config.replay_size,
            storage_device,
            device,
        )

        self.action_rng = np.random.default_rng(seed + 11)
        self.inject_rng = np.random.default_rng(seed + 22)
        self.eval_history: List[float] = []
        self.eval_length_history: List[float] = []
        self.eval_survival_history: List[float] = []
        self.loss_history: List[float] = []
        self.actor_loss_history: List[float] = []
        self.best_single_reward = -np.inf
        self.best_single_step = -1
        self.best_checkpoint_score = -np.inf
        self.best_checkpoint_step = -1
        self.best_actor_state: Optional[Dict[str, torch.Tensor]] = None
        self.stagnation_patience = 0
        self.last_evolution_step = -10**18
        self.evolution_count = 0
        self.shadow_env_steps = 0
        self.parent_shadow_steps = 0
        self.evaluation_env_steps = 0
        self.evaluation_count = 0
        self.training_episodes = 0

        self.text_log_path = run_dir / "training_log.txt"
        self.event_log_path = run_dir / "events.jsonl"
        self.best_model_path = run_dir / "best_stable_model.pth"
        self.best_single_model_path = run_dir / "best_single_model.pth"
        self.latest_checkpoint_path = run_dir / "latest_checkpoint.pth"
        self._write_text("========== TEDI run initialized ==========")

    def _draw_trigger_gap(self) -> float:
        """Sample a bounded uniform gap with the configured mean interval."""


        if self.trigger_jitter <= 0.0:
            return float(self.trigger_interval)
        u = float(self.trigger_rng.random())
        return float(self.trigger_interval) * (1.0 + self.trigger_jitter * (2.0 * u - 1.0))

    def _write_text(self, message: str) -> None:
        with self.text_log_path.open("a", encoding="utf-8") as file:
            file.write(message + "\n")

    def _write_event(self, event: Dict[str, Any]) -> None:
        with self.event_log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _evaluation_seeds(self, episodes: int, stream_id: int, group_index: int = 0) -> List[int]:
        if stream_id == 1 and self.config.eval_seed_mode == "fixed":
            return [self.seed + 100_000 + episode for episode in range(episodes)]
        if stream_id == 2:
            return [self.seed + 500_000 + episode for episode in range(episodes)]
        offset = group_index * self.config.eval_seed_stride
        return [
            deterministic_stream_seed(self.seed, stream_id, offset + episode)
            for episode in range(episodes)
        ]

    @torch.inference_mode()
    def evaluate(
        self,
        actor: nn.Module,
        evaluation_seeds: Sequence[int],
    ) -> Tuple[float, List[float], List[int], float]:

        returns: List[float] = []
        lengths: List[int] = []
        survived = 0
        actor.eval()
        for evaluation_seed in evaluation_seeds:
            state, _ = self.eval_env.reset(seed=int(evaluation_seed))
            total_reward = 0.0
            episode_steps = 0
            while True:
                action = self.td3.select_action(state, actor=actor)
                state, reward, terminated, truncated, _ = self.eval_env.step(action)
                total_reward += float(reward)
                episode_steps += 1
                self.evaluation_env_steps += 1
                if terminated or truncated:
                    survived += int(bool(truncated and not terminated))
                    break
            returns.append(total_reward)
            lengths.append(episode_steps)
        actor.train()
        return float(np.mean(returns)), returns, lengths, float(survived / len(evaluation_seeds))

    def _cpu_actor_state(self, actor: nn.Module) -> Dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}

    def _save_progress_arrays(self) -> None:
        np.save(self.run_dir / "eval_returns.npy", np.asarray(self.eval_history, dtype=np.float64))
        np.save(self.run_dir / "eval_lengths.npy", np.asarray(self.eval_length_history, dtype=np.float64))
        np.save(self.run_dir / "eval_survival_rates.npy", np.asarray(self.eval_survival_history, dtype=np.float64))
        np.save(self.run_dir / "critic_losses.npy", np.asarray(self.loss_history, dtype=np.float64))
        np.save(self.run_dir / "actor_losses.npy", np.asarray(self.actor_loss_history, dtype=np.float64))

    def _save_latest_checkpoint(self, step: int) -> None:
        payload = {
            "step": step,
            "seed": self.seed,
            "env_name": self.config.env_name,
            "config": self.config.serializable(),
            "td3": self.td3.checkpoint(),
            "eval_history": self.eval_history,
            "eval_length_history": self.eval_length_history,
            "eval_survival_history": self.eval_survival_history,
            "best_single_reward": self.best_single_reward,
            "best_single_step": self.best_single_step,
            "best_checkpoint_score": self.best_checkpoint_score,
            "best_checkpoint_step": self.best_checkpoint_step,
            "evolution_count": self.evolution_count,
            "evaluation_count": self.evaluation_count,
            "shadow_env_steps": self.shadow_env_steps,
            "parent_shadow_steps": self.parent_shadow_steps,
            "evaluation_env_steps": self.evaluation_env_steps,
        }
        torch.save(payload, self.latest_checkpoint_path)

    def _eval_model(self):
        """Actor used for every periodic/final evaluation: EMA or current weights."""
        return self.td3.ema_actor if self.config.eval_actor == "ema" else self.td3.actor

    def _handle_evaluation(self, step: int, progress: tqdm) -> None:
        evaluation_index = self.evaluation_count
        seed_group = evaluation_index if self.config.eval_seed_mode == "rotating" else 0
        evaluation_seeds = self._evaluation_seeds(self.config.eval_episodes, stream_id=1, group_index=seed_group)
        eval_mean, eval_returns, eval_lengths, survival_rate = self.evaluate(self._eval_model(), evaluation_seeds)
        self.evaluation_count += 1
        eval_std = float(np.std(eval_returns))
        mean_episode_length = float(np.mean(eval_lengths))
        reward_per_step = float(np.sum(eval_returns) / max(1, np.sum(eval_lengths)))
        self.eval_history.append(eval_mean)
        self.evaluation_main_steps.append(step)
        self.eval_length_history.append(mean_episode_length)
        self.eval_survival_history.append(survival_rate)
        recent_window = self.config.recent_window
        recent_values = self.eval_history[-recent_window:]
        recent_mean = float(np.mean(recent_values))
        current_std = float(np.std(recent_values))
        normalized_variation = float(current_std / (abs(recent_mean) + 1e-8))

        if eval_mean > self.best_single_reward:
            self.best_single_reward = eval_mean
            self.best_single_step = step
            torch.save(self._cpu_actor_state(self._eval_model()), self.best_single_model_path)

        checkpoint_values = self.eval_history[-self.config.checkpoint_window:]
        checkpoint_score = float(np.mean(checkpoint_values))
        if len(checkpoint_values) >= self.config.checkpoint_window and checkpoint_score > self.best_checkpoint_score:
            self.best_checkpoint_score = checkpoint_score
            self.best_checkpoint_step = step
            self.best_actor_state = self._cpu_actor_state(self._eval_model())
            torch.save(self.best_actor_state, self.best_model_path)

        older_mean: Optional[float] = None
        normalized_progress: Optional[float] = None
        raw_stagnation = False
        if len(self.eval_history) >= recent_window * 2:
            older_values = self.eval_history[-recent_window * 2 : -recent_window]
            older_mean = float(np.mean(older_values))
            normalized_progress = float((recent_mean - older_mean) / (abs(older_mean) + 1e-8))
            if self.config.stagnation_detector == "normalized_windows":
                raw_stagnation = bool(
                    normalized_progress < self.config.stagnation_progress_threshold
                    and normalized_variation < self.config.stagnation_variation_threshold
                )
            else:
                raw_stagnation = self.config.check_stagnation(recent_mean, older_mean)
            if raw_stagnation:
                self.stagnation_patience += 1
            else:
                self.stagnation_patience = 0

        message = (
            f"[EVALUATION] Step={step:08d}, score={eval_mean:.2f}, "
            f"recent mean={recent_mean:.2f}, recent std={current_std:.2f}, "
            f"progress={normalized_progress if normalized_progress is not None else float('nan'):.5f}, "
            f"variation={normalized_variation:.5f}, "
            f"length={mean_episode_length:.1f}, survival={survival_rate:.0%}, "
            f"actor lr={self.td3.current_actor_lr:.2e}, "
            f"best score={self.best_single_reward:.2f}, stagnation={self.stagnation_patience}"
        )
        self._write_text(message)
        self._write_event(
            {
                "type": "evaluation",
                "main_step": step,
                "evaluation_index": evaluation_index,
                "evaluation_seed_mode": self.config.eval_seed_mode,
                "evaluation_seed_group": seed_group,
                "evaluation_seeds": evaluation_seeds,
                "episode_returns": eval_returns,
                "episode_lengths": eval_lengths,
                "evaluation_mean": eval_mean,
                "evaluation_std": eval_std,
                "mean_episode_length": mean_episode_length,
                "survival_rate": survival_rate,
                "reward_per_step": reward_per_step,
                "recent_mean": recent_mean,
                "older_mean": older_mean,
                "recent_std": current_std,
                "normalized_progress": normalized_progress,
                "normalized_variation": normalized_variation,
                "raw_stagnation_signal": raw_stagnation,
                "best_evaluation": self.best_single_reward,
                "best_window_mean": self.best_checkpoint_score,
                "stagnation_patience": self.stagnation_patience,
                "actor_learning_rate": self.td3.current_actor_lr,
                "cumulative_evaluation_steps": self.evaluation_env_steps,
            }
        )
        progress.set_postfix(
            {
                "recent": f"{recent_mean:.1f}",
                "best": f"{self.best_single_reward:.1f}",
                "stagnation": self.stagnation_patience,
            }
        )

        if (older_mean is not None or (self.matched_budget is not None and self.trigger_mode != 'stagnation')) and self._can_evolve(
            step,
            recent_mean,
            current_std,
            eval_mean,
            survival_rate,
            normalized_variation,
        ):
            self.run_evolution(step, recent_mean)

    def train(self) -> Dict[str, Any]:
        state, _ = self.train_env.reset(seed=self.seed)
        episode_reward = 0.0
        progress = tqdm(range(self.total_steps), desc=f"Training {self.config.env_name}", dynamic_ncols=True)
        started_at = time.time()

        for step in progress:
            main_step = step + 1
            if step < self.config.start_steps:
                action = self.train_env.action_space.sample()
            else:
                noise_std = self.config.noise_std(step) * self.max_action
                action = self.td3.select_action(state, noise_std=noise_std, rng=self.action_rng)

            next_state, reward, terminated, truncated, _ = self.train_env.step(action)
            done = bool(terminated or truncated)
            terminal_for_buffer = float(terminated if self.config.bootstrap_on_truncation else done)
            self.buffer.add(
                np.asarray(state, dtype=np.float32),
                np.asarray(action, dtype=np.float32),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                terminal_for_buffer,
            )
            episode_reward += float(reward)

            if self.buffer.size >= self.config.batch_size and step >= self.config.update_after:
                for _ in range(self.config.updates_per_step):
                    metrics = self.td3.train(self.buffer, main_step)
                    self.loss_history.append(metrics["critic_loss"])
                    if np.isfinite(metrics["actor_loss"]):
                        self.actor_loss_history.append(metrics["actor_loss"])

            if done:
                self.training_episodes += 1
                self._write_event(
                    {
                        "type": "training_episode",
                        "main_step": main_step if self.config.eval_every_steps is not None else step,
                        "episode": self.training_episodes,
                        "training_return": episode_reward,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                )
                if (
                    self.config.eval_every_steps is None
                    and self.training_episodes % self.config.eval_every_episodes == 0
                ):
                    self._handle_evaluation(step, progress)
                if self.training_episodes % self.config.save_every_episodes == 0:
                    self._save_progress_arrays()
                    checkpoint_step = main_step if self.config.eval_every_steps is not None else step
                    self._save_latest_checkpoint(checkpoint_step)
                state, _ = self.train_env.reset()
                episode_reward = 0.0
            else:
                state = next_state
            if self.config.eval_every_steps is not None and main_step % self.config.eval_every_steps == 0:
                self._handle_evaluation(main_step, progress)

        self._save_progress_arrays()
        self._save_latest_checkpoint(self.total_steps)

        # Evaluate current and stable checkpoints with the same final seed set.
        final_evaluation_seeds = self._evaluation_seeds(self.config.final_eval_episodes, stream_id=2)
        current_mean, current_returns, current_lengths, current_survival = self.evaluate(
            self._eval_model(),
            final_evaluation_seeds,
        )
        best_mean: Optional[float] = None
        best_returns: Optional[List[float]] = None
        best_lengths: Optional[List[int]] = None
        best_survival: Optional[float] = None
        if self.best_actor_state is not None:
            best_actor = copy.deepcopy(self._eval_model())
            best_actor.load_state_dict(self.best_actor_state)
            best_mean, best_returns, best_lengths, best_survival = self.evaluate(
                best_actor,
                final_evaluation_seeds,
            )

        summary: Dict[str, Any] = {
            "environment": self.config.env_name,
            "seed": self.seed,
            "main_env_steps": self.total_steps,
            "shadow_env_steps": self.shadow_env_steps,
            "parent_shadow_steps": self.parent_shadow_steps,
            "evaluation_env_steps": self.evaluation_env_steps,
            "training_episodes": self.training_episodes,
            "periodic_evaluations": self.evaluation_count,
            "evolution_events": self.evolution_count,
            "periodic_best_evaluation": self.best_single_reward,
            "periodic_best_evaluation_step": self.best_single_step,
            "periodic_best_window_mean": self.best_checkpoint_score,
            "periodic_best_window_step": self.best_checkpoint_step,
            "final_actor_learning_rate": self.td3.current_actor_lr,
            "final_evaluation_seeds": final_evaluation_seeds,
            "eval_actor": self.config.eval_actor,
            "current_final_mean": current_mean,
            "current_final_returns": current_returns,
            "current_final_lengths": current_lengths,
            "current_final_survival_rate": current_survival,
            "best_stable_final_mean": best_mean,
            "best_stable_final_returns": best_returns,
            "best_stable_final_lengths": best_lengths,
            "best_stable_final_survival_rate": best_survival,
            "wall_time_seconds": time.time() - started_at,
        }
        if self.matched_budget is not None:
            summary.update(self.matched_budget.report(self.shadow_env_steps, self.evolution_count, self.parent_shadow_steps))
            summary.update(method=self.trigger_mode, main_steps=self.total_steps,
                           admitted_transitions=self.total_injected_transitions,
                           peak=self.best_single_reward, independent_final_mean=current_mean,
                           raw_return_AUC=float(np.trapz(self.eval_history, self.evaluation_main_steps)))
        write_json(self.run_dir / "training_summary.json", summary)
        success_message = (
            f"[TRAINING COMPLETE] Current final mean={current_mean:.2f}, "
            f"best stable final mean={best_mean if best_mean is not None else float('nan'):.2f}, "
            f"output directory={self.run_dir}"
        )
        print(success_message)
        self._write_text(success_message)
        return summary

    def close(self) -> None:
        self.train_env.close()
        self.eval_env.close()
        self.shadow_pool.close()
        if self.parent_shadow_env is not None:
            self.parent_shadow_env.close()
