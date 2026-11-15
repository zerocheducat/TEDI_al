# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .replay import Transition


class TEDIEvolutionMixin:
    def _mutate_policy(self, base_actor: nn.Module, sigma: float, role: str) -> nn.Module:
        mutated_actor = copy.deepcopy(base_actor)
        if self.config.mutation_style == "output_masked":
            with torch.no_grad():
                for name, parameter in mutated_actor.named_parameters():
                    if name == "l3.weight":
                        mask = torch.zeros_like(parameter)
                        selected_per_action = max(1, int(round(parameter.shape[1] * self.config.mutation_fraction)))
                        for action_index in range(parameter.shape[0]):
                            indices = torch.randperm(parameter.shape[1], device=parameter.device)[:selected_per_action]
                            mask[action_index, indices] = 1.0
                        parameter.add_(torch.randn_like(parameter) * mask * sigma)
                    elif name == "l3.bias":
                        parameter.add_(torch.randn_like(parameter) * sigma)
            mutated_actor.eval()
            return mutated_actor

        mutation_probability = 0.3 if role == "destroyer" else 0.2
        eligible_parameters = [
            (name, parameter)
            for name, parameter in mutated_actor.named_parameters()
            if "l3" in name or role == "destroyer"
        ]
        if not eligible_parameters:
            raise RuntimeError(f"No eligible parameters found for legacy mutation role: {role}")

        with torch.no_grad():
            selected_parameters = [
                (name, parameter)
                for name, parameter in eligible_parameters
                if torch.rand((), device=parameter.device).item() < mutation_probability
            ]

            if not selected_parameters and self.config.force_nonzero_mutation:
                device = eligible_parameters[0][1].device
                forced_index = int(
                    torch.randint(
                        low=0,
                        high=len(eligible_parameters),
                        size=(1,),
                        device=device,
                    ).item()
                )
                selected_parameters = [eligible_parameters[forced_index]]
            for _, parameter in selected_parameters:
                parameter.add_(torch.randn_like(parameter) * sigma)
        mutated_actor.eval()
        return mutated_actor

    @torch.inference_mode()
    def _evaluate_parent_in_shadow(self, actor: nn.Module, common_seed: int) -> Tuple[float, int]:
        if self.parent_shadow_env is None:
            raise RuntimeError("Parent shadow environment is unavailable while the parent gate is enabled")
        was_training = actor.training
        actor.eval()
        state, _ = self.parent_shadow_env.reset(seed=common_seed)
        total_reward = 0.0
        rollout_steps = 0
        for _ in range(self.config.evolution_horizon):
            action = self.td3.select_action(state, actor=actor)
            state, reward, terminated, truncated, _ = self.parent_shadow_env.step(action)
            total_reward += float(reward)
            rollout_steps += 1
            self.shadow_env_steps += 1
            self.parent_shadow_steps += 1
            if terminated or truncated:
                break
        if was_training:
            actor.train()
        return total_reward, rollout_steps

    def _can_evolve(
            self,
            step: int,
            recent_mean: float,
            current_std: float,
            eval_mean: float,
            survival_rate: float,
            normalized_variation: float,
    ) -> bool:
        if self.disable_evolution:
            return False
        budget = getattr(self, 'matched_budget', None)
        if budget is not None and not budget.available(self.shadow_env_steps, self.evolution_count):
            return False


        # The legacy trigger ablation bypasses the remaining trigger guards.
        if self.disable_trigger:
            if step < self.trigger_interval:
                return False
            return (step - self.last_evolution_step) >= self.trigger_interval


        if step < self.config.min_steps_before_evolution:
            return False
        if step - self.last_evolution_step < self.config.evolution_cooldown_steps:
            return False
        if recent_mean <= self.config.min_mean_reward_to_evolve:
            return False
        stop_reward = self.config.evolution_stop_reward
        if stop_reward is not None and recent_mean >= stop_reward:
            self.stagnation_patience = 0
            return False
        if (
                self.config.evolution_stop_survival_rate is not None
                and eval_mean >= self.config.evolution_stop_survival_min_reward
                and survival_rate >= self.config.evolution_stop_survival_rate
        ):
            self.stagnation_patience = 0
            return False


        if budget is not None and budget.mode != 'stagnation':
            # FIFO: consume a queued request only when a call is actually made.
            return budget.requested(step)
        if self.config.stagnation_detector == "normalized_windows":
            low_variation = normalized_variation < self.config.stagnation_variation_threshold
        else:
            threshold = self.config.dynamic_threshold_ratio * abs(self.config.target_reward)
            low_variation = current_std < threshold

        mode = getattr(self, "trigger_mode", "stagnation")
        if mode == "stagnation":
            return (
                    self.stagnation_patience >= self.config.stagnation_patience_limit
                    and low_variation
            )


        # Scheduled calls still obey the common reward and stopping guards.
        if self._trigger_next is None:
            return False
        if step < self._trigger_next:
            return False


        # Start the next interval from the actual call, including any guard delay.
        gap = self._draw_trigger_gap() if mode == "random" else float(self.trigger_interval)
        self._trigger_next = float(step) + gap
        return True

    def _select_injections(
        self,
        candidates: List[List[Transition]],
        returns: np.ndarray,
        roles: Sequence[str],
        parent_return: Optional[float],
    ) -> Tuple[List[Transition], List[int], Dict[str, Any]]:
        if len(returns) == 0:
            return [], [], {
                "return_threshold": float("nan"),
                "role_return_thresholds": {},
                "parent_return_floors": [],
                "population_gate_pass": [],
                "parent_gate_pass": [],
                "injectable_policies": [],
                "injected_transitions_by_role": {},
            }

        if self.disable_filtering:
            eligible = [index for index, items in enumerate(candidates) if items]
            selected: List[Transition] = []

            for index in eligible:
                selected.extend(candidates[index])
                if (not self.unlimited_injection
                        and len(selected) >= self.config.max_inject_per_evo):
                    break

            if not self.unlimited_injection:
                selected = selected[: self.config.max_inject_per_evo]

            return selected, eligible, {
                "return_threshold": float("-inf"),
                "role_return_thresholds": {},
                "parent_return_floors": [],
                "population_gate_pass": [True] * len(returns),
                "parent_gate_pass": [True] * len(returns),
                "injectable_policies": eligible,
                "injected_transitions_by_role": {},
            }

        role_thresholds = {
            role: float(np.quantile(returns, self.config.trajectory_quantile(role)))
            for role in ("conservative", "explorer", "destroyer")
        }
        parent_floors: List[Optional[float]] = []
        population_gate_pass: List[bool] = []
        parent_gate_pass: List[bool] = []
        eligible: List[int] = []
        for index, value in enumerate(returns):
            role = roles[index]
            passes_population = bool(value >= role_thresholds[role])
            if self.config.use_parent_return_gate:
                if parent_return is None:
                    raise RuntimeError("Parent return is required while the parent gate is enabled")
                parent_floor = float(parent_return - self.config.parent_return_tolerance(role))
                passes_parent = bool(value >= parent_floor)
            else:
                parent_floor = None
                passes_parent = True
            parent_floors.append(parent_floor)
            population_gate_pass.append(passes_population)
            parent_gate_pass.append(passes_parent)
            if passes_population and passes_parent:
                eligible.append(index)

        injectable = [index for index in eligible if candidates[index]]
        shuffled: Dict[int, List[Transition]] = {}
        for index in injectable:
            items = list(candidates[index])
            self.inject_rng.shuffle(items)
            shuffled[index] = items

        role_order = ("conservative", "explorer", "destroyer")
        members_by_role = {
            role: [index for index in injectable if roles[index] == role]
            for role in role_order
        }
        selected: List[Transition] = []
        cursor = {index: 0 for index in injectable}
        member_cursor = {role: 0 for role in role_order}
        injected_by_role = {role: 0 for role in role_order}
        budget = None if self.unlimited_injection else self.config.max_inject_per_evo
        while budget is None or len(selected) < budget:
            added = False
            for role in role_order:
                members = members_by_role[role]
                for _ in range(len(members)):
                    member_position = member_cursor[role] % len(members)
                    index = members[member_position]
                    member_cursor[role] += 1
                    position = cursor[index]
                    if position >= len(shuffled[index]):
                        continue
                    selected.append(shuffled[index][position])
                    cursor[index] += 1
                    injected_by_role[role] += 1
                    added = True
                    break
                if budget is not None and len(selected) >= budget:
                    break
            if not added:
                break

        metadata: Dict[str, Any] = {
            "return_threshold": float(np.quantile(returns, self.config.trajectory_keep_quantile)),
            "role_return_thresholds": role_thresholds,
            "parent_return_floors": parent_floors,
            "population_gate_pass": population_gate_pass,
            "parent_gate_pass": parent_gate_pass,
            "injectable_policies": injectable,
            "injected_transitions_by_role": injected_by_role,
        }
        return selected, eligible, metadata

    def _dump_event_candidates(self, step, log_rec, id_to_rec, selected, roles, eligible,
                               selection_metadata, parent_return, episodic_returns,
                               median_q_by_step, candidates):
        """Save candidate transitions and stage masks for offline diagnostics."""
        import json

        root = self.run_dir / "candidates"
        root.mkdir(parents=True, exist_ok=True)

        n = len(log_rec)
        if n == 0:
            return
        state_dim = int(log_rec[0][2].shape[0])
        action_dim = int(log_rec[0][3].shape[0])

        state = np.zeros((n, state_dim), dtype=np.float32)
        action = np.zeros((n, action_dim), dtype=np.float32)
        next_state = np.zeros((n, state_dim), dtype=np.float32)
        reward = np.zeros(n, dtype=np.float32)
        terminal = np.zeros(n, dtype=np.float32)
        truncated = np.zeros(n, dtype=np.float32)
        q_value = np.zeros(n, dtype=np.float32)
        q1_arr = np.full(n, np.nan, dtype=np.float32)
        q2_arr = np.full(n, np.nan, dtype=np.float32)
        penalty_arr = np.zeros(n, dtype=np.float32)
        qualified = np.zeros(n, dtype=bool)
        gated = np.zeros(n, dtype=bool)
        pop_gate = np.zeros(n, dtype=bool)
        parent_gate = np.zeros(n, dtype=bool)
        injectable = np.zeros(n, dtype=bool)
        selected_mask = np.zeros(n, dtype=bool)
        policy_idx = np.zeros(n, dtype=np.int16)
        role_id = np.zeros(n, dtype=np.int8)

        role_names = ("conservative", "explorer", "destroyer")
        role_map = {r: i for i, r in enumerate(role_names)}
        injectable_policies = list(selection_metadata.get("injectable_policies") or [])
        pop_pass = list(selection_metadata.get("population_gate_pass") or [])
        par_pass = list(selection_metadata.get("parent_gate_pass") or [])
        sel_ids = {id(t) for t in selected}

        step_index = np.zeros(n, dtype=np.int32)
        for i, rec in enumerate(log_rec):
            (st_idx, env_index, s, a, r, ns, term, q, qual,
             q1v, q2v, penv, truncv) = rec
            step_index[i] = st_idx
            state[i] = s
            action[i] = a
            next_state[i] = ns
            reward[i] = r
            terminal[i] = term
            truncated[i] = truncv
            q_value[i] = q
            q1_arr[i] = q1v
            q2_arr[i] = q2v
            penalty_arr[i] = penv
            qualified[i] = qual
            policy_idx[i] = env_index
            role_id[i] = role_map.get(roles[env_index], 0)
            pop_gate[i] = bool(pop_pass[env_index]) if env_index < len(pop_pass) else bool(env_index in eligible)
            parent_gate[i] = bool(par_pass[env_index]) if env_index < len(par_pass) else bool(env_index in eligible)
            gated[i] = pop_gate[i] and parent_gate[i]

            injectable[i] = bool(qual and env_index in injectable_policies)

        for tid, rec_index in id_to_rec.items():
            if tid in sel_ids and 0 <= rec_index < n:
                selected_mask[rec_index] = True


        min_q_next = np.full(n, np.nan, dtype=np.float32)
        y_pre_clean = np.full(n, np.nan, dtype=np.float32)
        y_post_clean = np.full(n, np.nan, dtype=np.float32)
        hit_lo = np.zeros(n, dtype=bool)
        hit_hi = np.zeros(n, dtype=bool)
        try:
            target_noise_rng = np.random.default_rng(int(self.seed) * 7919 + int(self.evolution_count) * 104729)
            chunk = 512
            for start in range(0, n, chunk):
                end = min(n, start + chunk)
                ns_t = torch.as_tensor(next_state[start:end], dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    noise = torch.as_tensor(
                        target_noise_rng.normal(0.0, float(self.config.policy_noise),
                                                size=(end - start, action_dim)).astype(np.float32),
                        device=self.device)
                    noise = noise.clamp(-float(self.config.noise_clip), float(self.config.noise_clip))
                    a_next = (self.td3.actor_target(ns_t) + noise).clamp(-float(self.max_action),
                                                                         float(self.max_action))
                    tq1, tq2 = self.td3.critic_target(ns_t, a_next)
                    mq = torch.minimum(tq1, tq2).squeeze(-1)
                    r_t = torch.as_tensor(reward[start:end], dtype=torch.float32, device=self.device)
                    d_t = torch.as_tensor(terminal[start:end], dtype=torch.float32, device=self.device)
                    y = r_t + float(self.config.gamma) * (1.0 - d_t) * mq
                    y_clip = y.clone()
                    if self.config.target_q_min is not None:
                        y_clip = torch.maximum(y_clip, torch.tensor(float(self.config.target_q_min),
                                                                    device=self.device))
                    if self.config.target_q_max is not None:
                        y_clip = torch.minimum(y_clip, torch.tensor(float(self.config.target_q_max),
                                                                    device=self.device))
                min_q_next[start:end] = mq.cpu().numpy()
                y_pre_clean[start:end] = y.cpu().numpy()
                y_post_clean[start:end] = y_clip.cpu().numpy()
                hit_lo[start:end] = (y.cpu().numpy() < float(self.config.target_q_min)) if \
                    self.config.target_q_min is not None else False
                hit_hi[start:end] = (y.cpu().numpy() > float(self.config.target_q_max)) if \
                    self.config.target_q_max is not None else False
        except Exception as exc:
            print(f"   [警告] Bellman target 配对计算失败: {type(exc).__name__}: {exc}")

        tag = f"event_{self.evolution_count:04d}_step{step}"
        np.savez_compressed(
            root / (tag + ".npz"),
            state=state, action=action, reward=reward, next_state=next_state,
            terminal=terminal, truncated=truncated,
            q_value=q_value, q1=q1_arr, q2=q2_arr, penalty=penalty_arr,
            qualified=qualified, gated=gated, population_gate_pass=pop_gate,
            parent_gate_pass=parent_gate,
            injectable=injectable, selected=selected_mask,
            policy_idx=policy_idx, role_id=role_id, step_idx=step_index,
            min_q_next=min_q_next, y_pre_clean=y_pre_clean, y_post_clean=y_post_clean,
            hit_lower=hit_lo, hit_upper=hit_hi,
        )
        meta = dict(
            main_step=int(step),
            event_index=int(self.evolution_count),
            common_seed=int(self.seed) + 300_000 + int(self.evolution_count) * 10_000,
            common_seed_formula="seed + 300000 + event_index*10000",
            transition_value_mode=self.config.transition_value_mode,
            target_q_min=self.config.target_q_min,
            target_q_max=self.config.target_q_max,
            policy_noise=float(self.config.policy_noise),
            noise_clip=float(self.config.noise_clip),
            gamma=float(self.config.gamma),
            parent_return=None if parent_return is None else float(parent_return),
            policy_returns=[float(x) for x in episodic_returns],
            policy_roles=list(roles),
            retained_policies=list(eligible),
            injectable_policies=injectable_policies,
            return_threshold=selection_metadata.get("return_threshold"),
            role_return_thresholds=selection_metadata.get("role_return_thresholds"),
            fixed_injection_budget=int(self.config.max_inject_per_evo),
            injected=int(selected_mask.sum()),
            n_candidates=n,
            n_qualified=int(qualified.sum()),
            median_q_per_step=[float(x) for x in median_q_by_step],
            state_dim=state_dim,
            action_dim=action_dim,
        )
        (root / (tag + ".json")).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def run_evolution(self, step: int, recent_mean: float) -> None:
        budget = getattr(self, 'matched_budget', None)
        if budget is not None:
            if not budget.available(self.shadow_env_steps, self.evolution_count):
                return
            budget.called(step)
        aux_before = self.shadow_env_steps
        self.evolution_count += 1
        self.last_evolution_step = step
        common_seed = self.seed + 300_000 + self.evolution_count * 10_000
        states = self.shadow_pool.reset(common_seed)
        done = np.zeros(self.pop_size, dtype=bool)
        probe_lengths = np.zeros(self.pop_size, dtype=int)
        budget_stopped = False
        episodic_returns = np.zeros(self.pop_size, dtype=np.float64)
        candidates: List[List[Transition]] = [[] for _ in range(self.pop_size)]

        parent = self.td3.ema_actor if self.config.mutation_parent == "ema" else self.td3.actor
        parent_return: Optional[float] = None
        parent_rollout_steps = 0
        if self.config.use_parent_return_gate and not self.disable_filtering:
            parent_return, parent_rollout_steps = self._evaluate_parent_in_shadow(parent, common_seed)
        else:
            parent_return = None
            parent_rollout_steps = 0
        roles: List[str] = []
        mutants: List[nn.Module] = []
        for index in range(self.pop_size):
            if index < max(1, self.pop_size // 3):
                role, sigma = "conservative", self.config.sigma_conservative
            elif index < max(2, 2 * self.pop_size // 3):
                role, sigma = "explorer", self.config.sigma_explorer
            else:
                role, sigma = "destroyer", self.config.sigma_destroyer
            roles.append(role)
            mutants.append(self._mutate_policy(parent, sigma * self.max_action, role))

        start_message = (
            f"[EVOLUTION] Main step {step}: mode={self.trigger_mode}; "
            f"starting shadow evolution event {self.evolution_count}."
        )
        print(start_message)
        self._write_text(start_message)


        log_rec = [] if self.log_candidates else None
        id_to_rec = {} if self.log_candidates else None
        median_q_by_step = [] if self.log_candidates else None

        for step_idx in range(self.config.evolution_horizon):
            if budget is not None and budget.env == 'Ant-v2' and self.shadow_env_steps >= budget.target:
                budget_stopped = True
                break
            active_indices = np.flatnonzero(~done)
            if len(active_indices) == 0:
                break
            active_states = states[active_indices]
            actions: List[np.ndarray] = []
            for index in active_indices:
                actions.append(self.td3.select_action(states[index], actor=mutants[index]))

            state_tensor = torch.as_tensor(active_states, dtype=torch.float32, device=self.device)
            action_tensor = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=self.device)
            q_values_tensor: Optional[torch.Tensor] = None
            q1_np = None
            q2_np = None
            pen_np = None
            with torch.no_grad():
                if self.config.transition_value_mode == "current_q":
                    q1, q2 = self.td3.critic(state_tensor, action_tensor)
                    q_values_tensor = torch.minimum(q1, q2).squeeze(-1)
                    if log_rec is not None:
                        q1_np = q1.squeeze(-1).cpu().numpy()
                        q2_np = q2.squeeze(-1).cpu().numpy()

            step_rewards: List[float] = []
            step_results: List[Tuple[int, np.ndarray, float, bool, bool]] = []
            for local_index, env_index in enumerate(active_indices):
                next_state, reward, terminated, truncated, _ = self.shadow_pool.envs[env_index].step(actions[local_index])
                step_rewards.append(float(reward))
                step_results.append((env_index, next_state, float(reward), bool(terminated), bool(truncated)))
                self.shadow_env_steps += 1
                probe_lengths[env_index] += 1

            if self.config.transition_value_mode == "one_step_target":
                next_state_tensor = torch.as_tensor(
                    np.asarray([result[1] for result in step_results]),
                    dtype=torch.float32,
                    device=self.device,
                )
                rewards_tensor = torch.as_tensor(step_rewards, dtype=torch.float32, device=self.device)
                terminal_tensor = torch.as_tensor(
                    [
                        float(result[3] if self.config.bootstrap_on_truncation else result[3] or result[4])
                        for result in step_results
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )
                with torch.no_grad():
                    reference_actions = parent(next_state_tensor)
                    target_q1, target_q2 = self.td3.critic_target(next_state_tensor, reference_actions)
                    q_values_tensor = rewards_tensor + self.config.gamma * (1.0 - terminal_tensor) * torch.minimum(
                        target_q1,
                        target_q2,
                    ).squeeze(-1)

            if q_values_tensor is None:
                raise RuntimeError("Transition quality values were not computed")
            if self.buffer.size >= self.config.batch_size and self.config.penalty_initial > 0.0:
                with torch.no_grad():
                    reference_states, _, _, _, _ = self.buffer.sample(self.config.batch_size)
                    distances = torch.cdist(state_tensor, reference_states)
                    min_distances = distances.min(dim=1).values
                    penalty = torch.clamp(1.0 - min_distances, min=0.0)
                    penalty_effect = penalty * self.config.penalty_weight(step)
                    q_values_tensor = q_values_tensor - penalty_effect
                    if log_rec is not None:
                        pen_np = penalty_effect.cpu().numpy()
            q_values = q_values_tensor.cpu().numpy()

            mean_reward = float(np.mean(step_rewards))
            # The shared reference is the median score of active probes at this step.
            base_threshold = float(np.median(q_values))
            if median_q_by_step is not None:
                median_q_by_step.append(base_threshold)
            for local_index, result in enumerate(step_results):
                env_index, next_state, reward, terminated, truncated = result
                is_done = terminated or truncated
                episodic_returns[env_index] += reward

                if self.disable_filtering:
                    qualified = True
                else:
                    qualified = self.config.is_qualified(
                        roles[env_index],
                        reward,
                        float(q_values[local_index]),
                        mean_reward,
                        base_threshold,
                        terminated,
                    )

                if log_rec is not None:
                    rec_index = len(log_rec)
                    log_rec.append((
                        step_idx,
                        env_index,
                        states[env_index].copy(),
                        np.asarray(actions[local_index], dtype=np.float32).copy(),
                        float(reward),
                        np.asarray(next_state, dtype=np.float32).copy(),
                        float(terminated),
                        float(q_values[local_index]),
                        bool(qualified),
                        float(q1_np[local_index]) if q1_np is not None else float("nan"),
                        float(q2_np[local_index]) if q2_np is not None else float("nan"),
                        float(pen_np[local_index]) if pen_np is not None else 0.0,
                        float(truncated),
                    ))

                if qualified:
                    terminal_for_buffer = float(terminated if self.config.bootstrap_on_truncation else is_done)
                    transition = (
                        states[env_index].copy(),
                        actions[local_index].copy(),
                        reward,
                        np.asarray(next_state, dtype=np.float32).copy(),
                        terminal_for_buffer,
                    )
                    candidates[env_index].append(transition)
                    if id_to_rec is not None:
                        id_to_rec[id(transition)] = rec_index
                states[env_index] = next_state
                done[env_index] = is_done

            selected, eligible, selection_metadata = self._select_injections(
                candidates,
                episodic_returns,
                roles,
                parent_return,
            )
            if budget is not None and budget.env == 'Ant-v2' and self.shadow_env_steps >= budget.target:
                budget_stopped = True
                break
        if budget is not None:
            if budget_stopped:
                budget.boundary_lengths = probe_lengths.tolist()
            if budget.env == 'HalfCheetah-v2':
                budget.hc_costs.append(self.shadow_env_steps - aux_before)
        self.total_injected_transitions = getattr(self, 'total_injected_transitions', 0) + len(selected)
        # Only final admissions enter the learner replay buffer.
        for transition in selected:
            self.buffer.add(*transition)

        if log_rec is not None:
            self._dump_event_candidates(
                step, log_rec, id_to_rec, selected, roles, eligible,
                selection_metadata, parent_return, episodic_returns,
                median_q_by_step, candidates,
            )

        finish_message = (
            f"[EVOLUTION COMPLETE] Parent return="
            f"{parent_return if parent_return is not None else float('nan'):.2f}, "
            f"policy returns={np.round(episodic_returns, 2).tolist()}, "
            f"retained policies={eligible}, fixed budget={self.config.max_inject_per_evo}, "
            f"injected transitions={len(selected)}, "
            f"cumulative shadow steps={self.shadow_env_steps}."
        )
        print(finish_message)
        self._write_text(finish_message)
        self._write_event(
            {
                "type": "evolution",
                "main_step": step,
                "event_index": self.evolution_count,
                "recent_mean": recent_mean,
                "parent_policy_return": parent_return,
                "parent_rollout_steps": parent_rollout_steps,
                "policy_roles": roles,
                "policy_returns": episodic_returns.tolist(),
                "return_threshold": selection_metadata["return_threshold"],
                "role_return_thresholds": selection_metadata["role_return_thresholds"],
                "parent_return_floors": selection_metadata["parent_return_floors"],
                "population_gate_pass": selection_metadata["population_gate_pass"],
                "parent_gate_pass": selection_metadata["parent_gate_pass"],
                "retained_policies": eligible,
                "injectable_policies": selection_metadata["injectable_policies"],
                "qualified_transitions_per_policy": [len(items) for items in candidates],
                "fixed_injection_budget": self.config.max_inject_per_evo,
                "injected_transitions_by_role": selection_metadata["injected_transitions_by_role"],
                "injected_transitions": len(selected),
                "cumulative_parent_shadow_steps": self.parent_shadow_steps,
                "cumulative_shadow_steps": self.shadow_env_steps,
                "probe_lengths": probe_lengths.tolist(),
                "budget_exhausted": budget_stopped if budget is None else self.shadow_env_steps >= budget.target,
                "collection_stop_reason": 'auxiliary_budget' if budget_stopped else 'horizon_or_environment_done',
                "event_auxiliary_steps": self.shadow_env_steps - aux_before,
            }
        )
        self.stagnation_patience = 0
