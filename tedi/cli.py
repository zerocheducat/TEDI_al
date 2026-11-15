# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from .config import ENV_PARAMS, EnvironmentConfig
from .trainer import TEDIFramework
from .utils import safe_name, set_global_seed, write_json

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TEDI training entry point")
    parser.add_argument("--env", type=str, default="HalfCheetah-v2", choices=list(ENV_PARAMS.keys()), help="Training environment")
    parser.add_argument("--steps", type=int, default=None, help="Main-environment training steps; defaults to ENV_PARAMS")
    parser.add_argument("--seed", type=int, default=10, help="Random seed")
    parser.add_argument("--pop_size", type=int, default=6, help="Number of shadow policies")
    parser.add_argument("--run_name", type=str, default=None, help="Run name")
    parser.add_argument("--output_dir", type=str, default=None, help="Output root directory")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Training device")
    parser.add_argument("--replay_on_cpu", action="store_true", help="Store the replay buffer on CPU to reduce GPU memory use")
    parser.add_argument("--disable_evolution", action="store_true", help="Disable evolution and run the matched TD3 control")
    parser.add_argument("--td3_baseline", action="store_true",
                        help="TD3 baseline (B0): our TD3 with the EMA actor removed and evaluation on the "
                             "current actor. LayerNorm and all other backbone settings are kept. "
                             "Use together with --disable_evolution.")
    parser.add_argument("--vanilla_td3", action="store_true",
                        help="Run the vanilla TD3 baseline (B0): no LayerNorm, no EMA evaluation, no target-Q / "
                             "TD-error / gradient clipping, plain MSE loss, official TD3 hyperparameters. "
                             "Use together with --disable_evolution.")
    parser.add_argument("--disable_trigger", action="store_true", help="Replace stagnation trigger with periodic evolution events")
    parser.add_argument("--disable_filtering", action="store_true", help="Disable hierarchical admission filtering")
    parser.add_argument("--trigger_interval", type=int, default=50000, help="Periodic/random evolution interval for trigger ablation")
    parser.add_argument("--trigger_mode", choices=["stagnation", "periodic", "random"], default="stagnation",
                        help="When to run evolution: stagnation (default, stagnation-triggered), "
                             "periodic (fixed interval, all other guards kept), "
                             "random (jittered interval with the same mean, all other guards kept)")
    parser.add_argument("--trigger_jitter", type=float, default=0.5,
                        help="Relative jitter for --trigger_mode random: gaps are uniform on "
                             "interval*(1-jitter, 1+jitter); event counts may differ")
    parser.add_argument("--unlimited_injection",action="store_true",help="Remove per-evolution injection budget",)
    parser.add_argument('--matched_aux_budget', action='store_true', help='20261115 per-seed actual auxiliary interaction protocol (Ant/HC)')
    parser.add_argument("--log_candidates", action="store_true",
                        help="Save every candidate transition (including rejected ones) of each evolution "
                             "event into <run_dir>/candidates/*.npz for offline admission-bias analysis")
    parser.add_argument("--no_overwrite", action="store_true", help="Compatibility flag; existing run directories are always protected")
    parser.add_argument("--print_config_only", action="store_true", help="Print the resolved configuration and exit")

    parser.add_argument("--target_reward", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--ema_beta", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--actor_lr", type=float, default=None)
    parser.add_argument("--actor_lr_final", type=float, default=None)
    parser.add_argument("--actor_lr_decay_start_step", type=int, default=None)
    parser.add_argument("--actor_lr_decay_steps", type=int, default=None)
    parser.add_argument("--critic_lr", type=float, default=None)
    parser.add_argument("--policy_noise", type=float, default=None)
    parser.add_argument("--noise_clip", type=float, default=None)
    parser.add_argument("--policy_freq", type=int, default=None)
    parser.add_argument("--start_steps", type=int, default=None)
    parser.add_argument("--update_after", type=int, default=None)
    parser.add_argument("--noise_initial", type=float, default=None)
    parser.add_argument("--noise_final", type=float, default=None)
    parser.add_argument("--noise_decay_steps", type=int, default=None)
    parser.add_argument("--stagnation_patience_limit", type=int, default=None)
    parser.add_argument("--stagnation_detector", choices=["legacy", "normalized_windows"], default=None)
    parser.add_argument("--stagnation_progress_threshold", type=float, default=None)
    parser.add_argument("--stagnation_variation_threshold", type=float, default=None)
    parser.add_argument("--min_mean_reward_to_evolve", type=float, default=None)
    parser.add_argument("--min_steps_before_evolution", type=int, default=None)
    parser.add_argument("--evolution_cooldown_steps", type=int, default=None)
    parser.add_argument("--sigma_conservative", type=float, default=None)
    parser.add_argument("--sigma_explorer", type=float, default=None)
    parser.add_argument("--sigma_destroyer", type=float, default=None)
    parser.add_argument("--max_inject_per_evo", type=int, default=None)
    parser.add_argument("--evolution_horizon", type=int, default=None)
    parser.add_argument("--evolution_stop_reward", type=float, default=None)
    parser.add_argument("--evolution_stop_survival_rate", type=float, default=None)
    parser.add_argument("--evolution_stop_survival_min_reward", type=float, default=None)
    parser.add_argument("--trajectory_keep_quantile", type=float, default=None)
    parser.add_argument("--trajectory_keep_quantile_conservative", type=float, default=None)
    parser.add_argument("--trajectory_keep_quantile_explorer", type=float, default=None)
    parser.add_argument("--trajectory_keep_quantile_destroyer", type=float, default=None)
    parser.add_argument("--parent_return_tolerance_conservative", type=float, default=None)
    parser.add_argument("--parent_return_tolerance_explorer", type=float, default=None)
    parser.add_argument("--parent_return_tolerance_destroyer", type=float, default=None)
    parser.add_argument("--transition_value_mode", choices=["current_q", "one_step_target"], default=None)
    parser.add_argument("--mutation_style", choices=["legacy", "output_masked"], default=None)
    parser.add_argument("--mutation_fraction", type=float, default=None)
    parser.add_argument("--eval_every_steps", type=int, default=None)
    parser.add_argument("--eval_actor", choices=["ema", "actor"], default=None, help="Actor used for periodic/final evaluation")
    parser.add_argument("--eval_episodes", type=int, default=None)
    parser.add_argument("--eval_seed_mode", choices=["fixed", "rotating"], default=None)
    parser.add_argument("--eval_seed_stride", type=int, default=None)
    parser.add_argument("--final_eval_episodes", type=int, default=None)
    parser.add_argument("--checkpoint_window", type=int, default=None)
    return parser

def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch did not detect an available GPU")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    keys = [
        "td3_baseline",
        "vanilla_td3",
        "target_reward",
        "gamma",
        "tau",
        "ema_beta",
        "batch_size",
        "actor_lr",
        "actor_lr_final",
        "actor_lr_decay_start_step",
        "actor_lr_decay_steps",
        "critic_lr",
        "policy_noise",
        "noise_clip",
        "policy_freq",
        "start_steps",
        "update_after",
        "noise_initial",
        "noise_final",
        "noise_decay_steps",
        "stagnation_patience_limit",
        "stagnation_detector",
        "stagnation_progress_threshold",
        "stagnation_variation_threshold",
        "min_mean_reward_to_evolve",
        "min_steps_before_evolution",
        "evolution_cooldown_steps",
        "sigma_conservative",
        "sigma_explorer",
        "sigma_destroyer",
        "max_inject_per_evo",
        "evolution_horizon",
        "evolution_stop_reward",
        "evolution_stop_survival_rate",
        "evolution_stop_survival_min_reward",
        "trajectory_keep_quantile",
        "trajectory_keep_quantile_conservative",
        "trajectory_keep_quantile_explorer",
        "trajectory_keep_quantile_destroyer",
        "parent_return_tolerance_conservative",
        "parent_return_tolerance_explorer",
        "parent_return_tolerance_destroyer",
        "transition_value_mode",
        "mutation_style",
        "mutation_fraction",
        "eval_every_steps",
        "eval_actor",
        "eval_episodes",
        "eval_seed_mode",
        "eval_seed_stride",
        "final_eval_episodes",
        "checkpoint_window",
    ]
    return {key: getattr(args, key) for key in keys if getattr(args, key) is not None}

def prepare_run_dir(args: argparse.Namespace, config: EnvironmentConfig) -> Path:
    root = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent.parent / "TEDI_runs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"{config.env_name}_s{args.seed}_{timestamp}"
    run_name = safe_name(args.run_name or default_name)
    run_dir = root / safe_name(config.env_name) / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}. Choose a new run name.")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    set_global_seed(args.seed)
    config = EnvironmentConfig.create(args.env, cli_overrides(args))
    total_steps = int(args.steps if args.steps is not None else config.default_steps)
    if total_steps <= 0:
        raise ValueError("Training steps must be a positive integer")
    matched_budget = None
    if args.matched_aux_budget:
        from .matched_budget import MatchedBudget
        if args.disable_evolution or args.disable_trigger or args.disable_filtering or args.unlimited_injection or args.pop_size != 6 or total_steps != 1000000:
            parser.error('matched auxiliary protocol requires 1M main steps, 6 probes and standard evolution/filtering')
        try:
            matched_budget = MatchedBudget(args.env, args.seed, args.trigger_mode)
        except KeyError:
            parser.error('matched protocol only supports documented Ant/HC paper seeds')
        expected_actor = 'actor' if args.env == 'Ant-v2' else 'ema'
        if config.eval_actor != expected_actor:
            parser.error(f'protocol requires --eval_actor {expected_actor}')
        if config.evolution_horizon != (500 if args.env == 'Ant-v2' else 1000):
            parser.error('matched protocol requires original rollout horizon')
        if args.env == 'Ant-v2' and (config.use_parent_return_gate or any(config.trajectory_quantile(r) != 0 for r in ('conservative','explorer','destroyer'))):
            parser.error('Ant atomic budget stop requires no parent gate and zero trajectory quantiles')

    print("\n========== RESOLVED ENVIRONMENT CONFIGURATION ==========")
    print(json.dumps(config.serializable(), ensure_ascii=False, indent=2))
    print(f"Main-environment training steps: {total_steps}")
    if matched_budget:
        print(f'Matched auxiliary target: {matched_budget.target}; schedule: {matched_budget.schedule}')
    print("Shadow-environment interactions are recorded separately from the main training budget.")
    if args.print_config_only:
        return 0

    import gymnasium as gym

    device = choose_device(args.device)
    run_dir = prepare_run_dir(args, config)
    metadata = {
        "environment": config.env_name,
        "seed": args.seed,
        "main_env_steps": total_steps,
        "population_size": args.pop_size,
        "device": str(device),
        "replay_on_cpu": bool(args.replay_on_cpu),
        "evolution_disabled": bool(args.disable_evolution),
        "trigger_disabled": bool(args.disable_trigger),
        "filtering_disabled": bool(args.disable_filtering),
        "trigger_interval": int(args.trigger_interval),
        "trigger_mode": str(args.trigger_mode),
        "trigger_jitter": float(args.trigger_jitter),
        "matched_aux_protocol": '20261115' if matched_budget else None,
        "aux_target": matched_budget.target if matched_budget else None,
        "planned_schedule": matched_budget.schedule if matched_budget else None,
        "unlimited_injection": bool(args.unlimited_injection),
        "log_candidates": bool(args.log_candidates),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "gymnasium_version": gym.__version__,
        "configuration": config.serializable(),
    }
    write_json(run_dir / "run_config.json", metadata)

    print(f"[TRAINING START] Environment={config.env_name}, seed={args.seed}, device={device}, output={run_dir}")
    framework = TEDIFramework(
        config=config,
        total_steps=total_steps,
        pop_size=args.pop_size,
        seed=args.seed,
        run_dir=run_dir,
        device=device,
        replay_on_cpu=args.replay_on_cpu,
        disable_evolution=args.disable_evolution,
        disable_trigger=args.disable_trigger,
        disable_filtering=args.disable_filtering,
        unlimited_injection=args.unlimited_injection,
        trigger_interval=args.trigger_interval,
        trigger_mode=args.trigger_mode,
        trigger_jitter=args.trigger_jitter,
        log_candidates=args.log_candidates,
        matched_budget=matched_budget,
    )
    try:
        framework.train()
    finally:
        framework.close()
    return 0
