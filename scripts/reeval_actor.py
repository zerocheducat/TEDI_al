# -*- coding: utf-8 -*-
"""Reeval actor."""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))


from scripts.analysis_io import write_new_text

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def load_checkpoint(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    td3 = ck.get("td3", ck)
    return ck, td3


def actor_dims(sd):
    state_dim = int(sd["l1.weight"].shape[1])
    action_dim = int(sd["l3.weight"].shape[0])
    use_ln = "ln1.weight" in sd
    return state_dim, action_dim, use_ln


def _actor_class():
    """Load the actor implementation compatible with the checkpoint."""
    try:
        import inspect
        from tedi.networks import RobustActor
        if "use_layernorm" in inspect.signature(RobustActor.__init__).parameters:
            return RobustActor
        print("   [提示] 仓库里的 RobustActor 是旧版（没有 use_layernorm），改用内联实现")
    except Exception as exc:
        print(f"   [提示] 导入 tedi.networks 失败（{exc}），改用内联实现")

    import torch.nn as nn
    import torch.nn.functional as F

    class _InlineActor(nn.Module):
        def __init__(self, state_dim, action_dim, max_action, use_layernorm=True):
            super(_InlineActor, self).__init__()
            self.use_layernorm = bool(use_layernorm)
            self.l1 = nn.Linear(state_dim, 256)
            self.l2 = nn.Linear(256, 256)
            self.l3 = nn.Linear(256, action_dim)
            self.ln1 = nn.LayerNorm(256) if self.use_layernorm else nn.Identity()
            self.ln2 = nn.LayerNorm(256) if self.use_layernorm else nn.Identity()
            self.max_action = float(max_action)

        def forward(self, state):
            a = F.relu(self.ln1(self.l1(state)))
            a = F.relu(self.ln2(self.l2(a)))
            return self.max_action * torch.tanh(self.l3(a))

    return _InlineActor


def build_actor(sd, max_action):
    state_dim, action_dim, use_ln = actor_dims(sd)
    net = _actor_class()(state_dim, action_dim, max_action, use_layernorm=use_ln)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state_dict 不匹配: missing={missing[:4]} unexpected={unexpected[:4]}")
    net.eval()
    return net


def eval_seeds(run_dir: Path, episodes: int):
    p = run_dir / "training_summary.json"
    if p.is_file():
        s = json.load(p.open(encoding="utf-8"))
        seeds = s.get("final_evaluation_seeds")
        if seeds:
            return [int(x) for x in seeds]
        seed = int(s.get("seed", 0))
    else:
        seed = 0
    return [seed + 500_000 + i for i in range(episodes)]


def make_env(env_name: str):
    import gymnasium as gym
    env = gym.make(env_name)
    return env


def evaluate(net, env, seeds, max_action: float):
    returns, lengths = [], []
    for seed in seeds:
        state, _ = env.reset(seed=int(seed))
        total, steps = 0.0, 0
        while True:
            with torch.inference_mode():
                a = net(torch.as_tensor(state, dtype=torch.float32).reshape(1, -1)).numpy().reshape(-1)
            a = np.clip(a, -max_action, max_action)
            state, reward, terminated, truncated, _ = env.step(a)
            total += float(reward)
            steps += 1
            if terminated or truncated:
                break
        returns.append(total)
        lengths.append(steps)
    return returns, lengths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="run 目录，可重复")
    ap.add_argument("--which", choices=["both", "actor", "ema"], default="both")
    ap.add_argument("--episodes", type=int, default=0, help="0 = 用该 run 记录的最终评估种子数")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--inspect", action="store_true", help="只检查 checkpoint，不跑仿真")
    ap.add_argument("--out", default=None, help="默认写到 <run>/reeval_actor.json")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    summary = []

    for raw in args.run:
        run_dir = Path(raw).expanduser().resolve()
        ckpt_path = run_dir / "latest_checkpoint.pth"
        if not ckpt_path.is_file():
            print(f"[跳过] 找不到 {ckpt_path}")
            continue

        ck, td3 = load_checkpoint(ckpt_path)
        env_name = str(ck.get("env_name") or (ck.get("config") or {}).get("environment") or "")
        recorded = (ck.get("config") or {}).get("eval_actor")
        have = [k for k in ("actor", "ema_actor") if k in td3]
        seeds = eval_seeds(run_dir, args.episodes)
        if args.episodes:
            seeds = seeds[: args.episodes]

        print(f"\n== {run_dir.name}")
        print(f"   env={env_name}  step={ck.get('step')}  config.eval_actor={recorded!r}")
        print(f"   checkpoint 内可用网络: {have}")
        for k in have:
            sd = td3[k]
            s, a, ln = actor_dims(sd)
            print(f"     {k:10s} state_dim={s} action_dim={a} layernorm={ln} "
                  f"参数量={sum(int(np.prod(v.shape)) for v in sd.values()):,}")
        if "actor" in td3 and "ema_actor" in td3:
            d = max(float((td3["actor"][k].float() - td3["ema_actor"][k].float()).abs().max())
                    for k in td3["actor"] if k in td3["ema_actor"])
            print(f"     actor 与 ema_actor 最大权重差 = {d:.3e}")
        print(f"   评估种子 {len(seeds)} 个: {seeds[:3]}{' ...' if len(seeds) > 3 else ''}")

        if args.inspect:
            continue

        targets = [t for t in (["actor", "ema_actor"] if args.which == "both" else
                               ["actor"] if args.which == "actor" else ["ema_actor"]) if t in td3]
        if not targets:
            print("   [跳过] checkpoint 里没有要评估的网络")
            continue

        env = make_env(env_name)
        max_action = float(np.asarray(env.action_space.high).reshape(-1)[0])
        env.action_space.seed(0)

        result = {}
        for name in targets:
            net = build_actor(td3[name], max_action).to(device)
            rets, lens = evaluate(net, env, seeds, max_action)
            m, sd_ = float(np.mean(rets)), float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
            result[name] = dict(mean=m, std=sd_, returns=rets, lengths=lens)
            print(f"   [{name:9s}] 均值 {m:9.2f}  ± {sd_:7.2f}   最低 {min(rets):8.2f}   最高 {max(rets):8.2f}")
        env.close()


        sp = run_dir / "training_summary.json"
        if sp.is_file():
            s = json.load(sp.open(encoding="utf-8"))
            print(f"   原始记录 current_final_mean = {s.get('current_final_mean')}")
            print(f"   原始记录 eval_actor         = {recorded!r}")
            record = result.get("actor" if recorded == "actor" else "ema_actor")
            if record and s.get("current_final_mean") is not None:
                diff = record["mean"] - float(s["current_final_mean"])
                print(f"   复现校验: 重评 {record['mean']:.2f} vs 记录 {float(s['current_final_mean']):.2f} "
                      f"(差 {diff:+.2f})")

        result["_meta"] = dict(run=run_dir.name, env=env_name, step=ck.get("step"),
                               recorded_eval_actor=recorded, n_seeds=len(seeds),
                               seeds=seeds, which=targets)
        out = Path(args.out) if args.out else run_dir / "reeval_actor.json"
        write_new_text(out, json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"   已写出 {out}")
        summary.append((run_dir.name, result))

    if len(summary) > 1 and not args.inspect:
        print("\n== 汇总 ==")
        print(f"{'run':38s}{'actor':>12}{'ema':>12}{'差值(actor-ema)':>18}")
        for name, r in summary:
            a = r.get("actor", {}).get("mean")
            e = r.get("ema_actor", {}).get("mean")
            d = f"{a - e:+.2f}" if (a is not None and e is not None) else "-"
            print(f"{name[:37]:38s}{(f'{a:.2f}' if a is not None else '-'):>12}"
                  f"{(f'{e:.2f}' if e is not None else '-'):>12}{d:>18}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
