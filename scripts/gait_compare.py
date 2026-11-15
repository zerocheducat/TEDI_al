# -*- coding: utf-8 -*-
"""Gait compare."""


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


def build_actor(state_dict, max_action):
    from tedi.networks import RobustActor

    state_dim = int(state_dict["l1.weight"].shape[1])
    action_dim = int(state_dict["l3.weight"].shape[0])
    use_ln = "ln1.weight" in state_dict
    net = RobustActor(state_dim, action_dim, max_action, use_layernorm=use_ln)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def load_run(run_dir: Path, weight: str):
    ck_path = run_dir / weight
    if not ck_path.is_file():
        ck_path = run_dir / "latest_checkpoint.pth"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    td3 = ck.get("td3", ck)
    env_name = str(ck.get("env_name") or (ck.get("config") or {}).get("environment"))
    return ck, td3, env_name


def rollout(net, env, seeds, max_action):
    traj, returns, lengths, survived, terminated_flags = [], [], [], 0, []
    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        buf, total, steps = [], 0.0, 0
        while True:
            buf.append(np.asarray(obs, dtype=np.float32).copy())
            with torch.inference_mode():
                a = net(torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)).numpy().reshape(-1)
            a = np.clip(a, -max_action, max_action)
            obs, r, term, trunc, _ = env.step(a)
            total += float(r)
            steps += 1
            if term or trunc:
                break
        traj.append(np.asarray(buf))
        returns.append(total)
        lengths.append(steps)
        survived += int(bool(trunc and not term))
        terminated_flags.append(bool(term))
    return dict(traj=traj, returns=returns, lengths=lengths,
                survival=survived / len(seeds), terminated=terminated_flags)


def dominant_period(x: np.ndarray):
    """Estimate the dominant period from the recorded signal."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 16:
        return float("nan")
    x = x - x.mean()
    if np.allclose(x, 0.0):
        return float("nan")
    spec = np.abs(np.fft.rfft(x))
    spec[0] = 0.0
    k = int(np.argmax(spec))
    if k <= 0:
        return float("nan")
    return float(x.size) / float(k)


def summarise(tag, res):
    ret = np.asarray(res["returns"], dtype=np.float64)
    ln = np.asarray(res["lengths"], dtype=np.float64)
    print(f"\n== {tag}")
    print(f"   生存率 {res['survival']:.3f}  平均回报 {ret.mean():.2f} ± {ret.std(ddof=1) if len(ret) > 1 else 0:.2f}"
          f"  平均回合长度 {ln.mean():.1f}")
    print(f"   提前终止 {sum(res['terminated'])}/{len(res['terminated'])}")
    n = min(t.shape[0] for t in res["traj"])
    stack = np.vstack([t[:n] for t in res["traj"]])
    if stack.ndim == 1:
        stack = stack.reshape(-1, 1)
    std = stack.std(axis=0)
    top = np.argsort(-std)[:6]
    print(f"   观测维度 {stack.shape[1]}，波动最大的 6 维: {top.tolist()}")
    per = {}
    for d in top:
        per[int(d)] = dominant_period(stack[:, d])
    print("   这些维度的主周期（步）: " + ", ".join(f"dim{d}={p:.0f}" for d, p in per.items() if p == p))
    return dict(n=n, stack=stack, top=top, periods=per)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="TEDI run 目录")
    ap.add_argument("--b", required=True, help="matched run 目录")
    ap.add_argument("--weight", default="latest_checkpoint.pth")
    ap.add_argument("--which", choices=["actor", "ema"], default="actor",
                    help="用哪个网络评估；两边必须一致（TEDI 历史结果的评估口径是 ema）")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="gait_compare.png")
    args = ap.parse_args()

    import gymnasium as gym

    run_a, run_b = Path(args.a).expanduser().resolve(), Path(args.b).expanduser().resolve()
    ck_a, td3_a, env_a = load_run(run_a, args.weight)
    ck_b, td3_b, env_b = load_run(run_b, args.weight)
    if env_a != env_b:
        print(f"环境不一致: {env_a} vs {env_b}")
        return 2

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    env = gym.make(env_a)
    max_action = float(np.asarray(env.action_space.high).reshape(-1)[0])
    base_seed = int(ck_a.get("seed") or 0)
    seeds = [base_seed + 500_000 + i for i in range(args.episodes)]

    net_key = "actor" if args.which == "actor" else "ema_actor"
    net_a = build_actor(td3_a[net_key], max_action).to(device)
    net_b = build_actor(td3_b[net_key], max_action).to(device)
    print(f"环境 {env_a}  种子 {seeds[0]}..{seeds[-1]}  网络 {args.which}")

    ra = rollout(net_a, env, seeds, max_action)
    rb = rollout(net_b, env, seeds, max_action)
    env.close()

    sa = summarise(f"A: {run_a.name}", ra)
    sb = summarise(f"B: {run_b.name}", rb)

    d_ret = np.mean(ra["returns"]) - np.mean(rb["returns"])
    d_sur = ra["survival"] - rb["survival"]
    print(f"\n== 差值 (A - B)  回报 {d_ret:+.2f}   生存率 {d_sur:+.3f}")

    out = Path(args.out).with_suffix(".json")
    write_new_text(out, json.dumps(dict(
        env=env_a, episodes=args.episodes, seeds=seeds,
        a=dict(run=run_a.name, survival=ra["survival"], returns=ra["returns"], lengths=ra["lengths"]),
        b=dict(run=run_b.name, survival=rb["survival"], returns=rb["returns"], lengths=rb["lengths"]),
    ), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"   数据已保存 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
