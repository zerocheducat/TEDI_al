# -*- coding: utf-8 -*-
"""Diagnose admission bias using perturbed candidate rewards."""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))


import argparse
import json
from pathlib import Path

import numpy as np

ROLE_NAMES = ("conservative", "explorer", "destroyer")


def qualify_ant(role, r, q, mr, bt, term, rng):
    if role == "conservative":
        return (q > bt) & (r >= mr)
    if role == "explorer":
        return (q > bt * 0.9) & (r > 0) & (rng.random(r.shape[0]) < 0.5)
    return (r > mr * 2.5) | (r > 5.0)


def qualify_hc(role, r, q, mr, bt, term, rng):
    if role == "conservative":
        return (r >= mr) & (q >= bt)
    if role == "explorer":
        return (~term.astype(bool)) & (r >= mr * 0.75) & (q >= bt * 0.90)
    return (~term.astype(bool)) & (r >= mr * 0.60) & (q >= bt * 0.80)


def qualify_hopper(role, r, q, mr, bt, term, rng):
    if role == "conservative":
        return (r >= mr) & (q > bt)
    if role == "explorer":
        return (r > mr * 0.8) & (q > bt * 0.8)
    return (r > mr * 0.9) & (~term.astype(bool))


def qualify_swimmer(role, r, q, mr, bt, term, rng):
    if role == "conservative":
        return r >= mr
    if role == "explorer":
        return (r > mr * 0.8) & (q > bt * 0.5)
    return (r > mr * 1.5) & (~term.astype(bool))


RULES = {
    "Ant-v2": qualify_ant,
    "HalfCheetah-v2": qualify_hc,
    "Hopper-v2": qualify_hopper,
    "Swimmer-v2": qualify_swimmer,
}
STOCHASTIC = {"Ant-v2"}


def pick(qualified, gated, policy_idx, roles, budget, rng):
    """Select admissions under the configured injection budget."""
    n_pol = len(roles)
    injectable = qualified & gated[policy_idx]
    per_policy = []
    for i in range(n_pol):
        pos = np.flatnonzero(injectable & (policy_idx == i))
        rng.shuffle(pos)
        per_policy.append(pos.tolist())

    members_by_role = {r: [i for i in range(n_pol) if roles[i] == r and per_policy[i]]
                       for r in ROLE_NAMES}
    cursor = {i: 0 for i in range(n_pol)}
    member_cursor = {r: 0 for r in ROLE_NAMES}
    chosen = []
    while budget is None or len(chosen) < budget:
        added = False
        for role in ROLE_NAMES:
            members = members_by_role[role]
            for _ in range(len(members)):
                pos_in_members = member_cursor[role] % len(members)
                pol = members[pos_in_members]
                member_cursor[role] += 1
                k = cursor[pol]
                if k >= len(per_policy[pol]):
                    continue
                chosen.append(per_policy[pol][k])
                cursor[pol] += 1
                added = True
                break
            if budget is not None and len(chosen) >= budget:
                break
        if not added:
            break
    return chosen


def replay(z, meta, cfg, env, eps, rng):
    """Recompute admission masks for perturbed candidate rewards."""
    reward0 = z["reward"].astype(np.float64)
    q0 = z["q_value"].astype(np.float64)
    pol = z["policy_idx"].astype(int)
    role_id = z["role_id"].astype(int)
    step = z["step_idx"].astype(int)
    term = z["terminal"].astype(np.float64)

    roles = meta["policy_roles"]
    returns0 = np.asarray(meta["policy_returns"], dtype=np.float64)
    budget = int(meta["fixed_injection_budget"])
    parent_return = meta.get("parent_return")
    use_parent_gate = bool(cfg.get("use_parent_return_gate"))
    q_offset = cfg.get("transition_value_mode") == "one_step_target"

    r = reward0 + eps
    q = q0 + (eps if q_offset else 0.0)
    returns = returns0 + np.bincount(pol, weights=eps, minlength=len(roles))

    quant = {}
    for role in ROLE_NAMES:
        v = cfg.get("trajectory_keep_quantile_%s" % role)
        quant[role] = float(v if v is not None else cfg.get("trajectory_keep_quantile", 0.5))
    tol = {role: float(cfg.get("parent_return_tolerance_%s" % role, 0.0) or 0.0)
           for role in ROLE_NAMES}

    thresholds = {role: float(np.quantile(returns, quant[role])) for role in ROLE_NAMES}
    gated = np.zeros(len(roles), dtype=bool)
    for i in range(len(roles)):
        role = roles[i]
        ok = bool(returns[i] >= thresholds[role])
        if use_parent_gate and parent_return is not None:
            ok = ok and bool(returns[i] >= float(parent_return) - tol[role])
        gated[i] = ok

    rule = RULES[env]
    qualified = np.zeros(len(reward0), dtype=bool)
    for s in np.unique(step):
        g = np.flatnonzero(step == s)
        mr = float(np.mean(r[g]))
        bt = float(np.median(q[g]))
        for role in ROLE_NAMES:
            sel = g[role_id[g] == ROLE_NAMES.index(role)]
            if sel.size == 0:
                continue
            qualified[sel] = rule(role, r[sel], q[sel], mr, bt, term[sel], rng)

    chosen = pick(qualified, gated, pol, roles, budget, rng)
    selected = np.zeros(len(reward0), dtype=bool)
    selected[np.asarray(chosen, dtype=int)] = True
    return dict(qualified=qualified, gated=gated, selected=selected, chosen=chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--levels", type=float, nargs="+", default=[0.1, 0.3])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true",
                    help="只在 0 噪声下比对重算与记录的一致性")
    args = ap.parse_args()

    run_dir = Path(args.run).expanduser().resolve()
    raw = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    cfg = raw.get("configuration", {})
    env = raw.get("environment")
    if env not in RULES:
        print(f"不支持的环境: {env}")
        return 2

    files = sorted((run_dir / "candidates").glob("*.npz"))
    if not files:
        print(f"{run_dir}/candidates 下没有 npz")
        return 1

    print(f"run = {run_dir.name}   env = {env}   "
          f"value_mode = {cfg.get('transition_value_mode')}   "
          f"budget = {cfg.get('max_inject_per_evo')}")
    if env in STOCHASTIC:
        print("注意：该环境的 explorer 规则含 50% 随机门，重算无法逐条复现，只能比分布。")
    print()

    if args.verify:
        if env in STOCHASTIC:
            print("Exact verification requires the recorded qualifier RNG state; no exact match is claimed.")
            return 2
        print("=" * 74)
        print("一致性校验（零噪声）")
        print("=" * 74)
        print(f"{'事件':>5}{'step':>9}{'记录合格':>9}{'重算合格':>9}{'记录选中':>9}{'重算选中':>9}  结论")
        ok_all = True
        for k, p in enumerate(files):
            z = np.load(p)
            meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
            rec_q = int(z["qualified"].sum())
            rec_s = int(z["selected"].sum())
            draws = 20 if env in STOCHASTIC else 1
            qs, ss = [], []
            for d in range(draws):
                rng = np.random.default_rng(args.seed + 100 * k + d)
                out = replay(z, meta, cfg, env, np.zeros(len(z["reward"])), rng)
                qs.append(int(out["qualified"].sum()))
                ss.append(len(out["chosen"]))
            qm, sm = float(np.mean(qs)), float(np.mean(ss))
            qsd = float(np.std(qs, ddof=1)) if len(qs) > 1 else 0.0
            zscore = (rec_q - qm) / qsd if qsd > 0 else 0.0
            if env in STOCHASTIC:
                verdict = "z=%+.2f %s" % (zscore, "OK" if abs(zscore) < 3 else "偏离!")
                ok_all = ok_all and abs(zscore) < 3
            else:
                pol = z["policy_idx"].astype(int)
                masks_match = np.array_equal(z["qualified"].astype(bool), out["qualified"])
                if "gated" in z.files:
                    recorded = z["qualified"].astype(bool) & z["gated"].astype(bool)
                    recomputed = out["qualified"] & out["gated"][pol]
                    masks_match = masks_match and np.array_equal(recorded, recomputed)
                    masks_match = masks_match and bool(np.all(~z["selected"].astype(bool) | recomputed))
                else:
                    masks_match = False
                verdict = "一致 OK" if (masks_match and qm == rec_q and sm == rec_s) else "不一致!"
                ok_all = ok_all and verdict.startswith("一致")
            print(f"{meta['event_index']:>5}{meta['main_step']:>9}{rec_q:>9}{qm:>9.1f}"
                  f"{rec_s:>9}{sm:>9.1f}  {verdict}")
        print()
        if not ok_all:
            print("⚠ 重算与原实现不一致，噪声结果不可信。")
        else:
            print("✅ 重算与原实现一致，可以跑噪声实验。")
        return 0 if ok_all else 2

    print(f"事件数 {len(files)}   重复 {args.reps} 次\n")
    for level in args.levels:
        print("=" * 84)
        print("噪声水平 %.2f × RMS" % level + ("（对照组，b_A 必须为 0）" if level == 0 else ""))
        print("=" * 84)
        print(f"{'事件':>5}{'step':>9}{'候选':>7}{'原注入':>7}{'重放选中':>9}"
              f"{'b_A':>10}{'b_A/sd':>9}{'选中ε均值':>11}{'正占比':>7}{'ε/c':>9}")
        all_b = []
        for k, p in enumerate(files):
            z = np.load(p)
            meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
            reward0 = z["reward"].astype(np.float64)
            rms = float(np.sqrt(np.mean(reward0 ** 2)))
            c = level * rms
            n = len(reward0)
            rng = np.random.default_rng(args.seed + 1000 * k)
            bs, ns, es = [], [], []
            for _ in range(args.reps):
                eps = rng.uniform(-c, c, size=n) if c > 0 else np.zeros(n)
                out = replay(z, meta, cfg, env, eps, rng)
                sel = out["selected"]
                bs.append(float(np.mean(sel * eps)))
                ns.append(int(sel.sum()))
                if sel.any():
                    es.append(float(eps[sel].mean()))
            bs = np.asarray(bs)
            sd = float(bs.std(ddof=1)) if len(bs) > 1 else 0.0
            print(f"{meta['event_index']:>5}{meta['main_step']:>9}{n:>7}"
                  f"{int(z['selected'].sum()):>7}{np.mean(ns):>9.0f}{bs.mean():>10.3f}"
                  f"{(bs.mean()/sd if sd else 0):>9.2f}{(np.mean(es) if es else 0):>11.3f}"
                  f"{((np.asarray(es) > 0).mean() if es else 0):>9.2f}"
                  f"{(np.mean(es)/c*100 if (es and c > 0) else 0):>8.1f}%")
            all_b.append(bs.mean())
        print(f"{'合计':>5}{'':>9}{'':>7}{'':>7}{'':>9}{np.mean(all_b):>10.3f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
