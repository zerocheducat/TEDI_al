# -*- coding: utf-8 -*-
"""Replay frozen candidate admission with controlled reward perturbations.

Use --verify to compare zero-noise stage masks and admission counts. Empty
admissions remain explicit in CSV output; conditional means are NaN.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))


import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

ROLES = ("conservative", "explorer", "destroyer")


def rule_ant(role_id, r, q, mr, bt, term, rng):
    out = np.zeros(len(r), dtype=bool)
    m0, m1, m2 = role_id == 0, role_id == 1, role_id == 2
    out[m0] = (q[m0] > bt) & (r[m0] >= mr)
    out[m1] = (q[m1] > bt * 0.9) & (r[m1] > 0) & (rng.random(int(m1.sum())) < 0.5)
    out[m2] = (r[m2] > mr * 2.5) | (r[m2] > 5.0)
    return out


def rule_hc(role_id, r, q, mr, bt, term, rng):
    out = np.zeros(len(r), dtype=bool)
    m0, m1, m2 = role_id == 0, role_id == 1, role_id == 2
    out[m0] = (r[m0] >= mr) & (q[m0] >= bt)
    out[m1] = (~term[m1].astype(bool)) & (r[m1] >= mr * 0.75) & (q[m1] >= bt * 0.90)
    out[m2] = (~term[m2].astype(bool)) & (r[m2] >= mr * 0.60) & (q[m2] >= bt * 0.80)
    return out


def rule_hopper(role_id, r, q, mr, bt, term, rng):
    out = np.zeros(len(r), dtype=bool)
    m0, m1, m2 = role_id == 0, role_id == 1, role_id == 2
    out[m0] = (r[m0] >= mr) & (q[m0] > bt)
    out[m1] = (r[m1] > mr * 0.8) & (q[m1] > bt * 0.8)
    out[m2] = (r[m2] > mr * 0.9) & (~term[m2].astype(bool))
    return out


def rule_swimmer(role_id, r, q, mr, bt, term, rng):
    out = np.zeros(len(r), dtype=bool)
    m0, m1, m2 = role_id == 0, role_id == 1, role_id == 2
    out[m0] = r[m0] >= mr
    out[m1] = (r[m1] > mr * 0.8) & (q[m1] > bt * 0.5)
    out[m2] = (r[m2] > mr * 1.5) & (~term[m2].astype(bool))
    return out


RULES = {"Ant-v2": rule_ant, "HalfCheetah-v2": rule_hc,
         "Hopper-v2": rule_hopper, "Swimmer-v2": rule_swimmer}


def admit(qualified, pop_pass, par_pass, policy_idx, policy_roles, returns, budget, rng):
    injectable = qualified & pop_pass[policy_idx] & par_pass[policy_idx]
    per_pol = []
    for i in range(len(returns)):
        pos = np.flatnonzero(injectable & (policy_idx == i))
        rng.shuffle(pos)
        per_pol.append(pos.tolist())
    members = {r: [i for i in range(len(returns)) if policy_roles[i] == r and per_pol[i]]
               for r in ROLES}
    cur = {i: 0 for i in range(len(returns))}
    mc = {r: 0 for r in ROLES}
    chosen = []
    while budget is None or len(chosen) < budget:
        added = False
        for role in ROLES:
            mem = members[role]
            for _ in range(len(mem)):
                p = mem[mc[role] % len(mem)]
                mc[role] += 1
                k = cur[p]
                if k >= len(per_pol[p]):
                    continue
                chosen.append(per_pol[p][k])
                cur[p] += 1
                added = True
                break
            if budget is not None and len(chosen) >= budget:
                break
        if not added:
            break
    return np.asarray(chosen, dtype=int)


def screen(z, meta, cfg, env, eps, rng):
    r0 = z["reward"].astype(np.float64)
    q0 = z["q_value"].astype(np.float64)
    step = z["step_idx"].astype(int)
    pol = z["policy_idx"].astype(int)
    role = z["role_id"].astype(int)
    term = z["terminal"].astype(np.float64)
    roles_list = meta["policy_roles"]
    returns0 = np.asarray(meta["policy_returns"], dtype=np.float64)
    parent = meta.get("parent_return")
    budget = None if cfg.get("unlimited_injection", False) else int(meta["fixed_injection_budget"])
    q_offset = (meta.get("transition_value_mode") == "one_step_target")

    r = r0 + eps
    q = q0 + (eps if q_offset else 0.0)
    returns = returns0 + np.bincount(pol, weights=eps, minlength=len(returns0))

    rule = RULES[env]
    qualified = np.zeros(len(r0), dtype=bool)
    for s in np.unique(step):
        g = np.flatnonzero(step == s)
        qualified[g] = rule(role[g], r[g], q[g], float(np.mean(r[g])),
                            float(np.median(q[g])), term[g], rng)

    qb = {}
    for rl in ROLES:
        value = cfg.get("trajectory_keep_quantile_" + rl)
        qb[rl] = float(cfg["trajectory_keep_quantile"] if value is None else value)
    thr = {rl: float(np.quantile(returns, qb[rl])) for rl in ROLES}
    pop = np.array([returns[i] >= thr[roles_list[i]] for i in range(len(returns))])
    if not cfg.get("use_parent_return_gate", False):
        par = np.ones(len(returns), dtype=bool)
    else:
        if parent is None:
            raise ValueError("Parent return is missing while the parent gate is enabled")
        tol = {rl: float(cfg.get("parent_return_tolerance_%s" % rl, 0.0) or 0.0) for rl in ROLES}
        par = np.array([returns[i] >= float(parent) - tol[roles_list[i]]
                        for i in range(len(returns))])
    if cfg.get("filtering_disabled", False):
        qualified[:] = True
        pop[:] = True
        par[:] = True
    chosen = admit(qualified, pop, par, pol, roles_list, returns, budget, rng)
    return dict(qualified=qualified, pop=pop, par=par, chosen=chosen, gated=qualified & pop[pol] & par[pol])


def verify_masks(z, out):
    qualified = z["qualified"].astype(bool)
    if "gated" not in z.files:
        return False
    gated = qualified & z["gated"].astype(bool)
    selected = z["selected"].astype(bool)
    policies = z["policy_idx"].astype(int)
    for field, key in (("population_gate_pass", "pop"), ("parent_gate_pass", "par")):
        if field in z.files and not np.array_equal(z[field].astype(bool), out[key][policies]):
            return False
    return bool(np.array_equal(qualified, out["qualified"])
                and np.array_equal(gated, out["gated"])
                and np.all(~selected | out["gated"])
                and int(selected.sum()) == len(out["chosen"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--dump-csv", default=None)
    ap.add_argument("--dump-empty", action="store_true",
                    help="Compatibility flag; empty admissions are always exported")
    args = ap.parse_args()

    dirs = []
    for pat in args.runs:
        dirs.extend(sorted(glob.glob(pat)))
    dirs = list(dict.fromkeys(Path(d).resolve() for d in dirs if (Path(d) / "candidates").is_dir()))
    if not dirs:
        print("没有找到带 candidates 的目录")
        return 1
    if args.reps <= 0 or any(s < 0 or not np.isfinite(s) for s in args.scales):
        raise ValueError("Repetitions must be positive and scales finite and nonnegative")
    if args.dump_csv and Path(args.dump_csv).exists():
        raise FileExistsError(f"Diagnostic output already exists: {args.dump_csv}")
    contexts = {}
    for d in dirs:
        raw = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
        cfg = dict(raw["configuration"])
        for flag in ("filtering_disabled", "unlimited_injection"):
            cfg[flag] = bool(raw.get(flag, False))
        env = raw["environment"]
        if env not in RULES:
            raise ValueError(f"Unsupported environment: {env}")
        if args.verify and env == "Ant-v2" and not cfg["filtering_disabled"]:
            raise ValueError("Exact Ant verification requires the recorded stochastic qualifier RNG state")
        contexts[d] = (cfg, env)
    events = []
    for d in dirs:
        for p in sorted((d / "candidates").glob("*.npz")):
            events.append((d, np.load(p), json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))))
    if not events:
        raise ValueError("No candidate batches found")
    print(f"Batches={len(events)}, runs={len(dirs)}; each run uses its own configuration.")
    print("Stage masks and counts are verified; randomized admission IDs are not asserted to match.")
    print("父策略回报：固定为干净记录值\n")

    if args.verify:
        print("=" * 76)
        print("零噪声核对：重算 vs 记录")
        print("=" * 76)
        print(f"{'run':24s}{'批次':>5}{'合格(记/算)':>15}{'门控(记/算)':>15}{'入池(记/算)':>13}  结论")
        bad = 0
        for d, z, meta in events:
            cfg, env = contexts[d]
            out = screen(z, meta, cfg, env, np.zeros(len(z["reward"])), np.random.default_rng(0))
            rq = int(z["qualified"].sum())
            rg = int(z["gated"].sum()) if "gated" in z.files else -1
            ra = int(z["selected"].sum())
            ok = verify_masks(z, out)
            bad += 0 if ok else 1
            print(f"{d.parent.name[-23:]:24s}{meta['event_index']:>5}"
                  f"{rq:>6}/{int(out['qualified'].sum()):<8}"
                  f"{rg:>6}/{int(out['gated'].sum()):<8}"
                  f"{ra:>5}/{len(out['chosen']):<7}  {'OK' if ok else '不一致!'}")
        print()
        print("全部一致 ✅" if bad == 0 else f"{bad} 个批次不一致 ❌")
        return 0 if bad == 0 else 2

    rows_csv = []
    for scale in args.scales:
        print("=" * 110)
        print(f"尺度 {scale:g} × std(clean_reward)")
        print("=" * 110)
        print(f"{'run':22s}{'批次':>5}{'合格ε':>8}{'门控ε':>8}{'入池ε':>8}{'选择差':>8}"
              f"{'空注入%':>8}{'入池(均)':>9}{'target前':>9}{'target后':>9}{'裁剪%':>7}")
        per_event = []
        for d, z, meta in events:
            cfg, env = contexts[d]
            r0 = z["reward"].astype(np.float64)
            pol = z["policy_idx"].astype(int)
            y_pre = z["y_pre_clean"].astype(np.float64) if "y_pre_clean" in z.files \
                else np.full(len(r0), np.nan)
            y_post0 = z["y_post_clean"].astype(np.float64) if "y_post_clean" in z.files \
                else y_pre.copy()
            qmin, qmax = meta.get("target_q_min"), meta.get("target_q_max")
            c = scale * float(np.std(r0))
            rng = np.random.default_rng(args.seed * 131 + int(meta["event_index"]) * 17
                                        + int(scale * 1000))
            rec = dict(qe=[], ge=[], ae=[], gap=[], dpre=[], dpost=[], adm=[], hit=[], empty=0)
            for repeat_id in range(args.reps):
                eps = rng.normal(0.0, c, size=len(r0)) if c > 0 else np.zeros(len(r0))
                out = screen(z, meta, cfg, env, eps, rng)
                qual, gated, chosen = out["qualified"], out["gated"], out["chosen"]
                rec["qe"].append(float(eps[qual].mean()) if qual.any() else np.nan)
                rec["ge"].append(float(eps[gated].mean()) if gated.any() else np.nan)
                if len(chosen) == 0:
                    rec["empty"] += 1
                    rec["adm"].append(0)
                    if args.dump_csv:
                        rows_csv.append([str(d), meta["event_index"], scale,
                                         float(eps.mean()), rec["qe"][-1], rec["ge"][-1],
                                         np.nan, 0, np.nan, repeat_id, True])
                    continue
                m = np.zeros(len(r0), dtype=bool)
                m[chosen] = True
                rec["ae"].append(float(eps[m].mean()))
                rec["gap"].append(float(eps[m].mean() - eps.mean()))
                rec["adm"].append(int(m.sum()))
                dy = y_pre[m] + eps[m]
                rec["dpre"].append(float(np.nanmean(eps[m])))
                post = dy.copy()
                if qmin is not None:
                    post = np.maximum(post, float(qmin))
                if qmax is not None:
                    post = np.minimum(post, float(qmax))
                rec["dpost"].append(float(np.nanmean(post - y_post0[m])))
                if "hit_lower" in z.files:
                    rec["hit"].append(float((z["hit_lower"][m] | z["hit_upper"][m]).mean() * 100))
                if args.dump_csv:
                    rows_csv.append([str(d), meta["event_index"], scale,
                                     float(eps.mean()), rec["qe"][-1], rec["ge"][-1],
                                     rec["ae"][-1], int(m.sum()), rec["dpre"][-1], repeat_id, False])

            def mm(k):
                v = np.asarray(rec[k], dtype=float)
                v = v[~np.isnan(v)]
                return float(v.mean()) if len(v) else float("nan")

            diff = mm("gap")
            per_event.append(dict(run=d.parent.name, batch=meta["event_index"], diff=diff,
                                  empty=rec["empty"] / args.reps,
                                  adm=float(np.mean(rec["adm"])) if rec["adm"] else 0.0))
            print(f"{d.parent.name[-21:]:22s}{meta['event_index']:>5}{mm('qe'):>8.3f}"
                  f"{mm('ge'):>8.3f}{mm('ae'):>8.3f}{diff:>8.3f}"
                  f"{rec['empty']/args.reps*100:>7.1f}%"
                  f"{np.mean(rec['adm']) if rec['adm'] else 0:>9.1f}"
                  f"{mm('dpre'):>9.3f}{mm('dpost'):>9.3f}{mm('hit'):>6.1f}%")

        by_seed = {}
        for e in per_event:
            by_seed.setdefault(e["run"], []).append(e)
        print()
        for name, es in by_seed.items():
            dd = np.asarray([e["diff"] for e in es], dtype=float)
            dd = dd[~np.isnan(dd)]
            emp = float(np.mean([e["empty"] for e in es]))
            if len(dd):
                print(f"    {name[-22:]:24s} 事件 {len(es):2d}  选择差中位 {np.median(dd):+.3f}  "
                      f"IQR {np.percentile(dd,25):+.3f}~{np.percentile(dd,75):+.3f}  空注入 {emp*100:.1f}%")
        pooled = [e["diff"] for e in per_event if e["diff"] == e["diff"]]
        if pooled:
            a = np.asarray(pooled)
            ci = np.percentile(a, [2.5, 97.5])
            print(f"    全体：事件 {len(a)}  中位 {np.median(a):+.3f}  均值 {a.mean():+.3f}  "
                  f"95%(事件级) [{ci[0]:+.3f}, {ci[1]:+.3f}]  为正 {int((a > 0).sum())}/{len(a)}")
        print(f"    空注入比例（事件×重复）：{np.mean([e['empty'] for e in per_event])*100:.2f}%")
        print()

    if args.dump_csv:
        with open(args.dump_csv, "x", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["run", "batch", "scale", "eps_all", "eps_qualified", "eps_gated",
                        "eps_admitted", "admitted_n", "target_shift_pre", "repeat_id", "empty_admission"])
            # Empty admissions remain explicit; conditional noise means are NaN.
            w.writerows(rows_csv)
        print(f"逐重复明细已写出 {args.dump_csv}（{len(rows_csv)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
