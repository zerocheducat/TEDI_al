# -*- coding: utf-8 -*-
"""Compare matched vs tedi."""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))


import csv
import json
from itertools import product
from pathlib import Path

import numpy as np

NEW_ROOT = Path(r"F:\workwork\workworkwork\6572652")
CANON = Path(r"F:\workwork\workworkwork\01_canonical")
OUT = Path(r"F:\workwork\workworkwork\03_notes")

CASES = {
    "Ant-v2": dict(
        tedi=CANON / "Ant" / "TEDI", matched=NEW_ROOT / "Ant-v2",
        seeds=[15, 20, 30, 50, 55],
    ),
    "HalfCheetah-v2": dict(
        tedi=CANON / "HalfCheetah" / "TEDI", matched=NEW_ROOT / "HalfCheetah-v2",
        seeds=[15, 10, 25, 40, 55],
    ),
    "Swimmer-v2": dict(
        tedi=CANON / "Swimmer" / "TEDI", matched=NEW_ROOT / "Swimmer-v2",
        seeds=[10, 25, 40, 45, 55],
    ),
}

SHORT = {"Ant-v2": "Ant", "HalfCheetah-v2": "HC", "Swimmer-v2": "Swimmer"}


def find_run(root: Path, seed: int) -> Path | None:
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.endswith(f"s{seed}") and (d / "training_summary.json").is_file():
            return d
    return None


def load(d: Path) -> dict:
    summ = json.loads((d / "training_summary.json").read_text(encoding="utf-8"))
    events, injected, shadow = 0, 0, 0
    with (d / "events.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if '"type": "evolution"' not in line:
                continue
            e = json.loads(line)
            events += 1
            injected += int(e.get("injected_transitions", 0))
            shadow = max(shadow, int(e.get("cumulative_shadow_steps", 0)))
    returns = np.load(d / "eval_returns.npy").ravel()
    surv = np.load(d / "eval_survival_rates.npy").ravel()
    lens = np.load(d / "eval_lengths.npy").ravel()
    return dict(
        events=events, injected=injected, shadow=shadow,
        peak=float(returns.max()),
        final=float(summ["current_final_mean"]),
        stable=float(summ["best_stable_final_mean"]) if summ.get("best_stable_final_mean") is not None else float("nan"),
        surv_final=float(summ["current_final_survival_rate"]),
        surv_last100=float(np.mean(surv[-100:])), surv_all=float(np.mean(surv)),
        len_last100=float(np.mean(lens[-100:])),
        wall=float(summ.get("wall_time_seconds", float("nan"))) / 60.0,
        curve=returns,
    )


def p_sign(d: np.ndarray) -> float:
    n = d.size
    obs = abs(d.mean())
    hits = sum(abs((d * np.array(s)).mean()) >= obs - 1e-12 for s in product((1, -1), repeat=n))
    return hits / 2 ** n


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=list(CASES), choices=list(CASES))
    args = parser.parse_args()
    envs = args.envs
    tag = "_".join(SHORT[e] for e in envs)

    rows = []
    for env in envs:
        cfg = CASES[env]
        for seed in cfg["seeds"]:
            td, md = find_run(cfg["tedi"], seed), find_run(cfg["matched"], seed)
            if td is None or md is None:
                raise FileNotFoundError(f"Missing fixed-seed pair: {env}, seed={seed}")
            t, m = load(td), load(md)
            rows.append(dict(env=env, seed=seed, t=t, m=m))

    hdr = (f"{'env':<12}{'seed':>5}{'TEDI峰':>9}{'m峰':>8}{'TEDI末':>9}{'m末':>8}{'Δ末':>8}"
           f"{'TEDI稳':>9}{'m稳':>8}{'T存活':>7}{'m存活':>7}{'事件':>5}{'注入':>7}{'shadow':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        t, m = r["t"], r["m"]
        print(f"{r['env']:<12}{r['seed']:>5}{t['peak']:>9.1f}{m['peak']:>8.1f}"
              f"{t['final']:>9.1f}{m['final']:>8.1f}{t['final'] - m['final']:>+8.1f}"
              f"{t['stable']:>9.1f}{m['stable']:>8.1f}"
              f"{t['surv_final']:>7.2f}{m['surv_final']:>7.2f}"
              f"{t['events']:>5}{t['injected']:>7}{t['shadow']:>7}")

    print()
    for env in envs:
        sub = [r for r in rows if r["env"] == env]
        if not sub:
            continue
        print(f"[{env}]  n={len(sub)}（按种子配对）")
        for name, key in (("最终", "final"), ("峰值", "peak"),
                          ("稳定末值", "stable"), ("存活率", "surv_final"), ("末100存活", "surv_last100")):
            t = np.array([r["t"][key] for r in sub])
            m = np.array([r["m"][key] for r in sub])
            d = t - m
            wins = int((d > 0).sum()) if key != "surv_final" or True else 0
            print(f"  {name:<9}TEDI {t.mean():9.2f}  matched {m.mean():9.2f}  "
                  f"Δ {d.mean():+8.2f} ± {d.std(ddof=1) / np.sqrt(len(d)):6.2f}(SE)  "
                  f"TEDI胜 {wins}/{len(d)}  p={p_sign(d):.3f}")
        ti = np.array([r["t"] for r in sub], dtype=object)
        print(f"  注入  TEDI 共 {sum(r['t']['injected'] for r in sub)} 条 / "
              f"{sum(r['t']['events'] for r in sub)} 次事件 / "
              f"{sum(r['t']['shadow'] for r in sub)} 影子步；matched 全 0")
        print(f"  耗时  TEDI {np.mean([r['t']['wall'] for r in sub]):.0f} min  "
              f"matched {np.mean([r['m']['wall'] for r in sub]):.0f} min")

    # ---------------- CSV ----------------
    csv_path = OUT / f"{tag}_matched_vs_TEDI.csv"
    with csv_path.open("x", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["env", "seed",
                    "tedi_peak", "m_peak", "tedi_final", "m_final", "d_final",
                    "tedi_stable", "m_stable", "tedi_surv", "m_surv",
                    "tedi_events", "tedi_injected", "tedi_shadow",
                    "tedi_wall_min", "m_wall_min"])
        for r in rows:
            t, m = r["t"], r["m"]
            w.writerow([r["env"], r["seed"],
                        t["peak"], m["peak"],
                        t["final"], m["final"], t["final"] - m["final"],
                        t["stable"], m["stable"],
                        t["surv_final"], m["surv_final"],
                        t["events"], t["injected"], t["shadow"],
                        t["wall"], m["wall"]])
    print(f"\n已写出 {csv_path}")


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
