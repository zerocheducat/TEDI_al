# -*- coding: utf-8 -*-
"""Read metrics."""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))


import argparse
import json
import os
import statistics
from pathlib import Path


def find_runs(root: Path) -> list[Path]:
    runs = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "training_summary.json" in filenames:
            runs.append(Path(dirpath))
    return sorted(runs)


def load_curve(run_dir: Path) -> tuple[list[float], list[int]]:
    """Load the recorded evaluation steps and returns."""
    vals, steps = [], []
    events = run_dir / "events.jsonl"
    if events.is_file():
        with events.open(encoding="utf-8") as f:
            for line in f:
                if '"evaluation"' not in line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("type") != "evaluation":
                    continue
                vals.append(float(e["evaluation_mean"]))
                steps.append(int(e.get("main_step", len(steps))))
        if vals:
            return vals, steps
    npy = run_dir / "eval_returns.npy"
    if npy.is_file():
        import numpy as np
        arr = np.load(npy).ravel()
        vals = [float(v) for v in arr]
        steps = list(range(len(vals)))
    return vals, steps


def load_events(run_dir: Path) -> dict:
    out = dict(n_evolution=0, injected=0, empty=0, shadow=0, triggers=0,
               by_role={}, first_step=None, last_step=None, steps=[])
    events = run_dir / "events.jsonl"
    if not events.is_file():
        return out
    with events.open(encoding="utf-8") as f:
        for line in f:
            if '"evolution"' not in line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") != "evolution":
                continue
            out["n_evolution"] += 1
            inj = int(e.get("injected_transitions", 0) or 0)
            out["injected"] += inj
            if inj == 0:
                out["empty"] += 1
            if e.get("population_gate_pass"):
                out["triggers"] += 1
            step = e.get("main_step")
            if step is not None:
                out["steps"].append(int(step))
                out["first_step"] = out["first_step"] if out["first_step"] is not None else step
                out["last_step"] = step
            for role, n in (e.get("injected_transitions_by_role") or {}).items():
                out["by_role"][role] = out["by_role"].get(role, 0) + int(n)
    return out


def analyse(run_dir: Path) -> dict:
    summary = json.load(open(run_dir / "training_summary.json", encoding="utf-8"))
    vals, steps = load_curve(run_dir)
    ev = load_events(run_dir)
    peak, peak_step = float("nan"), None
    if vals:
        i = max(range(len(vals)), key=lambda k: vals[k])
        peak, peak_step = vals[i], steps[i]
    shadow = summary.get("shadow_env_steps")
    if shadow is None:
        shadow = summary.get("parent_shadow_steps", 0)
    return dict(
        run=run_dir.name,
        seed=summary.get("seed"),
        peak=peak,
        peak_step=peak_step,
        final=summary.get("current_final_mean"),
        best_stable_final=summary.get("best_stable_final_mean"),
        n_eval=len(vals),
        n_evo=ev["n_evolution"],
        injected=ev["injected"],
        empty=ev["empty"],
        shadow=shadow,
        evolution_events_field=summary.get("evolution_events"),
        periodic_best=summary.get("periodic_best_evaluation"),
        periodic_best_step=summary.get("periodic_best_evaluation_step"),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    runs = find_runs(root)
    if not runs:
        print(f"在 {root} 下没找到 training_summary.json")
        return 1

    rows = [analyse(r) for r in runs]
    hdr = f"{'run':34s}{'seed':>5}{'peak':>10}{'peak@':>9}{'final':>10}{'评估数':>7}{'进化':>5}{'注入':>7}{'空':>4}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{str(r['run'])[:33]:34s}{str(r['seed']):>5}"
              f"{r['peak']:>10.2f}{str(r['peak_step']):>9}"
              f"{(r['final'] if r['final'] is not None else float('nan')):>10.2f}"
              f"{r['n_eval']:>7}{r['n_evo']:>5}{r['injected']:>7}{r['empty']:>4}")

    def agg(key):
        xs = [r[key] for r in rows if r[key] is not None and r[key] == r[key]]
        return (statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0) if xs else (float("nan"), 0.0)

    print("-" * len(hdr))
    for key, label in (("peak", "峰值"), ("final", "最终")):
        m, s = agg(key)
        print(f"  {label:6s} 均值 {m:8.2f}  ±{s:6.2f}   (n={len(rows)})")
    tot_inj = sum(r["injected"] for r in rows)
    tot_evo = sum(r["n_evo"] for r in rows)
    print(f"  进化事件 {tot_evo}   注入转移 {tot_inj}   空注入 {sum(r['empty'] for r in rows)}"
          f"   平均每事件 {tot_inj / tot_evo:.1f} 条" if tot_evo else "")
    sh = [r["shadow"] for r in rows if r["shadow"]]
    if sh:
        print(f"  shadow 步 合计 {sum(sh):,}   每 run {statistics.mean(sh):,.0f}")

    if args.csv:
        import csv
        with open(args.csv, "x", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  已写出 {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
