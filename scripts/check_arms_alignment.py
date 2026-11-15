# -*- coding: utf-8 -*-
"""Inspect evaluation settings and recorded step alignment across arms."""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))


import json
import os
import re
from pathlib import Path

CANON = Path(r"F:\workwork\workworkwork\01_canonical")

ARMS = {
    "Ant": ("TEDI", "matched_actor_wrong_actor", "Ant-v2_TEDI_s%d", "Ant-v2_matched_actor_s%d"),
    "HalfCheetah": ("TEDI", "matched", "HalfCheetah-v2_TEDI_s%d", "rebuttal_matched_s%d"),
    "Hopper": ("TEDI", "matched", "Hopper-v2_TEDI_s%d", "rebuttal_matched_s%d"),
    "Swimmer": ("TEDI", "matched", "Swimmer-v2_TEDI_s%d", "rebuttal_matched_s%d"),
}

EVAL_KEYS = ["eval_every_steps", "eval_episodes", "final_eval_episodes", "eval_seed_mode", "eval_actor"]


def load(run: Path):
    steps, events = [], []
    with (run / "events.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if '"type"' not in line:
                continue
            e = json.loads(line)
            if e.get("type") == "evaluation":
                steps.append(int(e["main_step"]))
            elif e.get("type") == "evolution":
                events.append(int(e["main_step"]))
    cfg = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    return steps, events, cfg.get("configuration", {}), cfg


def main():
    print("=" * 96)
    print("一、评估口径（配置层面）")
    print("=" * 96)
    for env, (ta, ma, tp, mp) in ARMS.items():
        td = CANON / env / ta
        md = CANON / env / ma
        if not td.is_dir() or not md.is_dir():
            print(f"{env}: 缺目录")
            continue

        t_run = next((p for p in sorted(td.iterdir()) if p.is_dir() and (p / "run_config.json").is_file()), None)
        m_run = next((p for p in sorted(md.iterdir()) if p.is_dir() and (p / "run_config.json").is_file()), None)
        _, _, tc, _ = load(t_run)
        _, _, mc, _ = load(m_run)
        diffs = []
        for k in EVAL_KEYS:
            tv = tc.get(k, "not_logged")
            mv = mc.get(k, "not_logged")
            if tv == "not_logged" or mv == "not_logged":
                diffs.append(f"{k}: incomplete evidence, TEDI={tv!r}, matched={mv!r}")
                continue
            if tv != mv:
                diffs.append(f"{k}: TEDI={tv!r} vs matched={mv!r}")
        status = "一致 ✅" if not diffs else "不一致 ❌"
        print(f"{env:14s} {status}")
        for d in diffs:
            print(f"      {d}")

    print()
    print("=" * 96)
    print("二、训练同源性（评估步数序列的重合长度）")
    print("=" * 96)
    for env, (ta, ma, tp, mp) in ARMS.items():
        print(f"\n-- {env}")
        print(f"   {'seed':>4}{'TEDI事件':>9}{'首次注入':>10}{'重合点数':>9}{'重合到步数':>12}  是否覆盖到首次注入前")
        for s in [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
            t_dir = CANON / env / ta / (tp % s)
            m_dir = CANON / env / ma / (mp % s)
            if not (t_dir / "events.jsonl").is_file() or not (m_dir / "events.jsonl").is_file():
                continue
            ts, te, _, _ = load(t_dir)
            ms, _, _, _ = load(m_dir)
            n = min(len(ts), len(ms))
            i = n
            for k in range(n):
                if ts[k] != ms[k]:
                    i = k
                    break
            first = te[0] if te else None
            cover = "—" if first is None else ("✅" if (i >= n or ts[i] > first) else "❌")
            print(f"   {s:>4}{len(te):>9}{(first if first else 0):>10}{i:>9}"
                  f"{(ts[i-1] if i else 0):>12}  {cover}")


if __name__ == "__main__":
    main()
