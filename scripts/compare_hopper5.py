# -*- coding: utf-8 -*-
"""Compare TEDI and matched runs for the five recorded Hopper seeds."""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))


import argparse
import csv
import json
from pathlib import Path

import numpy as np

SEEDS = (35, 10, 45, 20, 15)
ENV_DIR = "Hopper"
TEDI_ARM = "TEDI"
MATCHED_ARM = "matched"
MAX_STEPS = 1_000_000


def find_run(root: Path, arm: str, seed: int) -> Path | None:
    """Locate the requested run by its seed."""
    arm_dir = root / ENV_DIR / arm
    if not arm_dir.is_dir():
        return None
    hits = [p for p in arm_dir.iterdir() if p.is_dir() and p.name.endswith(f"s{seed}")]
    for p in hits:
        if (p / "training_summary.json").is_file():
            return p
    return None


def load_run(d: Path) -> dict:
    summ = json.loads((d / "training_summary.json").read_text(encoding="utf-8"))
    steps, scores, events = [], [], 0
    with (d / "events.jsonl").open(encoding="utf-8") as h:
        for line in h:
            if '"type"' not in line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") == "evaluation":
                steps.append(int(e["main_step"]))
                scores.append(float(e["evaluation_mean"]))
            elif e.get("type") == "evolution":
                events += 1
    steps = np.asarray(steps, np.int64)
    scores = np.asarray(scores, np.float64)
    keep = steps <= MAX_STEPS

    def npy(name):
        p = d / name
        return np.load(p).ravel() if p.is_file() else None

    sv = npy("eval_survival_rates.npy")
    ln = npy("eval_lengths.npy")
    return dict(
        peak=float(scores[keep].max()) if keep.any() else float("nan"),
        final=float(summ["current_final_mean"]),
        survival=float(np.mean(sv[-100:])) if sv is not None else float("nan"),
        length=float(np.mean(ln[-100:])) if ln is not None else float("nan"),
        events=events,
        steps=steps, scores=scores,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"F:\workwork\workworkwork\01_canonical",
                    help="canonical 根目录（其下应有 Hopper/TEDI 与 Hopper/matched）")
    ap.add_argument("--out-dir", default=None, help="输出目录，默认当前目录")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (root / ENV_DIR).is_dir():
        print(f"找不到 {root / ENV_DIR}")
        print("请用 --root 指向包含 Hopper/TEDI 和 Hopper/matched 的目录")
        return 2

    rows = []
    for s in SEEDS:
        ta, ma = find_run(root, TEDI_ARM, s), find_run(root, MATCHED_ARM, s)
        if ta is None or ma is None:
            raise FileNotFoundError(f"Missing fixed-seed pair: Hopper-v2, seed={s}")
        t, m = load_run(ta), load_run(ma)
        rows.append(dict(seed=s, events=t["events"],
                         t_peak=t["peak"], m_peak=m["peak"],
                         t_final=t["final"], m_final=m["final"],
                         t_surv=t["survival"], m_surv=m["survival"],
                         t_len=t["length"], m_len=m["length"], _t=t, _m=m))

    if not rows:
        print("没有可对比的数据")
        return 1

    hdr = (f"{'seed':>5}{'事件':>5}{'TEDI峰值':>10}{'m峰值':>9}{'Δ峰':>8}"
           f"{'TEDI最终':>10}{'m最终':>9}{'Δ末':>8}{'TEDI存活':>9}{'m存活':>8}{'TEDI长度':>9}{'m长度':>8}")
    print("=" * len(hdr))
    print("Hopper 五个种子：TEDI vs matched")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        flag = "  ← 零事件" if r["events"] == 0 else ""
        print(f"{r['seed']:>5}{r['events']:>5}{r['t_peak']:>10.1f}{r['m_peak']:>9.1f}"
              f"{r['t_peak']-r['m_peak']:>+8.1f}{r['t_final']:>10.1f}{r['m_final']:>9.1f}"
              f"{r['t_final']-r['m_final']:>+8.1f}{r['t_surv']:>9.3f}{r['m_surv']:>8.3f}"
              f"{r['t_len']:>9.0f}{r['m_len']:>8.0f}{flag}")

    def agg(sel, key):
        xs = [r[key] for r in sel]
        return float(np.mean(xs)) if xs else float("nan")


    print("-" * len(hdr))
    print(f"\n[全部 {len(rows)} 个种子]")
    print(f"  峰值  TEDI {agg(rows,'t_peak'):8.1f}  matched {agg(rows,'m_peak'):8.1f}  Δ {agg(rows,'t_peak')-agg(rows,'m_peak'):+.1f}")
    print(f"  最终  TEDI {agg(rows,'t_final'):8.1f}  matched {agg(rows,'m_final'):8.1f}  Δ {agg(rows,'t_final')-agg(rows,'m_final'):+.1f}")
    print(f"  存活  TEDI {agg(rows,'t_surv'):8.3f}  matched {agg(rows,'m_surv'):8.3f}  Δ {agg(rows,'t_surv')-agg(rows,'m_surv'):+.3f}")

    print("All fixed seeds, including zero-event runs, are retained in the summary.")

    csv_path = out_dir / "hopper5_对比.csv"
    with csv_path.open("x", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["seed", "events", "t_peak", "m_peak", "d_peak",
                    "t_final", "m_final", "d_final",
                    "t_survival", "m_survival", "t_length", "m_length"])
        for r in rows:
            w.writerow([r["seed"], r["events"],
                        r["t_peak"], r["m_peak"], r["t_peak"] - r["m_peak"],
                        r["t_final"], r["m_final"], r["t_final"] - r["m_final"],
                        r["t_surv"], r["m_surv"],
                        r["t_len"], r["m_len"]])
    print(f"\n已写出 {csv_path}")


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
