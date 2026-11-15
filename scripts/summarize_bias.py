# -*- coding: utf-8 -*-
"""Summarize bias."""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))


import argparse
import glob
import json
from pathlib import Path

import numpy as np

from scripts import admission_bias as ab


def run_one(d: Path, level: float, reps: int, seed: int):
    raw = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
    cfg = raw.get("configuration", {})
    env = raw.get("environment")
    files = sorted((d / "candidates").glob("*.npz"))
    rows = []
    for k, p in enumerate(files):
        z = np.load(p)
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        rms = float(np.sqrt(np.mean(z["reward"].astype(np.float64) ** 2)))
        c = level * rms
        n = len(z["reward"])
        rng = np.random.default_rng(seed + 1000 * k)
        ec, ss = [], []

        out0 = ab.replay(z, meta, cfg, env, np.zeros(n), np.random.default_rng(seed + 7 * k))
        ok = (int(out0["qualified"].sum()) == int(z["qualified"].sum())
              and len(out0["chosen"]) == int(z["selected"].sum()))
        for _ in range(reps):
            eps = rng.uniform(-c, c, size=n) if c > 0 else np.zeros(n)
            out = ab.replay(z, meta, cfg, env, eps, rng)
            sel = out["selected"]
            ss.append(int(sel.sum()))
            if sel.any() and c > 0:
                ec.append(float(eps[sel].mean() / c))
        rows.append(dict(event=meta["event_index"], step=meta["main_step"], ok=ok, n=n,
                         injected=int(z["selected"].sum()), sel_mean=float(np.mean(ss)),
                         ec=float(np.mean(ec)) if ec else 0.0))
    return env, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="支持通配符")
    ap.add_argument("--level", type=float, default=0.3)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump", action="store_true", help="按事件逐行输出 CSV，便于外部画图")
    args = ap.parse_args()

    dirs = []
    for pat in args.runs:
        dirs.extend(sorted(glob.glob(pat)))
    dirs = [Path(d) for d in dirs if (Path(d) / "candidates").is_dir()]
    if not dirs:
        print("没有找到带 candidates 的目录")
        return 1

    print(f"噪声水平 {args.level} × RMS   重复 {args.reps} 次   目录 {len(dirs)} 个\n")
    print(f"{'run':38s}{'事件':>5}{'自检':>6}{'零注入':>7}{'ε/c 中位':>10}{'ε/c 均值':>10}{'min':>8}{'max':>8}")
    all_ec = []
    collected = []
    for d in dirs:
        try:
            env, rows = run_one(d, args.level, args.reps, args.seed)
        except Exception as exc:
            print(f"{d.name:38s}  失败: {type(exc).__name__}: {exc}")
            continue
        collected.append((d.parent.name, rows))
        ec = np.asarray([r["ec"] for r in rows if r["injected"] > 0])
        ok = all(r["ok"] for r in rows)
        zero = sum(1 for r in rows if r["injected"] == 0)
        if len(ec):
            all_ec.extend(ec.tolist())
            print(f"{d.parent.name[-34:]:38s}{len(rows):>5}{'OK' if ok else 'FAIL':>6}{zero:>7}"
                  f"{np.median(ec)*100:>9.1f}%{ec.mean()*100:>9.1f}%"
                  f"{ec.min()*100:>7.1f}%{ec.max()*100:>7.1f}%")
        else:
            print(f"{d.parent.name[-34:]:38s}{len(rows):>5}{'OK' if ok else 'FAIL':>6}{zero:>7}"
                  f"{'-':>10}{'-':>10}{'-':>8}{'-':>8}")

    if args.dump:
        print()
        print("---- 逐事件 CSV ----")
        print("run,event,step,candidates,injected,eps_over_c_pct")
        for name, r2 in collected:
            for r in r2:
                val = "%.2f" % (r["ec"] * 100) if r["injected"] > 0 else ""
                print("%s,%d,%d,%d,%d,%s" % (name, r["event"], r["step"],
                                              r["n"], r["injected"], val))
        print("---- CSV 结束 ----")
    if all_ec:
        a = np.asarray(all_ec)
        print()
        print(f"合计 {len(a)} 个非零注入事件")
        print(f"  ε/c 中位 {np.median(a)*100:.1f}%   均值 {a.mean()*100:.1f}%   "
              f"IQR {np.percentile(a,25)*100:.1f}%–{np.percentile(a,75)*100:.1f}%   "
              f"min {a.min()*100:.1f}%   max {a.max()*100:.1f}%")
        print(f"  为正的事件 {int((a>0).sum())}/{len(a)}")
    print()
    print("注：ε/c 只统计原注入 > 0 的事件；零注入事件单独计数（它们在噪声下会变成注入）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
