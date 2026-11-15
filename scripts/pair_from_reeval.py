# -*- coding: utf-8 -*-
"""Pair from reeval."""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))


import argparse
import json
import re
from pathlib import Path

import numpy as np


def seed_of(name: str):
    m = re.search(r"s(\d+)$", name)
    return int(m.group(1)) if m else None


def collect(root: Path, which: str):
    out, bad = {}, 0
    for p in sorted(root.glob("*/reeval_actor.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:
            bad += 1
            continue
        meta = d.get("_meta") or {}
        name = meta.get("run") or p.parent.name
        s = seed_of(name)
        if s is None or which not in d:
            bad += 1
            continue
        if s in out:
            raise ValueError(f"Multiple run identities for seed {s}; provide an unambiguous root")
        out[s] = (name, float(d[which]["mean"]), float(d[which].get("std") or 0.0),
                  d[which].get("returns") or [])
    if bad:
        print(f"   [提示] {root.name} 下有 {bad} 个文件读不出分数，已跳过")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="A 组父目录（如 TEDI）")
    ap.add_argument("--b", required=True, help="B 组父目录（如 matched）")
    ap.add_argument("--which", default="ema_actor", choices=["actor", "ema_actor"])
    ap.add_argument("--label-a", default="TEDI")
    ap.add_argument("--label-b", default="matched")
    args = ap.parse_args()

    a = collect(Path(args.a).expanduser().resolve(), args.which)
    b = collect(Path(args.b).expanduser().resolve(), args.which)
    if set(a) != set(b):
        raise ValueError("Seed sets differ; refusing to silently discard unmatched runs")
    seeds = sorted(a)
    if not seeds:
        print("没有可配对的种子。检查两边的 reeval_actor.json 是否都存在。")
        return 1

    print(f"评估网络 {args.which}   可配对 {len(seeds)} 个种子: {seeds}")
    print()
    print(f"{'seed':>5}{args.label_a:>12}{args.label_b:>12}{'Δ':>10}")
    diffs = []
    for s in seeds:
        x = a[s][1]
        y = b[s][1]
        d = x - y
        diffs.append(d)
        print(f"{s:>5}{x:>12.1f}{y:>12.1f}{d:>+10.1f}")

    d = np.array(diffs)
    print()
    print(f"平均 {args.label_a} {np.mean([a[s][1] for s in seeds]):.1f}   "
          f"{args.label_b} {np.mean([b[s][1] for s in seeds]):.1f}   "
          f"Δ {d.mean():+.1f} ± {d.std(ddof=1) if len(d) > 1 else 0:.1f}")
    print(f"{args.label_a} 胜 {int((d > 0).sum())}/{len(d)}")
    if len(d) > 1:
        from scipy import stats
        t, p = stats.ttest_rel([a[s][1] for s in seeds], [b[s][1] for s in seeds])
        print(f"配对 t 检验 t={t:.2f}  p={p:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
