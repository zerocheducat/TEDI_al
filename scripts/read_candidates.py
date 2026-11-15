# -*- coding: utf-8 -*-
"""Read candidates."""


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    root = Path(args.run).expanduser().resolve() / "candidates"
    files = sorted(root.glob("*.npz"))
    if not files:
        print(f"{root} 下没有 npz")
        return 1

    print(f"{'event':>6}{'step':>9}{'候选':>7}{'合格':>6}{'可注入':>7}{'选中':>6}"
          f"{'C/E/D 选中':>14}{'选中Q均值':>11}{'落选Q均值':>11}")
    tot = dict(cand=0, qual=0, inj=0, sel=0)
    for p in files:
        z = np.load(p)
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        sel = z["selected"]
        role = z["role_id"]
        names = ("保守", "探索", "破坏")
        by_role = "/".join(str(int(((sel) & (role == i)).sum())) for i in range(3))
        q_sel = float(z["q_value"][sel].mean()) if sel.any() else float("nan")
        q_rej = float(z["q_value"][~sel].mean()) if (~sel).any() else float("nan")
        print(f"{meta['event_index']:>6}{meta['main_step']:>9}{len(sel):>7}"
              f"{int(z['qualified'].sum()):>6}{int(z['injectable'].sum()):>7}{int(sel.sum()):>6}"
              f"{by_role:>14}{q_sel:>11.1f}{q_rej:>11.1f}")
        tot["cand"] += len(sel)
        tot["qual"] += int(z["qualified"].sum())
        tot["inj"] += int(z["injectable"].sum())
        tot["sel"] += int(sel.sum())

    print()
    print(f"合计：候选 {tot['cand']}，合格 {tot['qual']}，可注入 {tot['inj']}，最终选中 {tot['sel']}")
    if tot["qual"]:
        print(f"      合格率 {100*tot['qual']/tot['cand']:.1f}%   "
              f"选中/合格 {100*tot['sel']/max(tot['qual'],1):.2f}%")
    print()
    print("字段：state / action / reward / next_state / terminal / q_value /")
    print("      qualified / gated / injectable / selected / policy_idx / role_id")
    print("      （role_id: 0=conservative 1=explorer 2=destroyer）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
