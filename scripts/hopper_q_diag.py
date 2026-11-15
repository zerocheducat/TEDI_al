# -*- coding: utf-8 -*-
"""Hopper q diag."""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))


from scripts.analysis_io import write_new_text

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROLES = ("conservative", "explorer", "destroyer")


def load_events(run_dir: Path):
    out = []
    for p in sorted((run_dir / "candidates").glob("*.npz")):
        z = np.load(p)
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        if meta.get("transition_value_mode") != "current_q":
            pass
        out.append((z, meta))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dirs = []
    for pat in args.runs:
        dirs.extend(sorted(glob.glob(pat)))
    dirs = [Path(d) for d in dirs if (Path(d) / "candidates").is_dir()]
    if not dirs:
        print("没有找到带 candidates 的目录")
        return 1

    rows = []
    print(f"{'run':28s}{'批次':>5}{'批次m_q':>9}{'符号':>4}{'时刻数':>6}{'负时刻':>6}"
          f"{'最小':>8}{'候选':>7}{'合格':>7}{'种群门':>7}{'父门':>6}{'入池':>5}")
    for d in dirs:
        for z, meta in load_events(d):
            q = z["q_value"].astype(float)
            mq = float(np.median(q))
            steps_per = meta.get("median_q_per_step") or []
            arr_step = np.asarray(steps_per, dtype=float) if steps_per else np.asarray([mq])
            n_neg_step = int((arr_step < 0).sum())
            min_step = float(arr_step.min()) if arr_step.size else float("nan")
            role = z["role_id"].astype(int)
            pol_pass = z["population_gate_pass"].astype(bool) if "population_gate_pass" in z.files \
                else z["gated"].astype(bool)
            par_pass = z["parent_gate_pass"].astype(bool) if "parent_gate_pass" in z.files \
                else z["gated"].astype(bool)

            qpass = {}
            for i, r in enumerate(ROLES):
                m = role == i
                if not m.any():
                    qpass[r] = float("nan")
                    continue
                if r == "conservative":
                    qpass[r] = float((q[m] > mq).mean())
                elif r == "explorer":
                    qpass[r] = float((q[m] > 0.8 * mq).mean())
                else:
                    qpass[r] = float("nan")
            rows.append(dict(run=d.parent.name, batch=meta["event_index"], mq=mq,
                             n=len(q), qual=int(z["qualified"].sum()),
                             pop=int(pol_pass.sum()), par=int(par_pass.sum()),
                             adm=int(z["selected"].sum()), qpass=qpass,
                             steps=arr_step, n_neg_step=n_neg_step,
                             injected=int(meta.get("injected", 0))))
            print(f"{d.parent.name[-27:]:28s}{meta['event_index']:>5}{mq:>9.1f}"
                  f"{'-' if mq < 0 else ('0' if mq == 0 else '+'):>4}{arr_step.size:>6}"
                  f"{n_neg_step:>6}{min_step:>8.1f}{len(q):>7}"
                  f"{int(z['qualified'].sum()):>7}{int(pol_pass.sum()):>7}{int(par_pass.sum()):>6}"
                  f"{int(z['selected'].sum()):>5}")

    all_step = np.concatenate([r["steps"] for r in rows]) if rows else np.asarray([])
    mqs = np.array([r["mq"] for r in rows])
    neg = int((mqs < 0).sum())
    zero = int((mqs == 0).sum())
    print()
    print(f"批次总数 {len(rows)}   m_q<0: {neg}   m_q==0: {zero}   m_q>0: {len(rows)-neg-zero}")
    print(f"m_q 范围 {mqs.min():.1f} ~ {mqs.max():.1f}   中位 {np.median(mqs):.1f}")
    if all_step.size:
        print(f"逐时刻（每批 {int(np.median([r['steps'].size for r in rows]))} 个时刻）："
              f"总时刻 {all_step.size}   负值时刻 {int((all_step < 0).sum())}   "
              f"最小时刻中位 {all_step.min():.1f}   中位 {np.median(all_step):.1f}")
    print()
    if neg == 0:
        print("结论：本诊断范围内未观察到负的中位数 Q。")
        print("      因此 Hopper 的「0.8·m_q 反而更严」阈值反转在实测中未出现；")
        print("      该反转是规则的数学性质，只在 m_q<0 时生效。")
    else:
        print("结论：出现了负中位数，阈值反转在以下批次实际发生，需单列分析：")
        for r in rows:
            if r["mq"] < 0:
                print(f"      {r['run']} 批次 {r['batch']}  m_q={r['mq']:.1f}")

    print()
    print("各角色 Q 条件通过率（按 m_q 符号分组）：")
    for sign, sel in (("m_q<0", [r for r in rows if r["mq"] < 0]),
                      ("m_q>0", [r for r in rows if r["mq"] > 0])):
        if not sel:
            continue
        parts = []
        for role in ("conservative", "explorer"):
            vals = [r["qpass"][role] for r in sel if r["qpass"][role] == r["qpass"][role]]
            parts.append(f"{role}={np.mean(vals)*100:.1f}%" if vals else f"{role}=NA")
        print(f"  {sign:6s} 批次 {len(sel):3d}   " + "   ".join(parts))

    if args.out:
        write_new_text(Path(args.out), json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
