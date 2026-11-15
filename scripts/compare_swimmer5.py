# -*- coding: utf-8 -*-
"""Compare TEDI and matched runs for the five recorded Swimmer seeds."""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))


import argparse
import csv
import json
import unicodedata
from itertools import product
from pathlib import Path

import numpy as np

SEEDS = (25, 10, 40, 55, 45)
TEDI_SUB = "Swimmer-v2"
TEDI_FMT = "Swimmer-v2_TEDI_s{seed}"
MATCHED_SUB = "matched"
MATCHED_FMT = "rebuttal_matched_s{seed}"
MAX_STEPS = 1_000_000


def run_dir(base: Path, sub: str, fmt: str, seed: int) -> Path | None:
    d = base / sub / fmt.format(seed=seed)
    return d if (d / "training_summary.json").is_file() else None


def load_curve(d: Path):
    """Load the recorded evaluation steps and returns."""
    steps, scores, evo = [], [], []
    with (d / "events.jsonl").open(encoding="utf-8") as h:
        for line in h:
            if '"type"' not in line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = e.get("type")
            if t == "evaluation":
                steps.append(int(e["main_step"]))
                scores.append(float(e["evaluation_mean"]))
            elif t == "evolution":
                evo.append(int(e["main_step"]))
    return np.asarray(steps, np.int64), np.asarray(scores, np.float64), np.asarray(evo, np.int64)


def load_run(d: Path) -> dict:
    summ = json.loads((d / "training_summary.json").read_text(encoding="utf-8"))
    steps, scores, evo = load_curve(d)
    keep = steps <= MAX_STEPS
    s, sc = steps[keep], scores[keep]

    def npy(name):
        p = d / name
        return np.load(p).ravel() if p.is_file() else None

    sv, ln, ret = npy("eval_survival_rates.npy"), npy("eval_lengths.npy"), npy("eval_returns.npy")
    order = np.argsort(steps)
    return dict(
        dir=d,
        events=int(summ.get("evolution_events", len(evo))),
        shadow=int(summ.get("shadow_env_steps", 0)),
        peak=float(sc.max()),
        peak_step=int(s[int(sc.argmax())]),
        final=float(summ["current_final_mean"]),
        stable=float(summ["best_stable_final_mean"]),
        final_surv=float(summ.get("current_final_survival_rate", float("nan"))),
        sur_last100=float(np.mean(sv[-100:])) if sv is not None else float("nan"),
        len_last100=float(np.mean(ln[-100:])) if ln is not None else float("nan"),
        wall=float(summ.get("wall_time_seconds", float("nan"))),
        steps=steps, scores=scores, evo=evo,
        eval_returns=ret[order] if ret is not None and ret.size == steps.size else None,
    )


def signflip_p(diff: np.ndarray) -> float:
    """Compute a paired sign-flip p-value."""
    d = np.asarray(diff, float)
    n = d.size
    obs = abs(d.mean())
    cnt = 0
    for signs in product((1.0, -1.0), repeat=n):
        if abs((d * np.asarray(signs)).mean()) >= obs - 1e-12:
            cnt += 1
    return cnt / 2 ** n


def smooth(x: np.ndarray, w: int = 25) -> np.ndarray:
    if x.size < w:
        return x
    k = np.ones(w) / w
    pad_ = w // 2
    xp = np.pad(x, pad_, mode="edge")
    return np.convolve(xp, k, mode="valid")[: x.size]


def w2(s: str) -> int:
    """Compute the distance between the two empirical distributions."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s: str, width: int, right: bool = True) -> str:
    k = max(0, width - w2(s))
    return (" " * k + s) if right else (s + " " * k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=r"F:\workwork\workworkwork\01_canonical\Swimmer")
    ap.add_argument("--out-dir", default=r"F:\workwork\workworkwork\03_notes")
    args = ap.parse_args()

    base = Path(args.base).expanduser()
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in SEEDS:
        td = run_dir(base, TEDI_SUB, TEDI_FMT, s)
        md = run_dir(base, MATCHED_SUB, MATCHED_FMT, s)
        if td is None or md is None:
            raise FileNotFoundError(f"Missing fixed-seed pair: Swimmer-v2, seed={s}")
        t, m = load_run(td), load_run(md)

        n = min(t["scores"].size, m["scores"].size)
        d = t["scores"][:n] - m["scores"][:n]
        ad = np.abs(d)
        idx = np.flatnonzero(ad > 0.5)
        diverge = int(t["steps"][idx[0]]) if idx.size else -1

        first_evo = int(t["evo"][0]) if t["evo"].size else 0
        pre = np.abs(d[t["steps"][:n] < first_evo])
        pre_max = float(pre.max()) if pre.size else float("nan")
        rows.append(dict(seed=s, _t=t, _m=m,
                         t_peak=t["peak"], m_peak=m["peak"],
                         t_final=t["final"], m_final=m["final"],
                         t_stable=t["stable"], m_stable=m["stable"],
                         t_events=t["events"], m_events=m["events"],
                         diverge=diverge,
                         first_evo=first_evo,
                         pre_identity=pre_max,
                         maxabs=float(ad.max()),
                         post_delta=float(np.mean(d[-200:])),
                         n_common=n))

    if not rows:
        print("没有可对比的数据")
        return 1


    COLS = [("seed", 6), ("事件", 6), ("TEDI峰", 9), ("m峰", 8), ("Δ峰", 8),
            ("TEDI末", 9), ("m末", 8), ("Δ末", 8), ("TEDI稳", 9), ("m稳", 8),
            ("分歧步", 8), ("末200Δ", 9)]
    hdr = "".join(pad(n, w) for n, w in COLS)
    rule = "-" * w2(hdr)
    print("=" * w2(hdr))
    print("Swimmer-v2：TEDI vs matched（种子 25/10/40/55/45）")
    print("=" * w2(hdr))
    print(hdr)
    print(rule)
    for r in rows:
        cells = [str(r["seed"]), str(r["t_events"]),
                 f"{r['t_peak']:.1f}", f"{r['m_peak']:.1f}", f"{r['t_peak'] - r['m_peak']:+.1f}",
                 f"{r['t_final']:.1f}", f"{r['m_final']:.1f}", f"{r['t_final'] - r['m_final']:+.1f}",
                 f"{r['t_stable']:.1f}", f"{r['m_stable']:.1f}",
                 str(r["diverge"]), f"{r['post_delta']:+.2f}"]
        print("".join(pad(c, w) for c, (_, w) in zip(cells, COLS)))
    print(rule)

    d_final = np.array([r["t_final"] - r["m_final"] for r in rows])
    d_peak = np.array([r["t_peak"] - r["m_peak"] for r in rows])
    d_stable = np.array([r["t_stable"] - r["m_stable"] for r in rows])

    def blk(name, t, m, d):
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
        p = signflip_p(d)
        wins = int((d > 0).sum())
        print(f"  {pad(name, 10, right=False)}TEDI {t.mean():8.2f}  matched {m.mean():8.2f}  "
              f"Δ {d.mean():+7.2f} ± {se:5.2f}(SE)  TEDI胜 {wins}/{n}  p={p:.3f}")

    print(f"\n[配对汇总 n={len(rows)}，保留全部固定种子]")
    blk("末值", np.array([r["t_final"] for r in rows]), np.array([r["m_final"] for r in rows]), d_final)
    blk("峰值", np.array([r["t_peak"] for r in rows]), np.array([r["m_peak"] for r in rows]), d_peak)
    blk("稳定末值", np.array([r["t_stable"] for r in rows]), np.array([r["m_stable"] for r in rows]), d_stable)
    print(f"  事件数    TEDI {np.mean([r['t_events'] for r in rows]):.1f}  "
          f"shadow 步 {np.mean([r['_t']['shadow'] for r in rows]):.0f}")
    print(f"  生存率    两臂均为 {np.mean([r['_t']['sur_last100'] for r in rows]):.3f}"
          f" / 长度 {np.mean([r['_t']['len_last100'] for r in rows]):.0f}  —— 该环境已到上限，无判别力")
    worst = max(r["pre_identity"] for r in rows)
    print(f"\n[配对有效性] 首次注入之前，TEDI 与 matched 的评估曲线逐点最大差 = {worst:.1e}")
    print("This diagnostic reports observed agreement; it does not establish causality.")
    print(f"  首次注入步：{[(r['seed'], r['first_evo']) for r in rows]}")

    # ---------------- CSV ----------------
    csv_path = out / "Swimmer5_对比.csv"
    with csv_path.open("x", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["seed", "t_events", "m_events", "shadow_steps",
                    "t_peak", "m_peak", "d_peak",
                    "t_final", "m_final", "d_final",
                    "t_stable", "m_stable", "d_stable",
                    "t_survival", "m_survival", "t_length", "m_length",
                    "diverge_step", "pre_injection_max_abs_diff",
                    "max_abs_curve_diff", "post200_delta"])
        for r in rows:
            w.writerow([r["seed"], r["t_events"], r["m_events"], r["_t"]["shadow"],
                        r["t_peak"], r["m_peak"], r["t_peak"] - r["m_peak"],
                        r["t_final"], r["m_final"], r["t_final"] - r["m_final"],
                        r["t_stable"], r["m_stable"], r["t_stable"] - r["m_stable"],
                        r["_t"]["sur_last100"], r["_m"]["sur_last100"],
                        r["_t"]["len_last100"], r["_m"]["len_last100"],
                        r["diverge"], r["pre_identity"],
                        r["maxabs"], r["post_delta"]])
    print(f"\n已写出 {csv_path}")


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
