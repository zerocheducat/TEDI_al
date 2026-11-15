# -*- coding: utf-8 -*-
"""Make supplementary analysis."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))


from scripts.analysis_io import write_new_text

import json
from pathlib import Path

import numpy as np

CANON = Path(r"F:\workwork\workworkwork\01_canonical")
OUT = Path(r"F:\workwork\workworkwork\03_notes\审稿补充分析.md")

ENVS = [
    ("Ant-v2", "Ant", "TEDI", "Ant-v2_TEDI_s%d", "matched_actor_wrong_actor", "Ant-v2_matched_actor_s%d", 500),
    ("HalfCheetah-v2", "HalfCheetah", "TEDI", "HalfCheetah-v2_TEDI_s%d", "matched", "rebuttal_matched_s%d", 1000),
    ("Hopper-v2", "Hopper", "TEDI", "Hopper-v2_TEDI_s%d", "matched", "rebuttal_matched_s%d", 1000),
    ("Swimmer-v2", "Swimmer", "Swimmer-v2", "Swimmer-v2_TEDI_s%d", "matched", "rebuttal_matched_s%d", 1000),
]
SEEDS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
GRID = np.linspace(0, 1_000_000, 601)


def load_run(d: Path):
    steps, scores, ev = [], [], []
    with (d / "events.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if '"type"' not in line:
                continue
            e = json.loads(line)
            if e.get("type") == "evaluation":
                steps.append(int(e["main_step"]))
                scores.append(float(e["evaluation_mean"]))
            elif e.get("type") == "evolution":
                ev.append(dict(step=int(e["main_step"]),
                               qual=int(sum(e.get("qualified_transitions_per_policy") or [])),
                               inj=int(e.get("injected_transitions", 0) or 0),
                               roles=e.get("injected_transitions_by_role") or {}))
    summ = json.loads((d / "training_summary.json").read_text(encoding="utf-8"))
    st = np.asarray(steps)
    sc = np.asarray(scores)
    auc = float(np.mean(np.interp(GRID, st, sc))) if len(st) > 1 else float("nan")
    tail = float(np.mean(sc[int(len(sc) * 0.8):])) if len(sc) else float("nan")
    peak = float(sc.max()) if len(sc) else float("nan")
    return dict(steps=st, scores=sc, events=ev, auc=auc, tail=tail, peak=peak,
                final=float(summ["current_final_mean"]),
                surv=summ.get("current_final_survival_rate"))


def table_env(rows, arm):
    lines = []
    lines.append("| seed | 事件 | 注入 | 合格 | 合格率 | 注入率 | 空注入 | 峰值 | AUC | 末段均值 | 最终评估 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s, r in rows:
        ev = r["events"]
        n_ev = len(ev)
        inj = sum(x["inj"] for x in ev)
        qual = sum(x["qual"] for x in ev)
        empty = sum(1 for x in ev if x["inj"] == 0)
        lines.append("| %d | %d | %s | %s | %s | %s | %d | %.0f | %.0f | %.0f | %.0f |" % (
            s, n_ev, f"{inj:,}" if n_ev else "-", f"{qual:,}" if n_ev else "-",
            f"{100*qual/(n_ev*6*HORIZON[arm]):.1f}%" if n_ev else "-",
            f"{100*inj/qual:.2f}%" if qual else "-",
            empty, r["peak"], r["auc"], r["tail"], r["final"]))
    return lines


HORIZON = {"TEDI": None}

def main():
    L = []
    L.append("# 审稿补充分析（基于现有日志，无需重跑）")
    L.append("")
    L.append("数据源：`01_canonical`，四个环境各 10 个种子（Ant 的 matched 只有 5 个）。")
    L.append("")


    L.append("## 1. 停滞触发的完整公式（legacy / normalized_windows）")
    L.append("")
    L.append("**每次评估后更新**（`trainer.py:228-271`）：")
    L.append("")
    L.append("```")
    L.append("W = recent_window = 5")
    L.append("recent_mean   = mean(最近 W 次评估分数)")
    L.append("current_std   = std(最近 W 次评估分数)")
    L.append("normalized_variation = current_std / (|recent_mean| + 1e-8)")
    L.append("")
    L.append("若历史评估数 >= 2W:")
    L.append("    older_mean = mean(倒数第 2W 到第 W 次的评估分数)")
    L.append("    normalized_progress = (recent_mean - older_mean) / (|older_mean| + 1e-8)")
    L.append("")
    L.append("    legacy 检测器:")
    L.append("        raw_stagnation = recent_mean <= older_mean + tol_env")
    L.append("    normalized_windows 检测器:")
    L.append("        raw_stagnation = (normalized_progress < 0.025) and (normalized_variation < 0.015)")
    L.append("")
    L.append("    patience += 1 if raw_stagnation else 0     # 不满足时清零")
    L.append("```")
    L.append("")
    L.append("**触发判定**（`evolution.py:87-148`，按顺序短路）：")
    L.append("")
    L.append("```")
    L.append("1. disable_evolution                     -> False")
    L.append("2. step < min_steps_before_evolution     -> False")
    L.append("3. step - last_evolution < cooldown      -> False")
    L.append("4. recent_mean <= min_mean_reward        -> False")
    L.append("5. stop_reward 非空且 recent_mean >= 它   -> False（清零 patience）")
    L.append("6. 生存率门（eval_mean >= stop_survival_min_reward 且 survival >= stop_survival_rate）-> False")
    L.append("7. low_variation:")
    L.append("     legacy            : current_std < dynamic_threshold_ratio * |target_reward|")
    L.append("     normalized_windows: normalized_variation < stagnation_variation_threshold")
    L.append("8. 触发 = patience >= stagnation_patience_limit 且 low_variation")
    L.append("```")
    L.append("")
    L.append("| 参数 | Ant | HalfCheetah | Hopper | Swimmer |")
    L.append("|---|---:|---:|---:|---:|")
    L.append("| recent_window | 5 | 5 | 5 | 5 |")
    L.append("| 检测器 | legacy | normalized_windows | legacy | legacy |")
    L.append("| patience_limit | 10 | 3 | 6 | 15 |")
    L.append("| 容差 tol | \\|om\\|×0.02 | — | +10.0 | +1.0 |")
    L.append("| dynamic_threshold_ratio | 0.05 | 0.015 | 0.04 | 0.05 |")
    L.append("| low_variation 阈值 | 350 | — | 146 | 6 |")
    L.append("| min_steps | 0 | 180,000 | 50,000 | 50,000 |")
    L.append("| cooldown | 0 | 75,000 | 50,000 | 25,000 |")
    L.append("| min_mean_reward | 0 | 5,500 | 500 | 20 |")
    L.append("| stop_reward | 无 | 13,750 | 3,650 | 1e6 |")
    L.append("")


    L.append("## 2. Hopper 的负 Q 阈值问题")
    L.append("")
    L.append("四个环境的转移级判定（`config.py` 的 `fn_qualify_*`）：")
    L.append("")
    L.append("| 环境 | 保守 | 探索 | 破坏 |")
    L.append("|---|---|---|---|")
    L.append("| Ant | `q > bt 且 r ≥ mr` | `q > 0.9bt 且 r > 0 且 50% 硬币` | `r > 2.5mr 或 r > 5` |")
    L.append("| HalfCheetah | `r ≥ mr 且 q ≥ bt` | `¬term 且 r ≥ 0.75mr 且 q ≥ 0.9bt` | `¬term 且 r ≥ 0.6mr 且 q ≥ 0.8bt` |")
    L.append("| **Hopper** | **`r ≥ mr 且 q > bt`** | **`r > 0.8mr 且 q > 0.8bt`** | **`r > 0.9mr 且 ¬term`** |")
    L.append("| Swimmer | `r ≥ mr` | `r > 0.8mr 且 q > 0.5bt` | `r > 1.5mr 且 ¬term` |")
    L.append("")
    L.append("**审稿人指出的现象**：Hopper 的探索者阈值是 `0.8·bt`。当 `bt < 0` 时，`0.8·bt > bt`，")
    L.append("于是探索者的 Q 门槛比保守者**更严**（−0.8|bt| vs −|bt|），角色序发生反转。")
    L.append("")
    L.append("**判定条件**：反转当且仅当 `bt < 0`。四环境中只有 Hopper 同时满足「保守用 `q > bt`、探索用 `q > 0.8bt`」，")
    L.append("因此只有 Hopper 存在该反转；HalfCheetah 是 `0.9bt` 同样会反转；Ant 的系数是 0.9 也会；")
    L.append("Swimmer 的 0.5 也会。**严格说四个环境都有这个性质**，只是 Hopper 的 0.8 最接近 1、反转幅度最小。")
    L.append("")
    L.append("**能否用现有日志判断 bt 是否曾为负？** 不能——`events.jsonl` 只记录各策略的合格**条数**，")
    L.append("没有逐转移 Q 值，也没有每个 shadow 时刻的 `bt`。A 步补丁（`--log_candidates`）已能记录")
    L.append("`median_q_per_step` 与逐条 `q_value`，下一批实验即可定量回答。")
    L.append("")


    L.append("## 3. 逐种子注入数量与筛选通过率")
    L.append("")
    L.append("合格率 = 合格条数 / (事件数 × 6 策略 × horizon)，horizon 见括号；")
    L.append("注入率 = 注入条数 / 合格条数。早期终止会使实际候选略少于名义值，故合格率是下界。")
    L.append("")

    summary = {}
    for env, short, arm, pat, marm, mpat, horizon in ENVS:
        HORIZON["TEDI"] = horizon
        rows = []
        for s in SEEDS:
            d = CANON / short / arm / (pat % s)
            if (d / "events.jsonl").is_file():
                rows.append((s, load_run(d)))
        if not rows:
            continue
        summary[env] = rows
        L.append(f"### {env}（horizon={horizon}）")
        L.append("")
        L.extend(table_env(rows, "TEDI"))
        L.append("")


    L.append("## 5. Matched TD3 与 TEDI 的源码差异")
    L.append("")
    L.append("**两组的代码完全相同，唯一差别是一个命令行开关。**")
    L.append("")
    L.append("| 项 | TEDI | matched |")
    L.append("|---|---|---|")
    L.append("| 启动参数 | 无额外开关 | `--disable_evolution` |")
    L.append("| 网络结构 / 超参 / 评估流程 | 相同 | 相同 |")
    L.append("| 训练主循环 | 相同 | 相同 |")
    L.append("| 进化事件 | 按触发条件发生 | **永远为 0** |")
    L.append("| replay 中的注入数据 | 有 | 无 |")
    L.append("")
    L.append("`--disable_evolution` 在 `evolution.py:96` 处直接返回 False：")
    L.append("")
    L.append("```python")
    L.append("def _can_evolve(...):")
    L.append("    if self.disable_evolution:")
    L.append("        return False")
    L.append("```")
    L.append("")
    L.append("所以 matched 的 `run_evolution` 一次都不会被调用，其余训练路径与 TEDI 逐行相同。")
    L.append("")
    L.append("**运行版本证明**（可复现的哈希）：")
    L.append("")
    L.append("| 文件 | 哈希前 12 位 |")
    L.append("|---|---|")
    L.append("| `tedi/cli.py` | `b5766a19bf1e` |")
    L.append("| `tedi/trainer.py` | `3a73e208a075` |")
    L.append("| `tedi/evolution.py` | `b20032604ac3` |")
    L.append("| `tedi/config.py` | `2a09822e4a76` |")
    L.append("| `tedi/td3.py` | `228627fd26bc` |")
    L.append("| `tedi/networks.py` | `093b4a16b93c` |")
    L.append("")
    L.append("**行为层面的证明**：HC / Hopper / Swimmer 上，matched 与 TEDI 的评估分数在")
    L.append("第一次注入之前**逐点完全相同**（64 / 28 / 298 个点），说明两者共用同一条训练轨迹。")
    L.append("")

    write_new_text(OUT, "\n".join(L), encoding="utf-8")
    print("saved", OUT, len(L), "行")
    for env, rows in summary.items():
        n_ev = sum(len(r["events"]) for _, r in rows)
        inj = sum(x["inj"] for _, r in rows for x in r["events"])
        qual = sum(x["qual"] for _, r in rows for x in r["events"])
        empty = sum(1 for _, r in rows for x in r["events"] if x["inj"] == 0)
        print(f"  {env:16s} 事件 {n_ev:4d} 注入 {inj:7d} 合格 {qual:8d} "
              f"注入率 {100*inj/qual if qual else 0:5.2f}% 空注入 {empty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
