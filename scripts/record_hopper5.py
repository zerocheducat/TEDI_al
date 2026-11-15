# -*- coding: utf-8 -*-
"""Record hopper5."""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))


from scripts.analysis_io import write_new_text

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.record_video import grab_frame, label, load_policy, make_env, pad_frames, rollout, write_video

SEEDS = (35, 10, 45, 20, 15)
ENV_DIR = "Hopper"


def find_run(root: Path, arm: str, seed: int):
    arm_dir = root / ENV_DIR / arm
    if not arm_dir.is_dir():
        return None
    for p in sorted(arm_dir.iterdir()):
        if p.is_dir() and p.name.endswith(f"s{seed}") and (p / "training_summary.json").is_file():
            return p
    return None


def count_events(run_dir: Path) -> int:
    n = 0
    with (run_dir / "events.jsonl").open(encoding="utf-8") as h:
        for line in h:
            if '"evolution"' in line:
                try:
                    if json.loads(line).get("type") == "evolution":
                        n += 1
                except json.JSONDecodeError:
                    pass
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"F:\workwork\workworkwork\01_canonical")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--which", choices=["actor", "ema"], default="ema")
    ap.add_argument("--all", action="store_true", help="零事件种子也录")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else root.parent / "03_notes" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    for s in SEEDS:
        t, m = find_run(root, "TEDI", s), find_run(root, "matched", s)
        if t is None or m is None:
            print(f"[跳过] seed {s}: TEDI={'有' if t else '缺'} matched={'有' if m else '缺'}")
            continue
        ev = count_events(t)
        pairs.append((s, t, m, ev))

    if not args.all:
        chosen = [p for p in pairs if p[3] > 0]
        skipped = [p[0] for p in pairs if p[3] == 0]
        if skipped:
            print(f"[提示] 种子 {skipped} 进化 0 次，TEDI 与 matched 是同一段训练，"
                  f"录像会完全一样，已跳过（要强制录制加 --all）")
    else:
        chosen = pairs

    if not chosen:
        print("没有可录的种子")
        return 1

    summary = []
    for s, t, m, ev in chosen:
        print(f"\n== seed {s}（TEDI 进化 {ev} 次）")
        ck, net_t, env_name = load_policy(t, "latest_checkpoint.pth", args.which)
        _, net_m, env_name2 = load_policy(m, "latest_checkpoint.pth", args.which)
        if env_name != env_name2:
            print(f"   环境不一致: {env_name} vs {env_name2}")
            continue

        env = make_env(env_name, render=True)
        max_action = float(np.asarray(env.action_space.high).reshape(-1)[0])
        base = int(ck.get("seed") or 0)
        frames_all, recs = [], []
        for k in range(args.episodes):
            seed = base + 500_000 + k
            ra = rollout(net_t, env, seed, max_action, render=True)
            rb = rollout(net_m, env, seed, max_action, render=True)
            print(f"   回合 {k+1} seed={seed}: "
                  f"TEDI 回报 {ra['ret']:8.1f} 长度 {ra['length']:4d} | "
                  f"matched 回报 {rb['ret']:8.1f} 长度 {rb['length']:4d}")
            recs.append(dict(seed=seed,
                             tedi=dict(ret=ra["ret"], length=ra["length"], truncated=ra["truncated"]),
                             matched=dict(ret=rb["ret"], length=rb["length"], truncated=rb["truncated"])))
            if not ra["frames"] and not rb["frames"]:
                continue
            n = max(len(ra["frames"]), len(rb["frames"]))
            fa = pad_frames(ra["frames"], n)
            fb = pad_frames(rb["frames"], n)
            for x, y in zip(fa, fb):
                frames_all.append(np.concatenate(
                    [label(x, f"TEDI       s{s} ep{k+1}"),
                     label(y, f"matched    s{s} ep{k+1}")], axis=1))
        env.close()

        if not frames_all:
            print("   [失败] 一帧都没抓到。先跑 `python render_test.py` 看看渲染能不能用。")
            continue
        png = out_dir / f"hopper_s{s}.mp4"
        ok = write_video(png, frames_all)
        print(f"   {'已保存' if ok else '保存失败'}: {png}  帧数 {len(frames_all)}")
        summary.append(dict(seed=s, events=ev, video=str(png), episodes=recs))

    meta = out_dir / "hopper5_video_summary.json"
    write_new_text(meta, json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总已写出 {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
