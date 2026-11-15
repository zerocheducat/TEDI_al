# -*- coding: utf-8 -*-
"""Record video."""


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
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _actor_class():
    """Load the actor implementation compatible with the checkpoint."""
    try:
        import inspect
        from tedi.networks import RobustActor
        if "use_layernorm" in inspect.signature(RobustActor.__init__).parameters:
            return RobustActor
    except Exception:
        pass

    import torch.nn as nn
    import torch.nn.functional as F

    class _InlineActor(nn.Module):
        def __init__(self, state_dim, action_dim, max_action, use_layernorm=True):
            super(_InlineActor, self).__init__()
            self.use_layernorm = bool(use_layernorm)
            self.l1 = nn.Linear(state_dim, 256)
            self.l2 = nn.Linear(256, 256)
            self.l3 = nn.Linear(256, action_dim)
            self.ln1 = nn.LayerNorm(256) if self.use_layernorm else nn.Identity()
            self.ln2 = nn.LayerNorm(256) if self.use_layernorm else nn.Identity()
            self.max_action = float(max_action)

        def forward(self, state):
            a = F.relu(self.ln1(self.l1(state)))
            a = F.relu(self.ln2(self.l2(a)))
            return self.max_action * torch.tanh(self.l3(a))

    return _InlineActor


def load_policy(run_dir: Path, weight: str, which: str):
    ck_path = run_dir / weight
    if not ck_path.is_file():
        ck_path = run_dir / "latest_checkpoint.pth"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    td3 = ck.get("td3", ck)
    env_name = str(ck.get("env_name") or (ck.get("config") or {}).get("environment"))
    key = "actor" if which == "actor" else "ema_actor"
    sd = td3[key]

    net = _actor_class()(
        int(sd["l1.weight"].shape[1]),
        int(sd["l3.weight"].shape[0]),
        1.0,
        use_layernorm=("ln1.weight" in sd),
    )
    net.load_state_dict(sd)
    net.eval()
    return ck, net, env_name


def make_env(env_name: str, render: bool):
    import gymnasium as gym
    if render:
        try:
            return gym.make(env_name, render_mode="rgb_array")
        except TypeError:
            pass
    return gym.make(env_name)


def grab_frame(env):
    try:
        return env.render()
    except Exception:
        try:
            return env.render(mode="rgb_array")
        except Exception:
            return None


def rollout(net, env, seed, max_action, render: bool):
    obs, _ = env.reset(seed=int(seed))
    frames, obss, ret, n = [], [], 0.0, 0
    while True:
        if render:
            f = grab_frame(env)
            if f is not None:
                frames.append(np.asarray(f, dtype=np.uint8))
        obss.append(np.asarray(obs, dtype=np.float32).copy())
        with torch.inference_mode():
            a = net(torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)).numpy().reshape(-1)
        a = np.clip(a, -max_action, max_action)
        obs, r, term, trunc, _ = env.step(a)
        ret += float(r)
        n += 1
        if term or trunc:
            break
    return dict(frames=frames, obs=np.asarray(obss), ret=ret, length=n,
                terminated=bool(term), truncated=bool(trunc))


def pad_frames(frames, target):
    if not frames:
        return frames
    if len(frames) >= target:
        return frames[:target]
    return frames + [frames[-1]] * (target - len(frames))


def label(img, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 18], fill=(0, 0, 0))
    d.text((5, 4), text, fill=(255, 255, 255))
    return np.asarray(im)


def write_video(path: Path, frames):
    try:
        import imageio.v2 as imageio
        imageio.mimwrite(str(path), frames, fps=30, quality=8)
        return True
    except Exception as exc:
        print(f"   [视频] imageio 失败: {exc}")
    try:
        import cv2
        h, w, _ = frames[0].shape
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        for f in frames:
            vw.write(f[:, :, ::-1])
        vw.release()
        return True
    except Exception as exc:
        print(f"   [视频] cv2 失败: {exc}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", default=None)
    ap.add_argument("--weight", default="latest_checkpoint.pth")
    ap.add_argument("--which", choices=["actor", "ema"], default="actor")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="demo.mp4")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    run_a = Path(args.a).expanduser().resolve()
    ck_a, net_a, env_name = load_policy(run_a, args.weight, args.which)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    net_a = net_a.to(device)

    net_b = None
    run_b = None
    if args.b:
        run_b = Path(args.b).expanduser().resolve()
        _, net_b, env_b = load_policy(run_b, args.weight, args.which)
        if env_b != env_name:
            print(f"环境不一致: {env_name} vs {env_b}")
            return 2
        net_b = net_b.to(device)

    render = not args.no_video
    env = make_env(env_name, render)
    max_action = float(np.asarray(env.action_space.high).reshape(-1)[0])
    base_seed = int(ck_a.get("seed") or 0)
    seeds = [base_seed + 500_000 + i for i in range(args.episodes)]

    print(f"环境 {env_name}  网络 {args.which}  回合 {args.episodes}  种子 {seeds[0]}..{seeds[-1]}")
    all_frames, stats = [], []
    for k, s in enumerate(seeds):
        ra = rollout(net_a, env, s, max_action, render)
        rb = rollout(net_b, env, s, max_action, render) if net_b is not None else None
        msg = f"   回合 {k+1} seed={s}: A 回报 {ra['ret']:9.1f} 长度 {ra['length']:5d}"
        if rb is not None:
            msg += f" | B 回报 {rb['ret']:9.1f} 长度 {rb['length']:5d}"
        print(msg)
        stats.append(dict(seed=s,
                          a=dict(ret=ra["ret"], length=ra["length"],
                                 terminated=ra["terminated"], truncated=ra["truncated"]),
                          b=None if rb is None else dict(ret=rb["ret"], length=rb["length"],
                                                         terminated=rb["terminated"],
                                                         truncated=rb["truncated"])))
        if render:
            if rb is None:
                all_frames += [label(f, f"A: {run_a.name}  seed {s}") for f in ra["frames"]]
            else:
                n = max(len(ra["frames"]), len(rb["frames"]))
                fa = pad_frames(ra["frames"], n)
                fb = pad_frames(rb["frames"], n)
                if fa and fb:
                    for x, y in zip(fa, fb):
                        all_frames.append(np.concatenate(
                            [label(x, f"A: {run_a.name}"), label(y, f"B: {run_b.name}")], axis=1))
    env.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if render and all_frames:
        ok = write_video(out, all_frames)
        print(f"   视频{'已保存' if ok else '保存失败'}: {out}  帧数 {len(all_frames)}")
    elif render:
        print("   [提示] 没抓到画面，可能是无渲染环境。试试 MUJOCO_GL=egl，或加 --no-video")

    side = "a" if net_b is None else "ab"
    def agg(key):
        xs = [st[side][key] for st in stats if st[side]]
        return float(np.mean(xs)) if xs else float("nan")
    print(f"\n== 汇总 ==")
    print(f"   A 平均回报 {agg('ret'):.1f}  平均长度 {agg('length'):.0f}")
    if net_b is not None:
        print(f"   B 平均回报 {agg('ret'):.1f}  平均长度 {agg('length'):.0f}")

    meta = out.with_suffix(".json")
    write_new_text(meta, json.dumps(dict(env=env_name, which=args.which, seeds=seeds,
                                    a=run_a.name, b=None if run_b is None else run_b.name,
                                    episodes=stats), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"   数据已保存 {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
