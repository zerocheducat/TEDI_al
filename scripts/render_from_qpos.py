# -*- coding: utf-8 -*-
"""Render from qpos."""


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
    sd = td3["actor" if which == "actor" else "ema_actor"]

    net = _actor_class()(
        int(sd["l1.weight"].shape[1]),
        int(sd["l3.weight"].shape[0]),
        1.0,
        use_layernorm=("ln1.weight" in sd),
    )
    net.load_state_dict(sd)
    net.eval()
    return ck, net, env_name


def find_xml(env):
    u = env.unwrapped
    for attr in ("fullpath", "model_path", "xml_file"):
        p = getattr(u, attr, None)
        if not p:
            continue
        p = str(p)
        if p.endswith(".xml"):
            return p
        cand = Path(p) / "hopper.xml"
        if cand.is_file():
            return str(cand)
    try:
        import gymnasium.envs.mujoco as gm
        cand = Path(gm.__file__).parent / "assets" / "hopper.xml"
        if cand.is_file():
            return str(cand)
    except Exception:
        pass
    return None


def simulate(net, env, seed, max_action):
    """Collect simulation frames and recorded trajectory data."""
    obs, _ = env.reset(seed=int(seed))
    qpos = []
    ret, n = 0.0, 0
    u = env.unwrapped
    while True:
        qpos.append(np.array(u.sim.data.qpos, dtype=np.float64).copy()
                    if hasattr(u, "sim") else np.array(u.data.qpos, dtype=np.float64).copy())
        with torch.inference_mode():
            a = net(torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)).numpy().reshape(-1)
        a = np.clip(a, -max_action, max_action)
        obs, r, term, trunc, _ = env.step(a)
        ret += float(r)
        n += 1
        if term or trunc:
            break
    return np.asarray(qpos), ret, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="模型 A 的 run 目录（TEDI）")
    ap.add_argument("--b", required=True, help="模型 B 的 run 目录（matched）")
    ap.add_argument("--weight", default="latest_checkpoint.pth")
    ap.add_argument("--which", choices=["actor", "ema"], default="ema")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--out", default="compare.mp4")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--speed", type=float, default=1.0, help="播放速度倍率，1.0 = 真实时间")
    ap.add_argument("--fps", type=int, default=0, help="直接指定帧率，0 = 按 --speed 自动算")
    ap.add_argument("--no-follow", action="store_true", help="关掉相机跟随")
    ap.add_argument("--before-fall", type=float, default=0.0,
                    help="只保留「先摔倒前」这么多秒（0 = 从头播，默认）")
    ap.add_argument("--after-fall", type=float, default=0.0,
                    help="只保留摔倒后这么多秒（0 = 一直播到结束，默认）")
    ap.add_argument("--label-a", default="A", help="左边画面顶部显示的名字")
    ap.add_argument("--label-b", default="B", help="右边画面顶部显示的名字")
    ap.add_argument("--hold", type=float, default=0.8,
                    help="某一方摔倒时，整个画面停顿这么多秒，让对比看得清")
    ap.add_argument("--show-step", action="store_true", help="右下角显示当前步数")
    args = ap.parse_args()

    import gymnasium as gym
    import mujoco

    run_a, run_b = Path(args.a).expanduser().resolve(), Path(args.b).expanduser().resolve()
    ck_a, net_a, env_name = load_policy(run_a, args.weight, args.which)
    _, net_b, env_b = load_policy(run_b, args.weight, args.which)
    if env_name != env_b:
        print(f"环境不一致: {env_name} vs {env_b}")
        return 2

    env = gym.make(env_name)
    max_action = float(np.asarray(env.action_space.high).reshape(-1)[0])
    xml = find_xml(env)
    if xml is None:
        print("找不到模型 XML，无法用新版 mujoco 渲染")
        return 3
    print(f"环境 {env_name}   网络 {args.which}")
    print(f"XML  {xml}")

    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, args.height, args.width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.0, 0.0, 1.0]
    cam.distance = 3.2
    cam.azimuth = 120.0
    cam.elevation = -18.0


    try:
        dt = float(env.unwrapped.frame_skip) * float(model.opt.timestep)
    except Exception:
        dt = 0.01
    fps = args.fps if args.fps > 0 else max(1, int(round(args.speed / dt)))
    print(f"   每步仿真 {dt*1000:.1f} ms  →  帧率 {fps} fps（{args.speed:.2f}× 实时）")

    base = int(ck_a.get("seed") or 0)
    seeds = [base + 500_000 + i for i in range(args.episodes)]

    from PIL import Image, ImageDraw, ImageFont

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    class _Writer(object):
        """Write rendered frames to the selected output format."""

        def __init__(self, path, size, fps):
            self.kind, self.impl, self.n = None, None, 0
            try:
                import imageio.v2 as iio
                self.impl = iio.get_writer(str(path), fps=fps, quality=8, macro_block_size=None)
                self.kind = "imageio"
            except Exception as exc:
                print(f"   [提示] imageio 不可用（{exc}），改用 cv2")
                import cv2
                self.impl = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                self.kind = "cv2"

        def append(self, frame):
            if self.kind == "imageio":
                self.impl.append_data(frame)
            else:
                self.impl.write(frame[:, :, ::-1])
            self.n += 1

        def close(self):
            try:
                if self.kind == "imageio":
                    self.impl.close()
                else:
                    self.impl.release()
            except Exception:
                pass

    def _load_font(size):
        for name in ("msyh.ttc", "msyhbd.ttc", "arial.ttf", "DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue
        return ImageFont.load_default()

    font_big = _load_font(30)
    font_small = _load_font(20)

    def _text_size(draw, txt, font):
        """Measure text bounds for frame annotations."""
        try:
            box = draw.textbbox((0, 0), txt, font=font)
            return box[2] - box[0], box[3] - box[1]
        except Exception:
            pass
        try:
            return draw.textsize(txt, font=font)
        except Exception:
            return (8 * len(txt), 12)

    writer = _Writer(out, (args.width * 2, args.height), fps)
    recs = []
    for k, s in enumerate(seeds):
        qa, ra_ret, ra_n = simulate(net_a, env, s, max_action)
        qb, rb_ret, rb_n = simulate(net_b, env, s, max_action)
        print(f"   回合 {k+1} seed={s}: A 回报 {ra_ret:8.1f} ({ra_n} 步) | "
              f"B 回报 {rb_ret:8.1f} ({rb_n} 步)")
        recs.append(dict(seed=s, a=dict(ret=ra_ret, steps=ra_n), b=dict(ret=rb_ret, steps=rb_n)))

        n = max(len(qa), len(qb))
        fall = min(len(qa), len(qb))
        if args.before_fall <= 0 and args.after_fall <= 0:
            start, end = 0, n
        else:
            end = min(n, fall + int(round(args.after_fall * fps)))
            start = 0 if args.before_fall <= 0 else max(0, fall - int(round(args.before_fall * fps)))
        print(f"      播放第 {start}-{end} 帧（先结束的一方在第 {fall} 步）")
        for t in range(start, end):
            imgs = []
            for q, lab in ((qa, args.label_a), (qb, args.label_b)):
                qq = q[min(t, len(q) - 1)]
                if len(qq) != model.nq:
                    print(f"   qpos 维度不匹配: {len(qq)} vs {model.nq}")
                    writer.close()
                    return 4
                data.qpos[:] = qq
                mujoco.mj_forward(model, data)
                if not args.no_follow:

                    cam.lookat[0] = float(qq[0])
                    cam.lookat[1] = 0.0
                    cam.lookat[2] = float(np.clip(qq[1], 0.4, 1.6))
                renderer.update_scene(data, camera=cam)
                im = np.asarray(renderer.render())
                pil = Image.fromarray(im)
                d = ImageDraw.Draw(pil)

                d.rectangle([0, 0, pil.width, 30], fill=(0, 0, 0))
                d.text((8, 4), f"{lab}   seed {k+1}", font=font_small, fill=(255, 255, 255))

                if t >= len(q):
                    d.rectangle([0, pil.height - 44, pil.width, pil.height], fill=(150, 0, 0))
                    d.text((10, pil.height - 40), "FALLEN", font=font_big, fill=(255, 255, 255))

                if args.show_step:
                    txt = f"step {t}"
                    tw, th = _text_size(d, txt, font_big)
                    x = pil.width - tw - 16
                    y = pil.height - th - 30
                    d.rectangle([x - 10, y - 6, x + tw + 10, y + th + 8], fill=(0, 0, 0))
                    d.text((x, y), txt, font=font_big, fill=(255, 235, 0))
                imgs.append(np.asarray(pil))
            writer.append(np.concatenate(imgs, axis=1))
            if args.hold > 0 and t == fall - 1:

                held = np.asarray(Image.fromarray(np.concatenate(imgs, axis=1)))
                for _ in range(int(round(args.hold * fps))):
                    writer.append(held)

    writer.close()
    print(f"   视频已保存 {out.resolve()}  共 {writer.n} 帧")
    env.close()
    try:
        renderer.close()
    except AttributeError:
        pass

    meta = out.with_suffix(".json")
    write_new_text(meta, json.dumps(dict(env=env_name, which=args.which, a=run_a.name,
                                    b=run_b.name, episodes=recs), ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"   数据已保存 {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
