# -*- coding: utf-8 -*-
"""Collect probe batches while keeping the learner weights fixed."""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))


from scripts.analysis_io import write_new_text

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


STRICT_KEYS = {"eval_actor", "stagnation_detector", "transition_value_mode", "mutation_style"}


def load_ckpt(run_dir: Path, weight: str = "latest_checkpoint.pth"):
    p = run_dir / weight
    if not p.is_file():
        raise FileNotFoundError(f"找不到 {p}")
    return torch.load(p, map_location="cpu", weights_only=False)


def _config_module():
    """Import the configuration module used by the training code."""
    import importlib.util

    path = REPO / "tedi" / "config.py"
    spec = importlib.util.spec_from_file_location("_tedi_config_standalone", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_config(raw: dict, env: str):
    mod = _config_module()
    ENV_PARAMS, EnvironmentConfig = mod.ENV_PARAMS, mod.EnvironmentConfig

    overrides, dropped = {}, []
    for k, v in (raw or {}).items():
        if k not in ENV_PARAMS[env]:
            continue
        default = ENV_PARAMS[env].get(k)


        if callable(default) and not callable(v):
            dropped.append(k)
            continue
        if v is None and k in STRICT_KEYS:
            continue
        overrides[k] = v
    if dropped:
        print(f"   [配置] 丢弃被序列化成字符串的可调用项: {dropped}")
    cfg = EnvironmentConfig.create(env, overrides)
    cfg.validate()
    return cfg


def build_framework(ck, cfg, device, work_dir: Path, log_candidates: bool = False):
    from tedi.trainer import TEDIFramework

    fw = TEDIFramework(
        config=cfg,
        total_steps=int(ck.get("step") or 1_000_000),
        pop_size=int(ck.get("pop_size") or 6),
        seed=int(ck.get("seed") or 0),
        run_dir=work_dir,
        device=device,
        replay_on_cpu=True,
        disable_evolution=False,
        disable_trigger=False,
        disable_filtering=False,
        log_candidates=bool(log_candidates),
    )
    td3_sd = ck["td3"]

    for name in ("actor", "actor_target", "ema_actor", "critic", "critic_target"):
        if name not in td3_sd:
            continue
        net = getattr(fw.td3, name)
        net.load_state_dict(td3_sd[name])
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
    return fw


def historical(run_dir: Path):
    out = []
    p = run_dir / "events.jsonl"
    if not p.is_file():
        return out
    with p.open(encoding="utf-8") as h:
        for line in h:
            if '"evolution"' not in line:
                continue
            e = json.loads(line)
            if e.get("type") != "evolution":
                continue
            q = e.get("qualified_transitions_per_policy") or []
            out.append(dict(step=int(e["main_step"]),
                            qualified=int(sum(q)),
                            injected=int(e.get("injected_transitions", 0)),
                            budget=int(e.get("fixed_injection_budget", 0))))
    return out


def last_event(run_dir: Path):
    p = run_dir / "events.jsonl"
    if not p.is_file():
        return None
    last = None
    with p.open(encoding="utf-8") as h:
        for line in h:
            if '"evolution"' in line:
                e = json.loads(line)
                if e.get("type") == "evolution":
                    last = e
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--batches", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--keep-workdir", action="store_true")
    ap.add_argument("--penalty", choices=["off", "auto"], default="off",
                    help="off=所有批次都不启用 OOD 惩罚（推荐，批次可比）；"
                         "auto=复现旧行为（惩罚参照探针自己的 buffer，会随批次漂移）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--weight", default="latest_checkpoint.pth",
                    help="要加载的 checkpoint 文件名（默认 latest_checkpoint.pth）")
    ap.add_argument("--probe_tag", default="candidates",
                    help="候选保存目录后缀：<run>/probe_<tag>，用于保留多次探针的结果")
    ap.add_argument("--dump_candidates", action="store_true",
                    help="把本次探针的全部候选写到 <run>/probe_candidates/，供 admission_bias 使用")
    args = ap.parse_args()

    run_dir = Path(args.run).expanduser().resolve()
    output_path = Path(args.out) if args.out else run_dir / "frozen_probe_diag.json"
    if output_path.exists() or output_path.with_suffix(".csv").exists():
        raise FileExistsError("Probe outputs already exist; choose a new --out path")
    if args.dump_candidates and (run_dir / ("probe_" + str(args.probe_tag))).exists():
        raise FileExistsError("Probe candidate directory already exists; choose a new --probe_tag")
    ck = load_ckpt(run_dir, args.weight)
    env = str(ck.get("env_name") or (ck.get("config") or {}).get("environment"))
    hist = historical(run_dir)

    print(f"== {run_dir.name}")
    print(f"   env={env}  step={ck.get('step')}  seed={ck.get('seed')}  pop_size={ck.get('pop_size')}")
    print(f"   历史进化事件 {len(hist)} 次", end="")
    if hist:
        q = [h["qualified"] for h in hist]
        inj = [h["injected"] for h in hist]
        print(f"  合格数 {min(q)}–{max(q)}（中位 {int(np.median(q))}）"
              f"  注入 {min(inj)}–{max(inj)}（中位 {int(np.median(inj))}）"
              f"  空注入 {sum(1 for x in inj if x == 0)}/{len(inj)}")
    else:
        print("（没有历史事件，无法对照）")

    try:
        cfg = build_config(ck.get("config") or {}, env)
    except Exception as exc:
        print(f"   [失败] 配置重建出错: {type(exc).__name__}: {exc}")
        return 2
    print(f"   配置重建 OK: horizon={cfg.evolution_horizon} budget={cfg.max_inject_per_evo} "
          f"keep_q={cfg.trajectory_keep_quantile} value_mode={cfg.transition_value_mode} "
          f"penalty_initial={cfg.penalty_initial}")

    if args.penalty == "off" and float(cfg.penalty_initial or 0.0) > 0.0:
        print(f"   [惩罚] 原始 penalty_initial={cfg.penalty_initial}，探针内已关闭。")
        print("          理由：训练时参照主 replay（几十万条真实转移），探针里 replay 只有自己")
        print("          刚注入的几百条，参照分布不同会让惩罚随批次漂移，批次之间不可比。")
        cfg.penalty_initial = 0.0
        cfg.penalty_final = 0.0
    elif args.penalty == "auto" and float(cfg.penalty_initial or 0.0) > 0.0:
        print("   [惩罚] --penalty auto：惩罚参照探针自己的 buffer，第 1 批为空不生效，")
        print("          第 2 批起突然生效 —— 相邻批次的合格数落差有一部分来自这里，不是接纳噪声。")

    if args.inspect:
        print("   --inspect 结束（未启动 MuJoCo，未跑探针）")
        return 0

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    work = Path(tempfile.mkdtemp(prefix="frozen_probe_"))

    try:
        seed = int(ck.get("seed") or 0)
        torch.manual_seed(seed)
        np.random.seed(seed)
        fw = build_framework(ck, cfg, device, work, log_candidates=args.dump_candidates)

        fw.evolution_count = len(hist) if hist else 0
        steps = [h["step"] for h in hist] if hist else None

        rows = []
        for i in range(args.batches):
            step = steps[i % len(steps)] if steps else int((i + 1) * 50_000)
            buf_before = int(getattr(fw.buffer, "size", 0))
            fw.run_evolution(step=step, recent_mean=0.0)
            ev = last_event(work)
            if ev is None:
                print(f"   [警告] 第 {i+1} 批没有产生事件")
                continue
            q = sum(ev.get("qualified_transitions_per_policy") or [])
            roles = ev.get("policy_roles") or []
            injectable = ev.get("injectable_policies") or []
            available = {roles[k] for k in injectable if k < len(roles)}
            by_role = ev.get("injected_transitions_by_role") or {}
            note = [r for r in ("conservative", "explorer", "destroyer")
                    if int(by_role.get(r, 0) or 0) == 0 and r not in available]
            rows.append(dict(batch=i + 1, step=step, qualified=int(q),
                             injected=int(ev.get("injected_transitions", 0)),
                             budget=int(ev.get("fixed_injection_budget", 0)),
                             parent_return=ev.get("parent_policy_return"),
                             policy_returns=ev.get("policy_returns"),
                             roles=by_role, injectable_policies=injectable,
                             role_absent=note, buffer_size_before=buf_before,
                             penalty_active=float(cfg.penalty_initial or 0.0) > 0.0))
            pr = ev.get("parent_policy_return")
            print(f"   批次 {i+1}: step={step} 合格 {q} 注入 {ev.get('injected_transitions')}"
                  f" / 预算 {ev.get('fixed_injection_budget')}"
                  f" 父策略 {pr if pr is None else round(float(pr), 1)}"
                  f" 角色 {by_role}" + (f"  [无可用成员: {note}]" if note else ""))

        if not rows:
            print("   [失败] 没有有效批次")
            return 3

        print("\n== 对照 ==")
        print(f"{'':10s}{'探针均值':>12}{'历史中位':>12}{'历史区间':>16}")
        pq = float(np.mean([r["qualified"] for r in rows]))
        pi = float(np.mean([r["injected"] for r in rows]))
        if hist:
            hq = [h["qualified"] for h in hist]
            hi = [h["injected"] for h in hist]
            print(f"{'合格数':10s}{pq:>12.0f}{np.median(hq):>12.0f}{f'{min(hq)}-{max(hq)}':>16}")
            print(f"{'注入数':10s}{pi:>12.1f}{np.median(hi):>12.1f}{f'{min(hi)}-{max(hi)}':>16}")
            zero = sum(1 for r in rows if r["injected"] == 0)
            print(f"{'空注入':10s}{zero:>12d}{sum(1 for x in hi if x == 0):>12d}")
        else:
            print(f"{'合格数':10s}{pq:>12.0f}{'-':>12}{'-':>16}")
            print(f"{'注入数':10s}{pi:>12.1f}{'-':>12}{'-':>16}")

        if args.dump_candidates:
            src = work / "candidates"
            dst_root = run_dir / ("probe_" + str(args.probe_tag))
            dst = dst_root / "candidates"
            dst.mkdir(parents=True, exist_ok=False)
            n_copied = 0
            if src.is_dir():
                for f in sorted(src.iterdir()):
                    shutil.copy2(str(f), str(dst / f.name))
                    n_copied += 1
            cfg_src = run_dir / "run_config.json"
            cfg_dst = dst_root / "run_config.json"
            if cfg_src.is_file():
                probe_config = json.loads(cfg_src.read_text(encoding="utf-8"))
                probe_config["configuration"] = cfg.serializable()
                probe_config["diagnostic_penalty_mode"] = args.penalty
                probe_config["filtering_disabled"] = False
                probe_config["unlimited_injection"] = False
                write_new_text(cfg_dst.with_name("probe_settings.json"),
                               json.dumps(probe_config, ensure_ascii=False, indent=2))
                with cfg_dst.open("x", encoding="utf-8") as handle:
                    json.dump(probe_config, handle, ensure_ascii=False, indent=2)
            print(f"\n   候选已落盘：{dst}（{n_copied} 个文件）")
            print(f"   配置已复制：{cfg_dst}")
            print(f"   下一步： python -m scripts.admission_bias --run {dst_root} --verify")

        out = Path(args.out) if args.out else run_dir / "frozen_probe_diag.json"
        import hashlib
        ck_path = run_dir / args.weight
        ck_sha = hashlib.sha256(ck_path.read_bytes()).hexdigest() if ck_path.is_file() else None
        write_new_text(out, json.dumps(dict(run=run_dir.name, env=env,
                                       checkpoint=args.weight,
                                       checkpoint_sha256=ck_sha,
                                       probe_tag=str(args.probe_tag),
                                       batches=rows, historical=hist),
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        csv_path = out.with_suffix(".csv")
        with csv_path.open("x", encoding="utf-8-sig") as fh:
            fh.write("batch,step,parent_return,qualified,injected,budget,c_share,e_share,d_share\n")
            for r in rows:
                roles = r.get("roles") or {}
                tot = max(1, int(r["injected"]))
                fh.write("%d,%d,%s,%d,%d,%d,%.3f,%.3f,%.3f\n" % (
                    r["batch"], r["step"],
                    "" if r.get("parent_return") is None else "%.2f" % float(r["parent_return"]),
                    r["qualified"], r["injected"], r["budget"],
                    (roles.get("conservative", 0) or 0) / tot,
                    (roles.get("explorer", 0) or 0) / tot,
                    (roles.get("destroyer", 0) or 0) / tot))
        print(f"   已写出 {csv_path}")

        print(f"\n   已写出 {out}")
        return 0
    except Exception as exc:
        import traceback
        print(f"   [失败] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 4
    finally:
        if not args.keep_workdir:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"   工作目录保留在 {work}")


if __name__ == "__main__":
    raise SystemExit(main())
