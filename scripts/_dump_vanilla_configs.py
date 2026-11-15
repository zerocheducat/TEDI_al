# -*- coding: utf-8 -*-
""" dump vanilla configs."""

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import json
from pathlib import Path
import os
import sys
import types

ROOT = Path(__file__).resolve().parent
if not (ROOT / "tedi").is_dir():
    ROOT = ROOT.parent
REPO = str(ROOT)
sys.path.insert(0, REPO)
pkg = types.ModuleType("tedi")
pkg.__path__ = [os.path.join(REPO, "tedi")]
sys.modules["tedi"] = pkg

from tedi.config import EnvironmentConfig  # noqa: E402

out = {}
for env in ["Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Swimmer-v2"]:
    out[env] = {
        "vanilla": EnvironmentConfig.create(env, {"vanilla_td3": True}).serializable(),
        "matched": EnvironmentConfig.create(env, {}).serializable(),
    }

path = os.path.join(REPO, "B0_TD3_configs.json")
with open(path, "x", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

for env, d in out.items():
    v, m = d["vanilla"], d["matched"]
    diff = sorted(k for k in v if k in m and v[k] != m[k])
    print(f"{env:16s} vanilla 共 {len(v)} 键，与 matched 有 {len(diff)} 项不同")
print(f"\n已写出: {path}")
