# -*- coding: utf-8 -*-
"""Compatibility entry point for the corrected frozen reward diagnostic."""

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.frozen_reward_diag import main

if __name__ == "__main__":
    raise SystemExit(main())
