"""Create derived outputs without replacing existing experimental records."""

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path


def write_new_text(path, text, encoding="utf-8"):
    path = Path(path)
    if path.name in {"training_summary.json", "run_config.json", "events.jsonl"}:
        raise ValueError(f"Refusing to write a raw experiment record: {path}")
    with path.open("x", encoding=encoding) as handle:
        handle.write(text)
