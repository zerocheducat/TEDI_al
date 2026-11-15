
import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import sys, io
from pypdf import PdfReader
src = r"F:\workwork\TEDI_Ttraining\22691_Evolution_Only_When_Need.pdf"
rd = PdfReader(src)
out = []
for i, p in enumerate(rd.pages, 1):
    t = p.extract_text() or ""
    out.append(f"\n===== PAGE {i} =====\n" + t)
txt = "".join(out)
open(r"F:\workwork\TEDI_Ttraining\_paper_text.txt", "x", encoding="utf-8").write(txt)
print("pages:", len(rd.pages), "chars:", len(txt))
