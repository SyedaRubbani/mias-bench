#!/usr/bin/env python3
"""Build viz/index.html by inlining results/viz_data.json into the template.

The explorer has to be one self-contained file: it is published as an artifact
and opened from a GitHub link, and in neither case can it fetch a sibling JSON.

Run:  python tools/build_viz.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "viz" / "explorer.template.html"
DATA = ROOT / "results" / "viz_data.json"
OUT = ROOT / "viz" / "index.html"


def main() -> int:
    if not DATA.exists():
        print("results/viz_data.json missing - run "
              "python -m mias.experiments.make_viz_data first", file=sys.stderr)
        return 1

    raw = DATA.read_text()
    payload = json.loads(raw)          # fail loudly on malformed data

    # The JSON sits inside a <script> element, so the only sequence that can
    # break out of it is a literal "</script"; nothing else needs escaping.
    embedded = raw.replace("</", "<\\/")

    html = TEMPLATE.read_text()
    if "__DATA__" not in html:
        print("template has no __DATA__ placeholder", file=sys.stderr)
        return 1
    html = html.replace("__DATA__", embedded)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)

    conds = len(payload["conditions"])
    pols = len(next(iter(payload["conditions"].values()))["policies"])
    rounds = len(next(iter(next(iter(payload["conditions"].values()))
                           ["policies"].values()))["rounds"])
    print(f"wrote {OUT.relative_to(ROOT)} "
          f"({OUT.stat().st_size // 1024} KB; "
          f"{conds} conditions x {pols} policies x {rounds} rounds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
