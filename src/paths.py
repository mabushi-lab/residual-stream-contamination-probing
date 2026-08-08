"""
paths.py
One place that knows where things live.

Every script resolves its inputs and outputs through here rather than
assuming a working directory, so `make` from the project root, `python3
src/run_rscp_eval.py` from anywhere, and an editor's run button all agree.

Override the root with the RSCP_ROOT environment variable if you move things.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get(
    "RSCP_ROOT", Path(__file__).resolve().parent.parent))

SRC = ROOT / "src"
VALIDATION = ROOT / "validation"
VAL_RESULTS = VALIDATION / "results"
RENDER = ROOT / "render"
EXPERIMENTS = ROOT / "experiments"
DATA = EXPERIMENTS / "data"
RUNS = EXPERIMENTS / "runs"
PAPER = ROOT / "paper"
GENERATED = PAPER / "generated"
FIGURES = ROOT / "figures"
CACHE = ROOT / "cache"
DOCS = ROOT / "docs"
TESTS = ROOT / "tests"

for _d in (VAL_RESULTS, DATA, RUNS, GENERATED, FIGURES, CACHE):
    _d.mkdir(parents=True, exist_ok=True)


def s(p) -> str:
    """Path as a string, for APIs that want one."""
    return str(p)


if __name__ == "__main__":
    for name in ("ROOT", "SRC", "VALIDATION", "VAL_RESULTS", "RENDER",
                 "EXPERIMENTS", "DATA", "RUNS", "PAPER", "GENERATED",
                 "FIGURES", "CACHE", "DOCS", "TESTS"):
        print(f"{name:12s} {globals()[name]}")
