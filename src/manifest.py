"""
manifest.py
Record a SHA256 of every generated artefact, so a rerun can be checked
against the run the paper reports.

    python3 manifest.py            write MANIFEST.sha256
    python3 manifest.py --check    compare against it
"""
from __future__ import annotations
import hashlib, os, subprocess, sys, json, platform

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import ROOT, VAL_RESULTS, GENERATED, FIGURES, RUNS, PAPER

_os.chdir(ROOT)
TRACKED = [
    "validation/results/phase0c_results.json",
    "validation/results/phase0b_results.json",
    "validation/results/phase0_results.json",
    "paper/generated/results_macros.tex",
    "paper/generated/results_tables.tex",
    "paper/generated/phase1_macros.tex",
    "paper/generated/phase1_table.tex",
    "experiments/phase1_summary.json",
    "paper/thesis.pdf", "paper/abstract_of_thesis.pdf",
]
TRACKED_GLOB2 = ("experiments/runs/phase1", (".json",))
TRACKED_GLOB = ("figures", (".pdf", ".png"))
# The item sets are hashed too. This paper's claim is about how the reference
# set is built, so an audit that cannot be tied to the exact items it ran on
# is not reproducible in the sense that matters here.
TRACKED_GLOB3 = ("experiments/data", (".jsonl",))

# Pile extracts are rebuilt rather than redistributed (see .gitignore), so a
# fresh clone is missing them by design. Report that separately from a real
# failure, otherwise `make verify` fails out of the box for every new user.
REBUILDABLE = "experiments/data/pile_"
REBUILD_CMD = (
    "  python3 src/build_itemsets.py --set pile --subdomain Wikipedia --n 500\n"
    "  python3 src/build_itemsets.py --set pile --subdomain GitHub --n 500")


def _files():
    out = [f for f in TRACKED if os.path.exists(f)]
    for d, exts in (TRACKED_GLOB, TRACKED_GLOB2, TRACKED_GLOB3):
        if os.path.isdir(d):
            out += sorted(os.path.join(d, f) for f in os.listdir(d)
                          if f.endswith(exts))
    return sorted(out)


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def env():
    import numpy, scipy, sklearn, matplotlib
    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform()}


def main():
    check = "--check" in sys.argv
    cur = {p: _sha(p) for p in _files()}
    if check:
        if not os.path.exists("MANIFEST.sha256"):
            sys.exit("no MANIFEST.sha256 to check against")
        old = json.load(open("MANIFEST.sha256"))["files"]
        bad = [p for p in cur if p in old and cur[p] != old[p]]
        absent = [p for p in old if p not in cur]
        miss = [p for p in absent if not p.startswith(REBUILDABLE)]
        rebuild = [p for p in absent if p.startswith(REBUILDABLE)]
        for p in bad:
            print(f"CHANGED  {p}")
        for p in miss:
            print(f"MISSING  {p}")
        for p in rebuild:
            print(f"REBUILD  {p}")
        new = [p for p in cur if p not in old]
        for p in new:
            print(f"NEW      {p}")
        print(f"{len(cur)} artefacts, {len(bad)} changed, {len(miss)} missing"
              + (f", {len(rebuild)} not redistributed" if rebuild else ""))
        if rebuild:
            print("\nThe Pile extracts are not shipped with the repository. "
                  "Rebuild them with:\n" + REBUILD_CMD +
                  "\nThe upstream revision is pinned, so the hashes recorded "
                  "here should reproduce exactly.")
        sys.exit(1 if (bad or miss) else 0)
    json.dump({"env": env(), "files": cur}, open("MANIFEST.sha256", "w"),
              indent=2, sort_keys=True)
    print(f"wrote MANIFEST.sha256 with {len(cur)} artefacts")


if __name__ == "__main__":
    main()
