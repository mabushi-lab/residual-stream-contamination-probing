"""
collect_phase1.py
Turn a directory of audit JSONs into the paper's first real table.

    python3 collect_phase1.py --dir runs/phase1
"""
from __future__ import annotations
import argparse, glob, json, os

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))
from paths import VAL_RESULTS, GENERATED, FIGURES, RUNS, DATA, CACHE, EXPERIMENTS, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=s(RUNS / "phase1"))
    ap.add_argument("--out", default=s(GENERATED / "phase1_table.tex"))
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.dir, "*.json"))):
        o = json.load(open(f))
        if o.get("dry_run"):
            continue
        name = os.path.basename(f)[:-5]
        model, _, split = name.partition("__")
        rows.append({
            "model": model.replace("_", "/"), "split": split,
            "ba_nuis": o["ba_nuisance"],
            "baseline_ok": o["contrast"]["recentred"],
            "T_adj": o["contrast"]["T_adj"],
            "p": o["contrast"]["p_value"],
        })
    if not rows:
        raise SystemExit(f"no audit results in {a.dir}")

    L = [r"\begin{table}[t]", r"\small", r"\centering",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
         r"Model & Split & $\BA_{\nuis}$ & $T_{\mathrm{adj}}$ & $p$ \\",
         r"\midrule"]
    for r in rows:
        t = "n/a" if not r["baseline_ok"] else f"{r['T_adj']:+.4f}"
        p = "n/a" if not r["baseline_ok"] else f"{r['p']:.3f}"
        L.append(f"{r['model']} & {r['split'].replace('_',' ')} & "
                 f"{r['ba_nuis']:.3f} & {t} & {p} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Phase 1. $\BA_{\nuis}$ is what a classifier with no "
          r"access to the model achieves on each split; values far above "
          r"$0.5$ mean the split is separable blind and no contamination "
          r"conclusion can be drawn from it. Entries marked n/a had no "
          r"admissible baseline.}",
          r"\label{tab:phase1}", r"\end{table}"]
    open(a.out, "w").write("\n".join(L) + "\n")
    print(f"wrote {a.out} with {len(rows)} rows")
    for r in rows:
        print(f"  {r['model']:28s} {r['split']:16s} "
              f"BAnuis={r['ba_nuis']:.3f}  p={r['p'] if r['baseline_ok'] else 'n/a'}")

if __name__ == "__main__":
    main()
