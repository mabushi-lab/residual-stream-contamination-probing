"""
make_arxiv.py
Assemble an arXiv-ready source tarball from paper/.

arXiv compiles from the tree you upload, so `\\graphicspath{{../figures/}}`
cannot work: the figures live outside the paper directory and `../` escapes
the upload. This copies the paper into a self-contained tree, rewrites the
graphics path, brings along only the figures actually included, and compiles
the result to prove it builds before you upload it.

    python3 src/make_arxiv.py            build and verify
    python3 src/make_arxiv.py --no-check  skip the test compile

Output: build/arxiv/ and build/arxiv.tar.gz

Nothing here edits paper/. The tarball is disposable; rebuild it whenever the
paper changes.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT, PAPER, GENERATED, FIGURES

BUILD = os.path.join(ROOT, "build")
OUT = os.path.join(BUILD, "arxiv")
TARBALL = os.path.join(BUILD, "arxiv.tar.gz")

MAIN = "thesis.tex"
GRAPHICS_RE = re.compile(r"\\graphicspath\{\{[^}]*\}\}")
INCLUDE_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
INPUT_RE = re.compile(r"\\input\{([^}]+)\}")


def build(check=True):
    src = os.path.join(PAPER, MAIN)
    if not os.path.exists(src):
        raise SystemExit(f"no {src}")

    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "generated"), exist_ok=True)

    tex = open(src).read()

    # arXiv flattens to the uploaded tree; ../ would escape it.
    tex, n = GRAPHICS_RE.subn(r"\\graphicspath{{figures/}}", tex)
    if n == 0:
        print("  warning: no \\graphicspath found, figures may not resolve")
    open(os.path.join(OUT, MAIN), "w").write(tex)
    print(f"  {MAIN}  (graphicspath rewritten)")

    # Only the figures the paper actually includes. Parsing beats a hardcoded
    # list, which would rot the next time a figure is added or dropped.
    missing = []
    for name in sorted(set(INCLUDE_RE.findall(tex))):
        p = os.path.join(FIGURES, os.path.basename(name))
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(OUT, "figures", os.path.basename(name)))
        else:
            missing.append(name)
    n_fig = len(os.listdir(os.path.join(OUT, "figures")))
    print(f"  figures/  ({n_fig} included)")
    if missing:
        raise SystemExit("missing figures, run `make figures` first:\n  "
                         + "\n  ".join(missing))

    for name in sorted(set(INPUT_RE.findall(tex))):
        p = os.path.join(PAPER, name)
        if not os.path.exists(p):
            raise SystemExit(f"missing {name}, run `make figures` first")
        shutil.copy2(p, os.path.join(OUT, os.path.basename(os.path.dirname(name))
                                     or ".", os.path.basename(name)))
    print(f"  generated/  ({len(os.listdir(os.path.join(OUT, 'generated')))} files)")

    if check:
        print("  test compile...")
        for i in range(2):
            r = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", MAIN],
                cwd=OUT, capture_output=True, text=True)
            if r.returncode != 0:
                tail = "\n".join(r.stdout.strip().splitlines()[-25:])
                raise SystemExit(f"arXiv tree does not compile:\n{tail}")
        log = open(os.path.join(OUT, "thesis.log")).read()
        pages = re.search(r"Output written on .*?\((\d+) pages", log)
        undefined = log.count("LaTeX Warning: Reference") \
            + log.count("LaTeX Warning: Citation")
        print(f"    compiles, {pages.group(1) if pages else '?'} pages, "
              f"{undefined} undefined references")
        for ext in (".aux", ".log", ".out", ".toc", ".pdf"):
            f = os.path.join(OUT, MAIN[:-4] + ext)
            if os.path.exists(f):
                os.remove(f)

    with tarfile.open(TARBALL, "w:gz") as t:
        for root, _, files in os.walk(OUT):
            for f in sorted(files):
                full = os.path.join(root, f)
                t.add(full, arcname=os.path.relpath(full, OUT))
    mb = os.path.getsize(TARBALL) / 1e6
    print(f"\n  {os.path.relpath(TARBALL, ROOT)}  ({mb:.1f} MB)")
    print("  Upload this to arXiv as TeX source, not as a PDF.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-check", action="store_true",
                    help="skip the verification compile")
    a = ap.parse_args()
    build(check=not a.no_check)
