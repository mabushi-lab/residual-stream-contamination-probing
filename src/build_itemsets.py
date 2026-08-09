"""
build_itemsets.py
Turn public datasets into the JSONL item files run_rscp_eval.py expects.

Every builder enforces the two things the protocol needs and that are easy to
get wrong by hand: the prefix must stop before the answer, and the reference
set must be twice the size of the suspect set.

    python3 build_itemsets.py --list
    python3 build_itemsets.py --set pile --subdomain Wikipedia --n 500
    python3 build_itemsets.py --set gsm8k_gsm1k --n 500
    python3 build_itemsets.py --set wikimia --n 250

Requires `datasets`. Nothing else here needs it, and the builders are the only
part of this repository that touches the network.

A note on what these sets are for. Only `contam` gives ground truth. `pile`
gives exact membership but an approximate exchangeability guarantee.
`gsm8k_gsm1k` is a commissioned twin, the best available for a benchmark of
live interest. `wikimia` is included precisely because it is a temporal split
and therefore does NOT satisfy Requirement E: it is the object of the Phase 1
nuisance audit, not a set to draw contamination conclusions from.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))
from paths import VAL_RESULTS, GENERATED, FIGURES, RUNS, DATA, CACHE, EXPERIMENTS, s


SETS = {
    "pile": "Pile train vs. validation for a Pythia audit (construction E3). "
            "Exact membership; exchangeability approximate and known to fail "
            "on drifting subdomains.",
    "gsm8k_gsm1k": "GSM8K against its commissioned twin GSM1k (E2). The best "
                   "available reference set for a benchmark in live use.",
    "contam": "Deliberately contaminated checkpoints of Oren et al. (E1). "
              "Randomised injection, so exchangeability is exact.",
    "wikimia": "WikiMIA. A TEMPORAL split, which does not satisfy Requirement "
               "E. Included as the subject of the nuisance audit.",
}


# Upstream dataset revisions, pinned so a rebuild is byte-identical.
#
# Without these, `load_dataset` follows the default branch and silently
# returns whatever the maintainer has pushed since. The seeds below make the
# sampling deterministic, but determinism over a moving corpus is worthless:
# the item files would change, their hashes in MANIFEST.sha256 would stop
# matching, and there would be no way to tell a corpus update apart from a
# bug in this file. openai/gsm8k, to take the obvious case, was modified in
# March 2026, well after the runs reported in the paper.
#
# To move to a newer revision, change the SHA here, rebuild, re-run the
# audits and regenerate the manifest. Do not edit one without the others.
REVISIONS = {
    "monology/pile-uncopyrighted": "3be90335b66f24456a5d6659d9c8d208c0357119",
    "swj0419/WikiMIA": "a89ab76d88f704e9bc5870ac39cc9d458a2a70ac",
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
}


def write(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {path}  ({len(rows)} items)")


def _check(suspect, reference):
    if len(reference) < 2 * len(suspect):
        print(f"  WARNING: reference is {len(reference)/max(len(suspect),1):.1f}x "
              f"the suspect set; the protocol wants 2x. Reduce --n.",
              file=sys.stderr)
    for r in suspect[:50] + reference[:50]:
        if r.get("answer") and r["answer"].strip() and \
                r["answer"].strip() in r["prefix"]:
            raise SystemExit("the gold answer appears inside a prefix; "
                             "the prefix rule is violated")


MIN_PER_SIDE = 250


def _need_datasets():
    """Import datasets, and defuse its torch integration.

    `datasets` calls torch.share_memory_() when constructing a streaming
    dataset, which fails on macOS whenever torch_shm_manager is not
    executable:

        RuntimeError: torch_shm_manager ... execl failed: Permission denied

    The shared-memory path is only used for persistent DataLoader workers,
    which nothing here uses, so we turn it off rather than requiring the user
    to chmod a file inside their site-packages. If you would rather fix it at
    the source, the one-liner is

        chmod +x "$(python3 -c 'import torch,os;
                    print(os.path.dirname(torch.__file__))')/bin/torch_shm_manager"
    """
    try:
        import datasets
    except ImportError:
        raise SystemExit("this builder needs `pip install datasets`")
    try:
        datasets.config.TORCH_AVAILABLE = False
    except Exception:
        pass
    return datasets


def _need_zstd():
    """The Pile ships as .jsonl.zst; datasets needs a zstd codec to read it."""
    import importlib.util
    if importlib.util.find_spec("zstandard") is None:
        raise SystemExit(
            "the Pile shards are zstd-compressed and no zstd codec is "
            "installed.\n  fix: pip install zstandard")


def _guard(suspect, reference, name):
    """Refuse to write an item set too small for the protocol to be valid.

    Phase 0c measured calibration degrading below a few hundred items per
    side. Writing 55 items and auditing them anyway produces a number that
    looks like a result and is not one.
    """
    n = min(len(suspect), len(reference) // 2)
    if n < MIN_PER_SIDE:
        raise SystemExit(
            f"{name}: only {n} items per side after balancing "
            f"(need >= {MIN_PER_SIDE}). The protocol needs |S| suspect and "
            f"2|S| reference items, and calibration degrades below a few "
            f"hundred per side. Lower --n will not help; this source does not "
            f"have enough items. Use a longer/pooled split or a different "
            f"source, or pass --allow-small to write it anyway and treat the "
            f"result as indicative only.")


# --------------------------------------------------------------------------

def build_pile(n, out, subdomain="Wikipedia (en)", max_words=220, seed=0,
               allow_small=False):
    _need_datasets()
    _need_zstd()
    from datasets import load_dataset
    rng = random.Random(seed)

    # The uncopyrighted mirror exposes only a `train` split through the
    # dataset script, with the held-out partitions as separate files, so the
    # split has to be named explicitly. Several layouts exist in the wild;
    # try them in order rather than assuming one.
    SPECS = [
        {"train": "train/00.jsonl.zst", "validation": "val.jsonl.zst"},
        {"train": "train/00.jsonl.zst", "validation": "validation.jsonl.zst"},
        {"train": "data/train-00000-of-00987.jsonl.zst",
         "validation": "data/val.jsonl.zst"},
    ]

    def _open(split):
        last = None
        for spec in SPECS:
            try:
                return load_dataset(
                    "monology/pile-uncopyrighted", data_files=spec,
                    split=split, streaming=True,
                    revision=REVISIONS["monology/pile-uncopyrighted"])
            except Exception as e:
                last = e
        try:
            return load_dataset(
                "monology/pile-uncopyrighted", split=split, streaming=True,
                revision=REVISIONS["monology/pile-uncopyrighted"])
        except Exception as e:
            last = e
        raise SystemExit(
            f"could not open the Pile '{split}' split. Last error:\n  {last}\n\n"
            + ("Compression type zstd not supported means the codec is "
               "missing: pip install zstandard\n\n"
               if "zstd" in str(last) else "") +
            "The Pile's held-out partitions are separate files on the Hub and "
            "the layout changes; check the repo's file list and add the right "
            "path to SPECS in build_itemsets.py. Without a held-out split "
            "there is no reference set, and a train/train split would measure "
            "nothing.")

    def take(split, k):
        ds = _open(split)
        rows = []
        for ex in ds:
            meta = ex.get("meta") or {}
            if subdomain and meta.get("pile_set_name") != subdomain:
                continue
            words = ex["text"].split()
            if len(words) < 60:
                continue
            rows.append({"prefix": " ".join(words[:max_words]), "answer": "",
                         "source": subdomain})
            if len(rows) >= k:
                break
        return rows

    suspect = take("train", n)
    reference = take("validation", 2 * n)
    if len(reference) < 2 * n:
        print("  note: validation split exhausted; reducing the suspect set")
        suspect = suspect[: len(reference) // 2]
    rng.shuffle(suspect); rng.shuffle(reference)
    _check(suspect, reference)
    if not allow_small:
        _guard(suspect, reference, f"pile/{subdomain}")
    write(f"{out}/pile_{subdomain.split()[0].lower()}_suspect.jsonl", suspect)
    write(f"{out}/pile_{subdomain.split()[0].lower()}_reference.jsonl", reference)


def build_gsm(n, out, seed=0, allow_small=False):
    _need_datasets()
    from datasets import load_dataset
    rng = random.Random(seed)

    gsm8k = None
    for name in ("openai/gsm8k", "gsm8k"):
        try:
            gsm8k = load_dataset(name, "main", split="test",
                                 revision=REVISIONS.get("openai/gsm8k"))
            break
        except Exception as e:
            last = e
    if gsm8k is None:
        raise SystemExit(f"could not load GSM8K: {last}")
    suspect = [{"prefix": f"Question: {e['question']}\nAnswer:",
                "answer": e["answer"].split("####")[-1].strip(),
                "source": "gsm8k"} for e in gsm8k]

    reference = []
    for name in ("Scale-AI/gsm1k", "scale-ai/gsm1k"):
        try:
            d = load_dataset(name, split="test")
            reference = [{"prefix": f"Question: {e['question']}\nAnswer:",
                          "answer": str(e.get("answer", "")).split("####")[-1].strip(),
                          "source": "gsm1k"} for e in d]
            break
        except Exception:
            continue
    if not reference:
        raise SystemExit(
            "GSM1k could not be loaded. It is released under access "
            "conditions; see Zhang et al. (2024). Without it there is no "
            "exchangeable reference set for GSM8K and the audit cannot be run "
            "as specified. Do not substitute a temporal split.")

    rng.shuffle(suspect); rng.shuffle(reference)
    k = min(n, len(reference) // 2)
    suspect, reference = suspect[:k], reference[: 2 * k]
    _check(suspect, reference)
    if not allow_small:
        _guard(suspect, reference, "gsm8k/gsm1k")
    write(f"{out}/gsm8k_suspect.jsonl", suspect)
    write(f"{out}/gsm1k_reference.jsonl", reference)


# Oren et al.'s injected files, fetched from their release repository.
CONTAM_REPO = ("https://raw.githubusercontent.com/tatsu-lab/"
               "test_set_contamination/main/detection_challenge_benchmarks")

# Which HF split supplies the reference items. The injected file is a whole
# test/validation set, so the reference has to come from the same benchmark's
# training split: same annotation pipeline, same format, not injected.
#
# This is construction E3, not E1. Oren et al. inject entire test sets rather
# than a random half of a pool, so no exactly-exchangeable withheld arm exists
# in their release. Membership ground truth is still exact, which is the point
# of using these checkpoints; exchangeability is approximate and BA_nuisance
# measures it rather than assuming it. Benchmarks whose training split cannot
# supply 2|S| items are excluded, which rules out the MMLU subsets despite
# their being the most heavily duplicated.
CONTAM_SETS = {
    "piqa": dict(file="piqa.jsonl", hf=("ybisk/piqa", None, "train"),
                 dup=50, q="goal", a="sol1"),
    "mnli": dict(file="mnli.jsonl", hf=("nyu-mll/glue", "mnli", "train"),
                 dup=10, q="premise", a="hypothesis"),
}


def build_contam(n, out, seed=0, allow_small=False, which="piqa"):
    """Suspect = Oren et al.'s injected items. Reference = same benchmark, untouched."""
    _need_datasets()
    import json as _json
    import urllib.request
    from datasets import load_dataset
    if which not in CONTAM_SETS:
        raise SystemExit(f"--contam-set must be one of {sorted(CONTAM_SETS)}")
    spec = CONTAM_SETS[which]
    rng = random.Random(seed)

    url = f"{CONTAM_REPO}/{spec['file']}"
    print(f"  fetching injected items: {url}")
    try:
        raw = urllib.request.urlopen(url, timeout=60).read().decode()
    except Exception as e:
        raise SystemExit(
            f"could not fetch {url}\n  {e}\n\n"
            "These are the files Oren et al. injected during training, and the "
            "audit is meaningless without exactly them. Do not substitute the "
            "public test set: it is shuffled relative to the injected copy. "
            "Clone github.com/tatsu-lab/test_set_contamination and point "
            "CONTAM_REPO at the local path if the raw URL has moved.")
    inj = [_json.loads(l) for l in raw.splitlines() if l.strip()]
    print(f"  {len(inj)} injected items, duplication {spec['dup']}x")

    def fmt(e, src):
        q, a = spec["q"], spec["a"]
        return {"prefix": f"{e[q]}", "answer": str(e.get(a, "")), "source": src}

    suspect = [fmt(e, f"{which}_injected") for e in inj if spec["q"] in e]
    if not suspect:
        raise SystemExit(
            f"injected file has no '{spec['q']}' field. Keys seen: "
            f"{sorted(inj[0])[:8]}. Fix CONTAM_SETS[{which!r}] rather than "
            "guessing, a mismatched field silently builds a nonsense prefix.")

    path, cfg, split = spec["hf"]
    ds = load_dataset(path, cfg, split=split) if cfg else load_dataset(path, split=split)
    reference = [fmt(e, f"{which}_heldout") for e in ds]
    rng.shuffle(suspect); rng.shuffle(reference)

    k = min(n, len(suspect), len(reference) // 2)
    suspect, reference = suspect[:k], reference[: 2 * k]
    _check(suspect, reference)
    if not allow_small:
        _guard(suspect, reference, f"contam/{which}")
    print(f"  NOTE: construction E3, not E1. Membership is exact; "
          f"exchangeability is approximate. Read BA_nuisance before the verdict.")
    write(f"{out}/contam_{which}_suspect.jsonl", suspect)
    write(f"{out}/contam_{which}_reference.jsonl", reference)


def build_wikimia(n, out, seed=0, lengths=(32, 64, 128, 256),
                  allow_small=False):
    """Pool every length split.

    A single split is far too small: length-128 has 250 rows in total, which
    after the 2:1 reference requirement leaves about 55 items per side. Pooled
    across lengths there are enough, at the cost of a wider length
    distribution, which is exactly what the length-matching control and the
    surface key are there to absorb.
    """
    _need_datasets()
    from datasets import load_dataset
    rng = random.Random(seed)
    mem, non = [], []
    for L in lengths:
        try:
            ds = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{L}",
                              revision=REVISIONS["swj0419/WikiMIA"])
        except Exception as e:
            print(f"  skipped length {L}: {e}")
            continue
        for e in ds:
            row = {"prefix": e["input"], "answer": "", "length_bucket": L}
            (mem if e["label"] == 1 else non).append(
                {**row, "source": f"wikimia_{'member' if e['label'] else 'nonmember'}"})
    rng.shuffle(mem); rng.shuffle(non)
    k = min(n, len(mem), len(non) // 2)
    suspect, reference = mem[:k], non[: 2 * k]
    if not allow_small:
        _guard(suspect, reference, "wikimia")
    write(f"{out}/wikimia_suspect.jsonl", suspect)
    write(f"{out}/wikimia_reference.jsonl", reference)
    print("  REMINDER: this is a temporal split. Report BA_nuisance for it; "
          "do not report a contamination verdict from it.")


BUILDERS = {"pile": build_pile, "gsm8k_gsm1k": build_gsm,
            "contam": build_contam, "wikimia": build_wikimia}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SETS))
    ap.add_argument("--n", type=int, default=500,
                    help="suspect items; twice as many reference items")
    ap.add_argument("--out", default=s(DATA))
    ap.add_argument("--subdomain", default="Wikipedia (en)")
    ap.add_argument("--contam-set", default="piqa",
                    choices=sorted(CONTAM_SETS),
                    help="which injected benchmark to audit (--set contam)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--allow-small", action="store_true",
                    help="write item sets below the size the protocol needs; "
                         "results are indicative only")
    a = ap.parse_args()
    if a.list or not a.set:
        for k, v in SETS.items():
            print(f"{k:14s} {v}")
        return
    print(f"building {a.set}")
    if a.set == "pile":
        build_pile(a.n, a.out, a.subdomain, seed=a.seed,
                   allow_small=a.allow_small)
    elif a.set == "contam":
        build_contam(a.n, a.out, seed=a.seed, allow_small=a.allow_small,
                     which=a.contam_set)
    else:
        BUILDERS[a.set](a.n, a.out, seed=a.seed, allow_small=a.allow_small)


if __name__ == "__main__":
    main()
