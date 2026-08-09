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


# Oren et al.'s injected files. Their release ships one zip per trained model
# under detection_challenge_benchmarks/<variant>/benchmarks.zip, not loose
# JSONL, and only two variants carry archives: the small/medium/large models
# share contam-1.4b's injected sets, and dupcount-lower shares
# dupcount-higher's.
CONTAM_REPO = ("https://raw.githubusercontent.com/tatsu-lab/"
               "test_set_contamination/main/detection_challenge_benchmarks")
# The full public benchmark files, which are the population the injected
# subsets were drawn from. The withheld arm is their difference.
POOL_REPO = ("https://raw.githubusercontent.com/tatsu-lab/"
             "test_set_contamination/main/benchmarks")

# Duplication counts, transcribed from their README. These are the reason to
# prefer one variant over another: 50x gives the protocol its best chance of
# firing, 1x is the hardest case and the one their own Table 4 shows every
# published method failing at.
CONTAM_DUP = {
    "contam-1.4b": {"piqa": 1},
    "contam-1.4b-dupcount-higher": {"piqa": 50},
}

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
# q/a are the prefix and answer fields; `pool` is the member of the public
# benchmarks.zip holding the full population the injected file was drawn from;
# `key` is what identifies an item across the two copies.
#
# Only piqa is listed. The public release covers eight benchmarks and the
# injected release covers twelve, and piqa is the one where a heavily
# duplicated injected file (50x) and its full pool are both present, which is
# what the difference construction needs. Adding another set means checking
# both zips carry it, not just guessing a member name.
CONTAM_SETS = {
    "piqa": dict(q="goal", a="sol1", key="goal", pool="piqa/tests.jsonl",
                 fields=["goal", "sol1", "sol2"]),
}

# How much of each record the probe reads.
#   goal  the question only, about 15 tokens. What the first Phase 3 run used.
#   full  goal and both solutions, joined by newlines. Closest to what the
#         model was trained on while keeping both arms formatted identically.
#   raw   the verbatim JSONL line.
#
# `raw` is the most faithful reproduction if injection was verbatim, but the
# injected file and the public pool are separate files and may not serialise
# identically. Any difference in key order or spacing would be a first-order
# surface cue, and the probe would learn file formatting rather than
# familiarity. BA_nuisance is exactly the instrument that catches this, so the
# mode is available, but read that number before anything else.
CONTAM_PREFIX_MODES = ("full", "goal", "raw")


def _zip_member(zf, which):
    """Find the archive member holding `which`, or explain what is there."""
    names = [n for n in zf.namelist() if not n.endswith("/")]
    for n in names:
        if os.path.basename(n).lower().split(".")[0] == which.lower():
            return n
    hits = [n for n in names if which.lower() in n.lower()]
    if len(hits) == 1:
        return hits[0]
    raise SystemExit(
        f"no member for {which!r} in the archive. Members:\n  "
        + "\n  ".join(sorted(names)[:40])
        + f"\n\nPick one and add it to CONTAM_SETS[{which!r}]. Do not guess "
          "from the name alone: the audit is only meaningful against the "
          "exact file that was injected.")


def _fetch_bytes(url):
    """Read a URL as bytes, using certifi's CA bundle."""
    import ssl
    import urllib.request
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    return urllib.request.urlopen(url, timeout=120, context=ctx).read()


def _fetch_text(url):
    """Read a URL, using certifi's CA bundle.

    Python installed from python.org ships no CA bundle and does not read the
    macOS keychain, so urllib fails with CERTIFICATE_VERIFY_FAILED on any https
    URL until either `Install Certificates.command` is run or a bundle is
    supplied. certifi is already present as a transitive dependency of
    `datasets`. Verification stays on: this file selects which items an audit
    calls contaminated, so an unauthenticated fetch is not an acceptable
    shortcut.
    """
    import ssl
    import urllib.request
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    return urllib.request.urlopen(url, timeout=60, context=ctx).read().decode()


def build_contam(n, out, seed=0, allow_small=False, which="piqa", local=None,
                 variant="contam-1.4b-dupcount-higher", mode="full"):
    """Suspect = Oren et al.'s injected items. Reference = same benchmark, untouched."""
    # No `datasets` dependency: both arms come out of Oren et al.'s own zips.
    import io
    import json as _json
    import zipfile
    if which not in CONTAM_SETS:
        raise SystemExit(f"--contam-set must be one of {sorted(CONTAM_SETS)}")
    if variant not in CONTAM_DUP:
        raise SystemExit(f"--contam-model must be one of {sorted(CONTAM_DUP)}")
    spec = CONTAM_SETS[which]
    dup = CONTAM_DUP[variant].get(which)
    rng = random.Random(seed)

    rel = f"{variant}/benchmarks.zip"
    local = local or os.environ.get("CONTAM_DIR")
    if local:
        p = os.path.join(os.path.expanduser(local), rel)
        print(f"  reading {p}")
        if not os.path.exists(p):
            raise SystemExit(
                f"{p} does not exist. --contam-dir should be the "
                "detection_challenge_benchmarks directory of a clone of "
                "github.com/tatsu-lab/test_set_contamination, which contains "
                f"one subdirectory per model. Expected {rel} inside it.")
        blob = open(p, "rb").read()
    else:
        url = f"{CONTAM_REPO}/{rel}"
        print(f"  fetching {url}")
        try:
            blob = _fetch_bytes(url)
        except Exception as e:
            hint = ""
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                hint = ("\n  This is the macOS no-CA-bundle problem:\n"
                        "    pip install certifi\n")
            raise SystemExit(
                f"could not fetch {url}\n  {e}\n{hint}\n"
                "Cloning is more robust anyway:\n"
                "    git clone https://github.com/tatsu-lab/test_set_contamination\n"
                "    python3 src/build_itemsets.py --set contam "
                f"--contam-set {which} --n {n} \\\n"
                "        --contam-dir test_set_contamination/detection_challenge_benchmarks")

    zf = zipfile.ZipFile(io.BytesIO(blob))
    member = spec.get("member") or _zip_member(zf, which)
    raw = zf.read(member).decode("utf-8", "replace")
    print(f"  {member}, duplication {dup}x in {variant}")

    inj = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = _json.loads(line)
        except _json.JSONDecodeError:
            rec = {"text": line}             # some members are plain .txt
        rec["_raw"] = line
        inj.append(rec)
    print(f"  {len(inj)} injected items")

    # What the probe reads. The model was trained on the whole record, so a
    # prefix of the goal alone asks whether ~15 tokens of a memorised item
    # carry a familiarity direction, which is a much harder question than the
    # one Phase 3 is meant to answer. `full` reconstructs the record.
    #
    # PIQA ships no label, so there is no correct answer to leak and the
    # prefix rule is not in tension here. That is specific to this benchmark:
    # do not copy this to a set where `answer` is the thing being predicted.
    fields = spec.get("fields", [spec["q"]])

    def fmt(e, src):
        if mode == "goal":
            prefix = str(e[spec["q"]])
        elif mode == "raw":
            prefix = e.get("_raw", "")
        else:
            prefix = "\n".join(str(e[f]) for f in fields if f in e)
        return {"prefix": prefix, "answer": str(e.get(spec["a"], "")),
                "source": src}

    suspect = [fmt(e, f"{which}_injected") for e in inj if spec["q"] in e]
    if not suspect:
        sample = _json.dumps(inj[0])[:400] if inj else "(empty)"
        raise SystemExit(
            f"no '{spec['q']}' field in {member}.\n"
            f"  keys: {sorted(inj[0]) if inj else []}\n"
            f"  first record: {sample}\n\n"
            f"Set the right q/a fields in CONTAM_SETS[{which!r}]. Do not "
            "guess: a mismatched field builds a nonsense prefix and the "
            "statistics will look perfectly healthy.")

    # The reference arm is the rest of the same pool. Oren et al. injected a
    # subset of a public benchmark file and shipped the whole file separately,
    # so the withheld items are recoverable by difference. Both arms then come
    # from one pool split before training, which is construction E1: the split
    # is randomised, so exchangeability holds by construction rather than by
    # argument. This is the only arm in the paper where that is true.
    if local:
        # --contam-dir is <clone>/detection_challenge_benchmarks; the pool
        # lives in the sibling <clone>/benchmarks.
        clone = os.path.dirname(os.path.abspath(os.path.expanduser(local)))
        pool_zip = os.path.join(clone, "benchmarks", "benchmarks.zip")
        print(f"  reading {pool_zip}")
        if not os.path.exists(pool_zip):
            raise SystemExit(
                f"{pool_zip} does not exist.\n\n"
                "The withheld arm is recovered by subtracting the injected "
                "items from the full public benchmark file, so both zips are "
                "needed. Point --contam-dir at "
                "<clone>/detection_challenge_benchmarks and keep the clone "
                "intact rather than copying one directory out of it.")
        pool_blob = open(pool_zip, "rb").read()
    else:
        print(f"  fetching {POOL_REPO}/benchmarks.zip")
        pool_blob = _fetch_bytes(f"{POOL_REPO}/benchmarks.zip")
    pz = zipfile.ZipFile(io.BytesIO(pool_blob))
    pool_member = spec.get("pool") or _zip_member(pz, which)
    pool = []
    for l in pz.read(pool_member).decode("utf-8", "replace").splitlines():
        if not l.strip():
            continue
        rec = _json.loads(l)
        rec["_raw"] = l
        pool.append(rec)
    print(f"  pool: {pool_member}, {len(pool)} items")

    key = spec.get("key", spec["q"])
    pool_keys = {str(e[key]) for e in pool if key in e}
    inj_keys = {str(e[key]) for e in inj if key in e}

    # Two different quantities, and conflating them hides the failure that
    # would matter. `found` is how many injected items exist in the public
    # pool, and it is the E1 claim: below 100% the two arms are not one
    # population. `dropped` is how many pool rows were excluded, and it
    # legitimately exceeds len(inj) because PIQA repeats `goal` across items
    # with different solution pairs. Excluding all of them is the conservative
    # direction: it costs a few reference items and guarantees no injected
    # text survives in the withheld arm.
    found = sum(1 for e in inj if key in e and str(e[key]) in pool_keys)
    reference = [fmt(e, f"{which}_withheld") for e in pool
                 if key in e and str(e[key]) not in inj_keys]
    dropped = len(pool) - len(reference)
    print(f"  injected items present in the pool: {found}/{len(inj)}"
          f"  ({len(inj_keys)} distinct keys)")
    print(f"  pool rows excluded: {dropped}, withheld: {len(reference)}")
    if found < 0.98 * len(inj):
        raise SystemExit(
            f"only {found} of {len(inj)} injected items are in the public "
            f"pool, matching on {key!r}.\n\n"
            "The withheld arm is defined as the pool minus the injected "
            "items, so this construction is only E1 if the injected file is "
            "a subset of the pool. It is not. The audit would compare two "
            "populations rather than two halves of one, which is the exact "
            "confound this protocol exists to remove. Check whether the "
            "injected copy was reformatted before injection.")

    rng.shuffle(suspect); rng.shuffle(reference)
    k = min(n, len(suspect), len(reference) // 2)
    suspect, reference = suspect[:k], reference[: 2 * k]
    _check(suspect, reference)
    if not allow_small:
        _guard(suspect, reference, f"contam/{which}")
    toks = sum(len(r["prefix"].split()) for r in suspect) / max(len(suspect), 1)
    print(f"  prefix mode {mode!r}, {toks:.0f} words per item on average")
    print("  construction E1: both arms are the same pool, split at random "
          "before training. Exchangeability is exact.")
    write(f"{out}/contam_{which}_{mode}_suspect.jsonl", suspect)
    write(f"{out}/contam_{which}_{mode}_reference.jsonl", reference)


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
    ap.add_argument("--contam-dir", default=None,
                    help="local detection_challenge_benchmarks/ directory, "
                         "instead of fetching over https")
    ap.add_argument("--contam-model", default="contam-1.4b-dupcount-higher",
                    choices=sorted(CONTAM_DUP),
                    help="which trained variant's injected sets to use; "
                         "dupcount-higher duplicates piqa 50x")
    ap.add_argument("--contam-prefix", default="full",
                    choices=CONTAM_PREFIX_MODES,
                    help="how much of each record the probe reads; "
                         "'full' is the whole record, 'goal' the question only")
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
                     which=a.contam_set, local=a.contam_dir,
                     variant=a.contam_model, mode=a.contam_prefix)
    else:
        BUILDERS[a.set](a.n, a.out, seed=a.seed, allow_small=a.allow_small)


if __name__ == "__main__":
    main()
