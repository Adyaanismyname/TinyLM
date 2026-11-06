"""
Task 1a: which test stories overlap the training data, and which of them did
each trained model actually encounter?

  python -B camera_ready/contamination.py

No model is needed: the overlap is found by hashing token sequences, and
"encountered" is reconstructed exactly from each run's seeded batch order
(common.trained_windows mirrors train.train_model + data.EpochBatcher; the
reconstruction is checked against the real EpochBatcher below).

Writes camera_ready/out/contamination_<set>.npz (per-story flags/exposures)
and camera_ready/out/contamination_summary.json.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (BS, OUT, ROOT, body_hashes, ids, mask, ngram_hashes, save_json,  # noqa: E402
                    story_table, target_coverage, trained_windows, _pow_table, _B, _BINV)

sys.path.insert(0, ROOT)
NGRAM_NS = (50, 100)

# Run configurations, read from code/JSON (asserted against the JSONs below).
E1_POOLS = [2_000_000, 8_000_000, 32_000_000, 100_000_000]
E1 = dict(batch_size=32, max_steps=6103, eval_interval=6103 // 20, eval_batches=10, seeds=[0, 1, 2])
E2 = dict(batch_size=32, max_steps=12207, eval_interval=12207 // 20, eval_batches=10, seeds=[0, 1, 2])
E3 = dict(batch_size=32, max_steps=1220, eval_interval=1220 // 10, eval_batches=10, seeds=[0, 1, 2, 3, 4])
EVAL_LIMITS = {"e12_tinystories_test": ("tinystories_test", 2_000_000),
               "e3_tinystories_test": ("tinystories_test", 1_000_000),
               "e3_instruct_test": ("instruct_test", 1_000_000)}


def log(m):
    print(f"[contamination] {m}", flush=True)


def check_configs():
    import json
    import glob
    for exp, cfg in (("e1", E1), ("e2", E2), ("e3", E3)):
        for p in glob.glob(os.path.join(ROOT, "results", "runs", exp, "*.json")):
            c = json.load(open(p))["config"]
            if c.get("max_steps", 0) == 0:
                continue  # frozen baseline
            assert c["max_steps"] == cfg["max_steps"] and c["batch_size"] == cfg["batch_size"], (p, c)
    log("run configs match the JSONs (steps, batch size)")


def validate_reconstruction():
    """Compare common.trained_windows with the real data.EpochBatcher driven
    in train_model's order, on fake data whose tokens are their positions."""
    try:
        from data import EpochBatcher, TokenStream
    except Exception as exc:  # torch unavailable
        log(f"skipping EpochBatcher cross-check ({exc})")
        return False

    class Positions:
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, s):
            return np.arange(s.start, min(s.stop, self.n), dtype=np.int64)

    cases = [("E1 2M pool seed1", (2_000_000 - 1) // BS * BS + 1, E1, 1, 400),
             ("E2 seed0", 904_470_724, E2, 0, 700),
             ("E3 seed2", 153_457_182, E3, 2, 300)]
    for label, n_tokens, cfg, seed, steps in cases:
        stream = TokenStream(Positions(n_tokens), BS)
        batcher = EpochBatcher(stream, cfg["batch_size"], seed=seed, device="cpu")
        real = []
        for step in range(1, steps + 1):
            x, _ = batcher.next_batch()
            real.append(x[:, 0].numpy() // BS)
            if step % cfg["eval_interval"] == 0 or step == 1:
                for _ in range(cfg["eval_batches"]):
                    batcher.next_batch()
        real = np.concatenate(real)
        mine = trained_windows(len(stream), cfg["batch_size"], seed, steps, cfg["eval_interval"], cfg["eval_batches"])
        assert np.array_equal(real, mine), label
        log(f"reconstruction == real EpochBatcher for {label} ({steps} steps)")
    return True


def ngram_flags(test_sets, train_names):
    """For each test story body, fraction of its n-grams that occur anywhere
    in the training streams. Returns {n: {set_name: frac array}}."""
    out = {}
    for n in NGRAM_NS:
        t0 = time.time()
        grams, owner = [], {}
        for sname, (a, tab) in test_sets.items():
            hs, ow = [], []
            for i, (s, e) in enumerate(zip(tab["body0"], tab["eos"])):
                h = ngram_hashes(a[s:e], n)
                hs.append(h)
                ow.append(np.full(len(h), i, dtype=np.int64))
            owner[sname] = (np.concatenate(hs), np.concatenate(ow))
            grams.append(owner[sname][0])
        allg = np.unique(np.concatenate(grams))
        found = np.zeros(len(allg), dtype=bool)
        chunk = 20_000_000
        powB, powBinv = _pow_table(_B, chunk + n), _pow_table(_BINV, chunk + n)
        for tname in train_names:
            a = np.asarray(ids(tname))
            for c0 in range(0, len(a) - n + 1, chunk):
                seg = a[c0:min(len(a), c0 + chunk + n - 1)]
                h = ngram_hashes(seg, n, powB, powBinv)
                k = np.searchsorted(allg, h)
                k[k == len(allg)] = 0
                hit = allg[k] == h
                found[k[hit]] = True
        out[n] = {}
        for sname, (h, ow) in owner.items():
            f = found[np.searchsorted(allg, h)]
            num = np.bincount(ow, weights=f.astype(np.float64), minlength=len(test_sets[sname][1]["eos"]))
            den = np.bincount(ow, minlength=len(test_sets[sname][1]["eos"]))
            out[n][sname] = np.where(den > 0, num / np.maximum(den, 1), 0.0)
        log(f"{n}-gram scan done ({time.time() - t0:.0f}s)")
    return out


def main():
    check_configs()
    validated = validate_reconstruction()

    t0 = time.time()
    tabs = {name: story_table(name) for name in ("tinystories_train", "instruct_train", "tinystories_test", "instruct_test")}
    arr = {name: np.asarray(ids(name)) for name in tabs}
    hashes = {name: body_hashes(arr[name], tabs[name]["body0"], tabs[name]["eos"]) for name in tabs}
    log(f"story tables + body hashes ({time.time() - t0:.0f}s)")

    # index of training copies: hash -> list of (stream, body0, eos)
    copies = {}
    test_union = np.union1d(hashes["tinystories_test"], hashes["instruct_test"])
    for tname in ("tinystories_train", "instruct_train"):
        h = hashes[tname]
        for k in np.flatnonzero(np.isin(h, test_union)):
            copies.setdefault(int(h[k]), []).append((tname, int(tabs[tname]["body0"][k]), int(tabs[tname]["eos"][k])))

    ng = ngram_flags({t: (arr[t], tabs[t]) for t in ("tinystories_test", "instruct_test")},
                     ["tinystories_train", "instruct_train"])

    # per-run window training counts
    nw = {t: (len(arr[t]) - 1) // BS for t in ("tinystories_train", "instruct_train")}
    counts = {}
    for s in E2["seeds"]:
        w = trained_windows(nw["tinystories_train"], E2["batch_size"], s, E2["max_steps"], E2["eval_interval"], E2["eval_batches"])
        counts[("e2", s)] = np.bincount(w, minlength=nw["tinystories_train"])
    for pool in E1_POOLS:
        n_pool = (pool - 1) // BS
        for s in E1["seeds"]:
            w = trained_windows(n_pool, E1["batch_size"], s, E1["max_steps"], E1["eval_interval"], E1["eval_batches"])
            c = np.zeros(nw["tinystories_train"], dtype=np.int64)
            c[:n_pool] = np.bincount(w, minlength=n_pool)
            counts[("e1", pool, s)] = c
    for s in E3["seeds"]:
        w = trained_windows(nw["instruct_train"], E3["batch_size"], s, E3["max_steps"], E3["eval_interval"], E3["eval_batches"])
        counts[("e3", s)] = np.bincount(w, minlength=nw["instruct_train"])
    log("reconstructed trained windows for all runs")

    summary = {"reconstruction_validated_against_EpochBatcher": validated,
               "ngram_ns": list(NGRAM_NS), "sets": {}}

    for test in ("tinystories_test", "instruct_test"):
        tab, h = tabs[test], hashes[test]
        L = tab["eos"] - tab["body0"]
        exact_ts = np.isin(h, hashes["tinystories_train"])
        exact_it = np.isin(h, hashes["instruct_train"])
        n_copies_ts = np.array([sum(1 for c in copies.get(int(x), []) if c[0] == "tinystories_train") for x in h])

        def exposure(stream, cnt):
            ex = np.zeros(len(h))
            for i, x in enumerate(h):
                cs = [c for c in copies.get(int(x), []) if c[0] == stream]
                if cs and L[i] > 0:
                    ex[i] = sum(target_coverage(cnt, a, b) for _, a, b in cs) / L[i]
            return ex

        expo = {}
        for s in E2["seeds"]:
            expo[f"e2_seed{s}"] = exposure("tinystories_train", counts[("e2", s)])
        if test == "tinystories_test":
            for pool in E1_POOLS:
                for s in E1["seeds"]:
                    expo[f"e1_pool{pool}_seed{s}"] = exposure("tinystories_train", counts[("e1", pool, s)])
        for s in E3["seeds"]:
            expo[f"e3ft_seed{s}"] = exposure("instruct_train", counts[("e3", s)])

        np.savez_compressed(os.path.join(OUT, f"contamination_{test}.npz"),
                            first=tab["first"], eos=tab["eos"], body0=tab["body0"], hdr_end=tab["hdr_end"],
                            exact_tinystories_train=exact_ts, exact_instruct_train=exact_it,
                            n_copies_tinystories_train=n_copies_ts,
                            **{f"ngram{n}_frac": ng[n][test] for n in NGRAM_NS},
                            **{f"exposure_{k}": v for k, v in expo.items()})

        for key, (tname, limit) in EVAL_LIMITS.items():
            if tname != test:
                continue
            last_target = ((limit - 1) // BS) * BS  # targets are positions 1..last_target
            sel = tab["first"] <= last_target          # stories owning at least one evaluated target
            own = np.minimum(tab["eos"], last_target) - np.maximum(tab["first"], 1) + 1
            if test == "instruct_test":                 # only story tokens (mask==1) are scored
                own = np.minimum(tab["eos"], last_target) - np.maximum(tab["hdr_end"], 1) + 1
            own = np.clip(own, 0, None) * sel
            tok = own.sum()

            def share(flag):
                return {"stories": int((flag & sel).sum()), "stories_pct": 100 * (flag & sel).sum() / sel.sum(),
                        "scored_tokens_pct": 100 * own[flag & sel].sum() / tok}

            exact_any = exact_ts | exact_it
            d = {"stories_evaluated": int(sel.sum()), "scored_tokens": int(tok),
                 "exact_in_tinystories_train": share(exact_ts), "exact_in_instruct_train": share(exact_it),
                 "exact_any": share(exact_any)}
            for n in NGRAM_NS:
                any_gram = ng[n][test] > 0
                d[f"shares_any_{n}gram"] = share(any_gram)
                d[f"exact_or_any_{n}gram"] = share(exact_any | any_gram)
                d[f"clean_no_exact_no_{n}gram"] = share(~(exact_any | any_gram))
            d["exposure"] = {}
            for k, v in expo.items():
                if key == "e12_tinystories_test" and k.startswith("e3ft"):
                    continue
                if key.startswith("e3") and k.startswith("e1"):
                    continue
                d["exposure"][k] = {"any_exposure": share(v > 0), "at_least_one_full_pass": share(v >= 1),
                                    "mean_passes_among_exact_dups": float(v[exact_any & sel].mean()) if (exact_any & sel).any() else 0.0}
            summary["sets"][key] = d

    # Task 3 check: Words-field visibility inside the windows each E3 run actually trained on
    a, m = arr["instruct_train"], np.asarray(mask("instruct_train"))
    tab = tabs["instruct_train"]
    T = [53, 143, 65, 80, 26]  # tokenizer.encode("Words:")
    k = len(T)
    cand = np.ones(len(a) - k + 1, dtype=bool)
    for j, t in enumerate(T):
        cand &= a[j:len(a) - k + 1 + j] == t
    cpos = np.flatnonzero(cand)
    ex_of = np.searchsorted(tab["first"], cpos, side="right") - 1
    keep = cpos < tab["hdr_end"][ex_of]
    words_pos = np.full(len(tab["first"]), -1, dtype=np.int64)
    words_pos[ex_of[keep][::-1]] = cpos[keep][::-1]  # first occurrence wins
    visible = np.zeros(len(a), dtype=bool)  # visible[p]: target p's window contains its example's Words: field
    has = words_pos >= 0
    for e in np.flatnonzero(has):
        lo, hi = tab["hdr_end"][e], tab["eos"][e]
        lim = min(hi, (words_pos[e] // BS + 1) * BS)
        if lim >= lo:
            visible[lo:lim + 1] = True
    sup = m.astype(bool)
    vis_stats = {"examples_with_words_field_pct": 100 * has.mean(),
                 "all_windows_supervised_with_words_visible_pct": 100 * (visible & sup).sum() / sup.sum()}
    nwin = nw["instruct_train"]
    for s in E3["seeds"]:
        c = counts[("e3", s)]
        per_win_sup = np.add.reduceat(sup[1:1 + nwin * BS].astype(np.int64), np.arange(0, nwin * BS, BS))
        per_win_vis = np.add.reduceat((sup & visible)[1:1 + nwin * BS].astype(np.int64), np.arange(0, nwin * BS, BS))
        vis_stats[f"seed{s}"] = {"supervised_targets_trained": int((c * per_win_sup).sum()),
                                 "with_words_visible_pct": 100 * (c * per_win_vis).sum() / (c * per_win_sup).sum(),
                                 "epochs_of_stream": float(c.sum() / nwin)}
    summary["e3_words_visibility"] = vis_stats
    save_json(summary, "contamination_summary.json")


if __name__ == "__main__":
    main()
