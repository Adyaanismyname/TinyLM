"""
Shared helpers for the camera-ready re-analysis (no training happens here).

Everything is reconstructed from the files the original runs already wrote:
data/*.bin (token streams), results/runs/*/*.json (configs + metrics), and
the saved checkpoints. Run every script from the repository root.
"""

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RUNS = os.path.join(ROOT, "results", "runs")
OUT = os.path.join(ROOT, "camera_ready", "out")
BS = 512  # block_size used by every experiment

with open(os.path.join(DATA, "meta.json")) as _f:
    META = json.load(_f)
EOS = META["eos_id"]


def ids(name):
    return np.memmap(os.path.join(DATA, f"{name}.bin"), dtype=np.uint16, mode="r")


def mask(name):
    return np.memmap(os.path.join(DATA, f"{name}_mask.bin"), dtype=np.uint8, mode="r")


def story_table(name):
    """Story boundaries for one stream.

    Returns dict of int64 arrays:
      first  -- first token position owned by the story/example (header
                included for Instruct streams)
      eos    -- position of the story's EOS token (owned by the story)
      body0  -- first token of the story *body* used for identity matching:
                TinyStories: == first; Instruct: header end + 1 (the story is
                encoded as " " + text, so its first token is a lone space that
                a TinyStories copy of the same story does not have)
    A token at position p belongs to the story i with first[i] <= p <= eos[i].
    """
    a = np.asarray(ids(name))
    eos = np.flatnonzero(a == EOS).astype(np.int64)
    first = np.concatenate([[0], eos[:-1] + 1]).astype(np.int64)
    if name.startswith("instruct"):
        m = np.asarray(mask(name))
        hdr_end = (np.flatnonzero(np.diff(m.astype(np.int8)) == 1) + 1).astype(np.int64)
        assert len(hdr_end) == len(eos), (name, len(hdr_end), len(eos))
        body0 = hdr_end + 1
    else:
        hdr_end = first.copy()
        body0 = first.copy()
    return {"first": first, "eos": eos, "hdr_end": hdr_end, "body0": body0}


# ---------------------------------------------------------------------------
# Exact story identity: a hash of the whole token sequence of the story body.
_rng = np.random.default_rng(7)
_MAXL = 16384
_W = _rng.integers(1, 2**63 - 1, size=_MAXL, dtype=np.uint64) | np.uint64(1)


def body_hashes(a, s, e, chunk_tokens=40_000_000):
    """Hash of a[s[i]:e[i]] for every i (position-weighted sum + length)."""
    a = np.asarray(a)
    out = np.zeros(len(s), dtype=np.uint64)
    i = 0
    n = len(s)
    while i < n:
        j = int(np.searchsorted(e, s[i] + chunk_tokens, side="right"))
        j = max(j, i + 1)
        ss, ee = s[i:j], e[i:j]
        L = (ee - ss).astype(np.int64)
        total = int(L.sum())
        rel = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(L) - L, L)
        pos = np.repeat(ss, L) + rel
        vals = (a[pos].astype(np.uint64) + np.uint64(1)) * _W[np.minimum(rel, _MAXL - 1)]
        starts = np.cumsum(L) - L
        h = np.zeros(len(ss), dtype=np.uint64)
        nz = L > 0
        if nz.any():
            h[nz] = np.add.reduceat(vals, starts[nz])
        out[i:j] = h ^ (L.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15))
        i = j
    return out


# ---------------------------------------------------------------------------
# n-gram (shared-span) hashing for near-duplicate detection
_B = np.uint64(0x100000001B3)          # odd -> invertible mod 2^64
_BINV = np.uint64(pow(int(_B), -1, 2**64))


def _pow_table(base, n):
    p = np.empty(n, dtype=np.uint64)
    p[0] = 1
    if n > 1:
        p[1:] = base
        p = np.cumprod(p, dtype=np.uint64)
    return p


def ngram_hashes(tokens, n, powB=None, powBinv=None):
    """Polynomial hash of every length-n window of `tokens` (1D array).
    H[i] = sum_j (t[i+j]+1) * B^(n-1-j)  (mod 2^64)."""
    t = np.asarray(tokens).astype(np.uint64) + np.uint64(1)
    L = len(t)
    if L < n:
        return np.zeros(0, dtype=np.uint64)
    if powBinv is None or len(powBinv) < L:
        powBinv = _pow_table(_BINV, L)
    if powB is None or len(powB) < L:
        powB = _pow_table(_B, L)
    q = np.zeros(L + 1, dtype=np.uint64)
    np.cumsum(t * powBinv[:L], dtype=np.uint64, out=q[1:])
    i = np.arange(L - n + 1)
    return (q[i + n] - q[i]) * powB[i + n - 1]


# ---------------------------------------------------------------------------
# Reconstruct which windows a training run actually trained on.
def trained_windows(num_windows, batch_size, seed, max_steps, eval_interval, eval_batches):
    """Mirrors train.train_model + data.EpochBatcher exactly: one training
    batch per step, and at step 1 and every `eval_interval` steps,
    estimate_loss draws `eval_batches` more batches from the *training*
    batcher (evaluated, not trained on). Epoch k uses
    np.random.default_rng(seed + k).permutation(num_windows); short final
    batch dropped. Returns the trained window indices (with repeats)."""
    nb = num_windows // batch_size
    state = {"epoch": 0, "b": 0, "perm": np.random.default_rng(seed).permutation(num_windows)}

    def next_batch():
        if state["b"] >= nb:
            state["epoch"] += 1
            state["perm"] = np.random.default_rng(seed + state["epoch"]).permutation(num_windows)
            state["b"] = 0
        b = state["b"]
        state["b"] += 1
        return state["perm"][b * batch_size:(b + 1) * batch_size]

    out = np.empty(max_steps * batch_size, dtype=np.int64)
    for step in range(1, max_steps + 1):
        out[(step - 1) * batch_size:step * batch_size] = next_batch()
        if step % eval_interval == 0 or step == 1:
            for _ in range(eval_batches):
                next_batch()
    return out


def target_coverage(window_counts, a, b):
    """Number of target-token trainings that positions [a, b) received,
    given per-window training counts (window w trains targets w*BS+1 ..
    w*BS+BS). Positions are in the training stream's coordinates."""
    if b <= a:
        return 0
    w0 = (a - 1) // BS if a >= 1 else 0
    w1 = (b - 1 - 1) // BS
    tot = 0
    for w in range(max(w0, 0), min(w1, len(window_counts) - 1) + 1):
        lo = max(a, w * BS + 1)
        hi = min(b, w * BS + BS + 1)
        if hi > lo:
            tot += (hi - lo) * int(window_counts[w])
    return tot


def save_json(obj, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"wrote {os.path.relpath(path, ROOT)}")
