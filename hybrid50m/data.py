"""Data stage (needs no GPU): download + tokenize the training mix and held-out sets, pick the student vocabulary,
tokenize the instruction data, download the benchmark sets. Everything lands in C.DATA_DIR as plain files."""
import itertools
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from core import PROMPT_TEMPLATE, hms

SMOLLM = "datasets/HuggingFaceTB/smollm-corpus"
COSMO_FILES, WEB_FILES = 104, 234
COSMO_VAL_REF = dict(n_files=16, ref_rounds=44, tokens_per_doc=700)   # the 10M run's validation recipe, kept so val text stays the same


def paths(data_dir):
    p = lambda n: os.path.join(data_dir, n)
    return dict(train=p("train.bin"), cosmo_val=p("cosmo_val.bin"), web_val=p("web_val.bin"), wiki_val=p("wiki_val.bin"),
                counts=p("token_counts.npy"), meta=p("data_meta.json"), sft=p("sft_raw.npz"), bench=p("bench.json"))


def retry(fn, tries=8):
    # the Hub rate-limits parallel readers (HTTP 429): back off and try again
    for k in range(tries):
        try:
            return fn()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def open_files(subdir, total, n_files):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    idx = np.linspace(0, total - 1, n_files).astype(int)
    path = lambda i: f"{SMOLLM}/{subdir}/train-{i:05d}-of-{total:05d}.parquet"
    with ThreadPoolExecutor(n_files) as ex:
        pfs = list(ex.map(lambda i: retry(lambda: pq.ParquetFile(fs.open(path(i), "rb"))), idx))
    return idx.tolist(), pfs


def read_group(pf, gi, columns):
    # one unreadable row-group costs nothing (the files are shuffled): skip it instead of losing a half-finished tokenization
    try:
        return retry(lambda: pf.read_row_group(gi, columns=columns))
    except Exception as e:
        print(f"skipping row-group {gi} ({type(e).__name__}: {e})")
        return None


def read_cosmo(pf, gi):
    t = read_group(pf, gi, ["text"])
    return t.column("text").to_pylist() if t is not None else []


_WEB_NESTED_OK = [True]


def read_web(pf, gi, min_score, min_tokens=100):
    # FineWeb-Edu-dedup rows carry the classifier's educational score: keep only the better-than-`min_score` ones
    if _WEB_NESTED_OK[0]:
        t = read_group(pf, gi, ["text", "metadata.score", "metadata.token_count"])
        if t is not None:
            try:
                meta = t.column("metadata").to_pylist()
                return [x for x, m in zip(t.column("text").to_pylist(), meta)
                        if m["score"] >= min_score and m["token_count"] >= min_tokens]
            except Exception as e:
                print(f"metadata filter unavailable ({type(e).__name__}); keeping unfiltered text")
                _WEB_NESTED_OK[0] = False
    t = read_group(pf, gi, ["text"])
    return t.column("text").to_pylist() if t is not None else []


def rounds_iter(pfs, groups, read_fn):
    # one row-group per file per round, read by one thread per file; the next round downloads while this one is tokenized
    nf, rounds = len(pfs), min(len(g) for g in groups)
    with ThreadPoolExecutor(nf) as pool:
        submit = lambda r: [pool.submit(read_fn, pfs[i], groups[i][r]) for i in range(nf)]
        cur = submit(0)
        for r in range(rounds):
            nxt = submit(r + 1) if r + 1 < rounds else None
            yield [t for f in cur for t in f.result()]
            cur = nxt


def spread(n_groups, rounds, exclude=()):
    g = np.linspace(0, n_groups - 2, rounds).astype(int)
    return [int(x) for x in dict.fromkeys(g.tolist()) if x not in exclude]


def tokenize_into(f, texts, tok, eos, counts=None):
    n = nd = 0
    for i in range(0, len(texts), 256):
        enc = tok(texts[i: i + 256])["input_ids"]
        flat = np.concatenate([np.asarray(x + [eos], dtype=np.uint32) for x in enc])
        flat.tofile(f)
        if counts is not None:
            counts += np.bincount(flat, minlength=len(counts))
        n += len(flat); nd += len(enc)
    return n, nd


def wikitext_test():
    # a corpus neither the student nor the teacher's data selection ever saw: true out-of-domain check (approximate detokenization)
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    pf = retry(lambda: pq.ParquetFile(HfFileSystem().open(
        "datasets/Salesforce/wikitext/wikitext-103-raw-v1/test-00000-of-00001.parquet", "rb")))
    s = "".join(retry(lambda: pf.read()).column("text").to_pylist())
    for a, b in ((" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."), (" , ", ", "), (" . ", ". "), (" ; ", "; "), (" : ", ": "),
                 (" 's", "'s"), (" n't", "n't"), ("( ", "("), (" )", ")"), (" \n", "\n")):
        s = s.replace(a, b)
    return re.sub(r"\n{3,}", "\n\n", s)


def prepare_training_data(C, tok, eos, teacher_vocab):
    P = paths(C.DATA_DIR)
    need = [P[k] for k in ("train", "cosmo_val", "web_val", "counts", "meta")]
    if all(os.path.exists(p) for p in need):
        print("token files already on disk:", json.load(open(P["meta"])))
        return
    t0 = time.time()
    counts = np.zeros(teacher_vocab, np.int64)
    meta = dict(tokens={}, docs={})

    def write(name, texts, count=False, f=None):
        own = f is None
        f = f or open(P[name] + ".tmp", "wb")
        n, nd = tokenize_into(f, texts, tok, eos, counts if count else None)
        if own:
            f.close(); os.replace(P[name] + ".tmp", P[name])
        meta["tokens"][name] = meta["tokens"].get(name, 0) + n
        meta["docs"][name] = meta["docs"].get(name, 0) + nd
        return n

    # ---- validation sets (never read by training) ----
    ref = COSMO_VAL_REF
    ref_idx, ref_pfs = open_files("cosmopedia-v2", COSMO_FILES, ref["n_files"])
    val_groups = {fi: min(max(1, (pf.metadata.num_row_groups // ref["ref_rounds"]) // 2), pf.metadata.num_row_groups - 1)
                  for fi, pf in zip(ref_idx, ref_pfs)}
    per_file = int(np.ceil(C.VAL_TOKENS / (ref["n_files"] * ref["tokens_per_doc"])))
    with ThreadPoolExecutor(len(ref_pfs)) as ex:
        texts = [t for ts in ex.map(lambda a: read_cosmo(a[1], val_groups[a[0]])[:per_file], zip(ref_idx, ref_pfs)) for t in ts]
    write("cosmo_val", texts)

    web_idx, web_pfs = open_files("fineweb-edu-dedup", WEB_FILES, C.WEB_N_FILES)
    web_val_groups = {fi: pf.metadata.num_row_groups // 2 for fi, pf in zip(web_idx, web_pfs)}
    n_val_files = min(len(web_pfs), 16)
    per_file = int(np.ceil(C.VAL_TOKENS / (n_val_files * 1150)))
    with ThreadPoolExecutor(n_val_files) as ex:
        texts = [t for ts in ex.map(lambda i: read_web(web_pfs[i], web_val_groups[web_idx[i]], C.WEB_MIN_SCORE)[:per_file],
                                    range(n_val_files)) for t in ts]
    write("web_val", texts)
    try:
        write("wiki_val", [wikitext_test()])
    except Exception as e:
        print(f"wikitext out-of-domain set skipped ({type(e).__name__}: {e})")
    print({k: f"{v:,}" for k, v in meta["tokens"].items()}, "validation tokens")

    # ---- training mix: Cosmopedia first, then quality-filtered FineWeb-Edu; windows are shuffled at training time ----
    train = open(P["train"] + ".tmp", "wb")
    cosmo_idx, cosmo_pfs = open_files("cosmopedia-v2", COSMO_FILES, C.COSMO_N_FILES)
    rounds = int(np.ceil(1.1 * C.COSMO_TOKENS / (len(cosmo_pfs) * 1000 * 700))) + 1
    groups = []
    for fi, pf in zip(cosmo_idx, cosmo_pfs):
        n = pf.metadata.num_row_groups
        groups.append(spread(n, rounds, {val_groups[fi]} if fi in val_groups else ()))
    print(f"Cosmopedia: {len(cosmo_pfs)} files x up to {min(len(g) for g in groups)} row-groups")
    preview = None
    got = 0
    for texts in rounds_iter(cosmo_pfs, groups, read_cosmo):
        preview = preview or texts[0][:300]
        got += write("train_cosmo", texts, count=True, f=train)
        print(f"\r  cosmopedia {got / 1e6:8.1f}M / {C.COSMO_TOKENS / 1e6:.0f}M tokens | {hms(time.time() - t0)}", end="", flush=True)
        if got >= C.COSMO_TOKENS:
            break
    print()

    web_idx2, web_pfs2 = web_idx, web_pfs
    keep_rate, avg = 0.41, 1150     # measured on FineWeb-Edu-dedup: ~41% of documents score >= 3.0
    rounds = int(np.ceil(1.1 * C.WEB_TOKENS / (len(web_pfs2) * 1000 * keep_rate * avg))) + 1
    groups = [spread(pf.metadata.num_row_groups, rounds, {web_val_groups[fi]}) for fi, pf in zip(web_idx2, web_pfs2)]
    print(f"FineWeb-Edu (score >= {C.WEB_MIN_SCORE}): {len(web_pfs2)} files x up to {min(len(g) for g in groups)} row-groups")
    got = 0
    reader = lambda pf, gi: read_web(pf, gi, C.WEB_MIN_SCORE)
    for texts in rounds_iter(web_pfs2, groups, reader):
        got += write("train_web", texts, count=True, f=train)
        print(f"\r  fineweb-edu {got / 1e6:8.1f}M / {C.WEB_TOKENS / 1e6:.0f}M tokens | {hms(time.time() - t0)}", end="", flush=True)
        if got >= C.WEB_TOKENS:
            break
    print()
    train.close()
    os.replace(P["train"] + ".tmp", P["train"])
    np.save(P["counts"], counts)
    meta["seconds"] = round(time.time() - t0)
    json.dump(meta, open(P["meta"], "w"))
    print(f"done in {hms(time.time() - t0)}: " + ", ".join(f"{k} {v:,}" for k, v in meta["tokens"].items()))
    print("--- first Cosmopedia document ---\n" + (preview or "") + "\n---")


# ------------------------------------------------------------------ instruction data (smol-smoltalk)
def first_exchange(messages):
    # (user, assistant) of a conversation's first turn; a leading system message is dropped
    ms = [m for m in messages if m["role"] != "system"]
    if len(ms) >= 2 and ms[0]["role"] == "user" and ms[1]["role"] == "assistant":
        return ms[0]["content"].strip(), ms[1]["content"].strip()
    return None


def prepare_sft(C, tok, eos, teacher_vocab):
    """Tokenize the first exchange of smol-smoltalk conversations into sft_raw.npz (UNK filtering waits for the vocab)."""
    P = paths(C.DATA_DIR)
    if os.path.exists(P["sft"]):
        print("sft_raw.npz already on disk")
        return
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    ids, plen, tlen, seen = [], [], [], 0
    for fi in range(4):
        pf = retry(lambda: pq.ParquetFile(fs.open(f"datasets/HuggingFaceTB/smol-smoltalk/data/train-{fi:05d}-of-00004.parquet", "rb")))
        for gi in range(pf.metadata.num_row_groups):
            t = read_group(pf, gi, ["messages"])
            for msgs in (t.column("messages").to_pylist() if t is not None else []):
                ex = first_exchange(msgs)
                seen += 1
                if ex is None:
                    continue
                p = tok(PROMPT_TEMPLATE.format(query=ex[0]))["input_ids"]
                r = tok(ex[1])["input_ids"] + [eos]
                if len(r) < 2 or len(p) + len(r) > C.SFT_MAX_LEN + 1:
                    continue
                ids.append(np.asarray(p + r, dtype=np.uint32)); plen.append(len(p)); tlen.append(len(p) + len(r))
            if len(ids) >= C.SFT_MAX_EXAMPLES:
                break
        if len(ids) >= C.SFT_MAX_EXAMPLES:
            break
    n = min(len(ids), C.SFT_MAX_EXAMPLES)
    np.savez(P["sft"], ids=np.concatenate(ids[:n]), plen=np.array(plen[:n], np.int32), tlen=np.array(tlen[:n], np.int32))
    print(f"instruction data: kept {n:,} of {seen:,} conversations (first exchange, <= {C.SFT_MAX_LEN} tokens), "
          f"{sum(tlen[:n]) / 1e6:.1f}M tokens")


# ------------------------------------------------------------------ vocabulary
def build_vocab(C, tok, eos, teacher_vocab):
    P = paths(C.DATA_DIR)
    counts = np.load(P["counts"])
    sft = np.load(P["sft"])
    sft_counts = np.bincount(sft["ids"], minlength=teacher_vocab)
    weighted = counts + C.SFT_VOCAB_SHARE * counts.sum() / max(1, sft_counts.sum()) * sft_counts
    keep = np.argsort(-weighted, kind="stable")[: C.VOCAB_SIZE - 1]
    if eos not in keep:
        keep[-1] = eos
    keep = np.sort(keep)
    cov = lambda c: c[keep].sum() / max(1, c.sum())
    print(f"distinct teacher tokens seen: {(weighted > 0).sum():,} | keeping {len(keep):,} + UNK")
    print(f"token coverage  train {cov(counts):.3%} | instruction data {cov(sft_counts):.3%}")
    for name in ("cosmo_val", "web_val", "wiki_val"):
        if os.path.exists(P[name]):
            c = np.bincount(np.fromfile(P[name], dtype=np.uint32), minlength=teacher_vocab)
            print(f"token coverage  {name:9s} {cov(c):.3%}  (UNK rate {1 - cov(c):.2%})")
    json.dump({"teacher": C.TEACHER_NAME, "teacher_vocab_size": int(teacher_vocab), "eos_teacher_id": int(eos),
               "teacher_ids": keep.tolist()}, open(C.VOCAB_PATH, "w"))


# ------------------------------------------------------------------ benchmarks
def prepare_bench(C):
    P = paths(C.DATA_DIR)
    if os.path.exists(P["bench"]):
        print("bench.json already on disk")
        return
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    rng = np.random.RandomState(0)
    load = lambda p: retry(lambda: pq.read_table(fs.open("datasets/" + p, "rb"))).to_pylist()
    pick = lambda rows: [rows[i] for i in rng.permutation(len(rows))[: C.BENCH_N]]
    bench = {}

    def attempt(name, fn):
        try:
            bench[name] = fn()
            print(f"  {name}: {len(bench[name]['items'])} items")
        except Exception as e:
            print(f"  {name} skipped ({type(e).__name__}: {e})")

    def arc():
        items = [dict(ctx=f"Question: {r['question']}\nAnswer:", choices=[" " + t for t in r["choices"]["text"]],
                      label=r["choices"]["label"].index(r["answerKey"]))
                 for r in pick(load("allenai/ai2_arc/ARC-Easy/test-00000-of-00001.parquet"))]
        return dict(type="mc", items=items)

    def hellaswag():
        def pre(t):
            t = t.strip().replace(" [title]", ". ")
            return re.sub(r"\[.*?\]", "", t).replace("  ", " ")
        items = [dict(ctx=pre(r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize()),
                      choices=[" " + pre(e) for e in r["endings"]], label=int(r["label"]))
                 for r in pick(load("Rowan/hellaswag/data/validation-00000-of-00001.parquet"))]
        return dict(type="mc", items=items)

    def sciq():
        items = []
        for r in pick(load("allenai/sciq/data/test-00000-of-00001.parquet")):
            if len(r["support"]) > 1200:
                continue
            ch = [r["distractor1"], r["distractor2"], r["distractor3"], r["correct_answer"]]
            items.append(dict(ctx=f"{r['support'].strip()}\nQuestion: {r['question']}\nAnswer:".strip(),
                              choices=[" " + c for c in ch], label=3))
        return dict(type="mc", items=items)

    def lambada():
        f = retry(lambda: fs.open("datasets/EleutherAI/lambada_openai/data/lambada_test.jsonl", "rb"))
        rows = [json.loads(line)["text"] for line in f.read().decode().splitlines() if line.strip()]
        items = [dict(ctx=t.rsplit(" ", 1)[0], target=" " + t.rsplit(" ", 1)[1]) for t in [rows[i] for i in rng.permutation(len(rows))[: C.BENCH_N]]]
        return dict(type="lm", items=items)

    for name, fn in (("arc_easy", arc), ("hellaswag", hellaswag), ("sciq", sciq), ("lambada", lambada)):
        attempt(name, fn)
    json.dump(bench, open(P["bench"], "w"))
