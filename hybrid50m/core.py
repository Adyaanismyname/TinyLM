"""Shared pieces for the 50M distillation run: vocab map, teacher, KD loss, token files, evaluation, benchmarks,
checkpoints, export and the dashboard figure. Used by train.py and by the notebook."""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import Config, StudentLM

NEG_LOGIT = -1e4   # stands in for -inf so 0 * (t - s) stays 0 inside the KL
PROMPT_TEMPLATE = "### Question:\n{query}\n\n### Answer:\n"
LN2 = math.log(2)


def hms(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


# ------------------------------------------------------------------ vocabulary
class VocabMap:
    """Student ids 0..K-1 are a subset of the teacher's ids; student id K is UNK (every other teacher token)."""

    def __init__(self, data):
        if not isinstance(data, dict):
            data = json.load(open(data))
        self.raw = data
        self.teacher_ids = torch.tensor(data["teacher_ids"], dtype=torch.long)
        self.unk_id = len(self.teacher_ids)
        self.size = self.unk_id + 1
        lookup = torch.full((data["teacher_vocab_size"],), self.unk_id, dtype=torch.long)
        lookup[self.teacher_ids] = torch.arange(len(self.teacher_ids))
        self.t2s = lookup
        self.eos_id = int(lookup[data["eos_teacher_id"]])
        self._dev = {}
        self.sbytes = None

    def _on(self, device):
        if device not in self._dev:
            self._dev[device] = self.t2s.to(device)
        return self._dev[device]

    def to_student(self, ids):
        if isinstance(ids, torch.Tensor):
            return self._on(ids.device)[ids]
        return self.t2s[torch.tensor(ids, dtype=torch.long)].tolist()

    def to_teacher(self, ids):  # for decoding; UNK has no teacher token and is dropped
        return [int(self.teacher_ids[i]) for i in ids if i != self.unk_id]

    def set_bytes(self, tok_bytes):
        # UTF-8 bytes each student token stands for (UNK: 0). Bits-per-byte is comparable across tokenizers, CE is not.
        b = torch.tensor(np.asarray(tok_bytes), dtype=torch.float32)[self.teacher_ids]
        self.sbytes = torch.cat([b, torch.zeros(1)])


def teacher_token_bytes(tokenizer, teacher_vocab):
    """Byte length of every teacher token. Qwen's tokens are byte-level BPE: one character of the token string = one byte."""
    out = np.zeros(teacher_vocab, np.float32)
    for tok, i in tokenizer.get_vocab().items():
        if i < teacher_vocab:
            out[i] = len(tok)
    for i in tokenizer.added_tokens_decoder:      # <|endoftext|> and friends stand for no text
        if i < teacher_vocab:
            out[i] = 0
    return out


# ------------------------------------------------------------------ teacher + loss
class Teacher:
    """Frozen teacher whose logits are computed only over the student's vocabulary subset (hidden states x K head rows)."""

    def __init__(self, name, vocab, device, dtype):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        model.to(device).eval().requires_grad_(False)
        self.name, self.dtype = name, dtype
        self.backbone = model.get_decoder()
        self.head = model.get_output_embeddings().weight.index_select(0, vocab.teacher_ids.to(device)).contiguous()
        del model

    @torch.no_grad()
    def logits(self, input_ids):
        h = self.backbone(input_ids=input_ids).last_hidden_state
        logits = (h @ self.head.T).float()
        unk = torch.full((*logits.shape[:-1], 1), NEG_LOGIT, device=logits.device)
        return torch.cat([logits, unk], dim=-1)


def kd_loss(student_logits, teacher_logits, targets, temperature=1.0, alpha=0.5):
    """alpha * T^2 * KL(teacher || student) + (1 - alpha) * CE over targets != -100. Returns (total, kl, ce)."""
    V = student_logits.size(-1)
    s = student_logits.float().view(-1, V)
    tgt = targets.reshape(-1)
    n = (tgt != -100).float().sum().clamp(min=1)
    ce = F.cross_entropy(s, tgt, ignore_index=-100, reduction="sum") / n
    if teacher_logits is None or alpha == 0:
        return ce, torch.zeros((), device=ce.device), ce
    t_logp = F.log_softmax(teacher_logits.view(-1, V) / temperature, dim=-1)
    s_logp = F.log_softmax(s / temperature, dim=-1)
    kl = ((t_logp.exp() * (t_logp - s_logp)).sum(-1) * (tgt != -100).float()).sum() / n * temperature ** 2
    return alpha * kl + (1 - alpha) * ce, kl, ce


def split_batch(batch, vocab, device):
    # batch: (B, T+1) teacher ids -> teacher input, student input, student targets (UNK targets ignored)
    batch = batch.to(device, non_blocking=True)
    xt = batch[:, :-1]
    y = vocab.to_student(batch[:, 1:]).clone()
    y[y == vocab.unk_id] = -100
    return xt, vocab.to_student(xt), y


def load_teacher_checked(name, vocab, device, dtype, probe_batches, log=print):
    """fp16 teacher, verified against fp32 on a few batches; falls back to fp32 if fp16 misbehaves."""
    def ce_of(t):
        tot = 0.0
        for b in probe_batches:
            xt, _, y = split_batch(b, vocab, device)
            tot += kd_loss(t.logits(xt), None, y, alpha=0.0)[2].item()
        return tot / len(probe_batches)

    t = Teacher(name, vocab, device, dtype)
    if dtype == torch.float32 or not probe_batches:
        return t
    finite = all(bool(torch.isfinite(t.logits(split_batch(b, vocab, device)[0])[..., :-1]).all()) for b in probe_batches)
    ce_low = ce_of(t)
    ref = Teacher(name, vocab, device, torch.float32)
    ce_ref = ce_of(ref)
    ok = finite and abs(ce_low - ce_ref) < 0.02
    log(f"teacher {str(dtype).replace('torch.', '')} check: CE {ce_low:.4f} vs fp32 {ce_ref:.4f} (finite: {finite}) -> "
        + ("OK" if ok else "MISMATCH, using fp32 teacher"))
    if ok:
        del ref
        return t
    del t
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ref


# ------------------------------------------------------------------ token files
class TokenFile:
    """Flat uint32 token file read through np.memmap as consecutive windows of seq_len + 1 tokens."""

    def __init__(self, path, seq_len):
        self.path, self.seq_len = path, seq_len
        self.n_tokens = os.path.getsize(path) // 4
        self.n_windows = max(0, (self.n_tokens - 1) // seq_len)
        self.mm = None

    def get(self, idx):
        if self.mm is None:
            self.mm = np.memmap(self.path, dtype=np.uint32, mode="r")
        T = self.seq_len
        return torch.from_numpy(np.stack([self.mm[i * T: i * T + T + 1] for i in idx]).astype(np.int64))

    def fixed(self, n_windows):
        # n evenly spaced windows: the same fixed evaluation batch every time, representative of the whole file
        return self.get(np.linspace(0, self.n_windows - 1, min(n_windows, self.n_windows)).astype(int))


class WindowSampler:
    """Infinite shuffled stream of window ids. `take(start, count)` is a pure function of its arguments (epoch permutations
    are seeded), so a resumed run continues with exactly the next windows."""

    def __init__(self, n, seed):
        self.n, self.seed, self._ep, self._perm = n, seed, -1, None

    def take(self, start, count):
        out = []
        while count:
            ep, off = divmod(start, self.n)
            if ep != self._ep:
                self._perm = torch.randperm(self.n, generator=torch.Generator().manual_seed(self.seed + ep)).numpy()
                self._ep = ep
            k = min(count, self.n - off)
            out.append(self._perm[off: off + k])
            start, count = start + k, count - k
        return np.concatenate(out)


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(model, vocab, windows, device, amp, teacher=None, batch_size=8):
    """Held-out metrics over fixed windows. ce: nats per predicted token (UNK targets excluded); bpb: bits per UTF-8 byte of
    those tokens; with a teacher also kl (teacher||student), top-1 agreement and the teacher's own ce / bpb."""
    was_training = model.training
    model.eval()
    nll = byt = n = kl_sum = agree = t_nll = 0.0
    for i in range(0, len(windows), batch_size):
        xt, xs, y = split_batch(windows[i: i + batch_size], vocab, device)
        with amp():
            s = model(xs)
        s = s.float()
        V = s.size(-1)
        valid = y != -100
        yy = y.clamp(min=0)
        s_logp = F.log_softmax(s, dim=-1)
        nll += -(s_logp.gather(-1, yy[..., None]).squeeze(-1) * valid).sum().item()
        byt += (vocab.sbytes.to(device)[yy] * valid).sum().item()
        n += valid.sum().item()
        if teacher is not None:
            t_logp = F.log_softmax(teacher.logits(xt), dim=-1)
            kl_sum += ((t_logp.exp() * (t_logp - s_logp)).sum(-1) * valid).sum().item()
            agree += ((s.argmax(-1) == t_logp.argmax(-1)) & valid).sum().item()
            t_nll += -(t_logp.gather(-1, yy[..., None]).squeeze(-1) * valid).sum().item()
    model.train(was_training)
    n = max(n, 1)
    out = {"ce": nll / n, "ppl": math.exp(min(20, nll / n)), "bpb": nll / LN2 / max(byt, 1)}
    if teacher is not None:
        out.update(kl=kl_sum / n, agree=100 * agree / n, t_ce=t_nll / n, t_bpb=t_nll / LN2 / max(byt, 1))
    return out


# ------------------------------------------------------------------ benchmarks (zero-shot, log-likelihood)
def prep_bench(raw, tokenizer):
    """bench.json (plain text) -> token ids, tokenized once."""
    enc = lambda s: tokenizer(s)["input_ids"]
    out = {}
    for name, task in raw.items():
        items = []
        for it in task["items"]:
            if task["type"] == "mc":
                items.append(dict(ctx=enc(it["ctx"]), conts=[enc(c) for c in it["choices"]],
                                  nbytes=[max(1, len(c.encode())) for c in it["choices"]], label=it["label"]))
            else:
                items.append(dict(ctx=enc(it["ctx"]), conts=[enc(it["target"])]))
        out[name] = dict(type=task["type"], items=items)
    return out


@torch.no_grad()
def _score(model, vocab, seqs, device, amp, max_len, bs=32):
    # seqs: [(teacher ids, n_cont)] -> per sequence (sum log-prob of non-UNK continuation tokens, #scored, all-greedy-correct, has_unk)
    model.eval()
    res = [None] * len(seqs)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
    for k in range(0, len(order), bs):
        rows = order[k: k + bs]
        items = []
        for i in rows:
            ids, c = seqs[i]
            ids = ids[-(max_len + 1):]
            items.append((vocab.to_student(ids), min(c, len(ids) - 1)))
        L = max(len(s) for s, _ in items)
        x = torch.zeros(len(rows), L, dtype=torch.long)
        cmask = torch.zeros(len(rows), L - 1, dtype=torch.bool)
        for r, (s, c) in enumerate(items):
            x[r, : len(s)] = torch.tensor(s)
            cmask[r, len(s) - c - 1: len(s) - 1] = True
        x, cmask = x.to(device), cmask.to(device)
        with amp():
            logits = model(x[:, :-1])
        logp = F.log_softmax(logits.float(), dim=-1)
        tgt = x[:, 1:]
        known = tgt != vocab.unk_id
        lp = logp.gather(-1, tgt[..., None]).squeeze(-1)
        use = cmask & known
        ok = ((logp.argmax(-1) == tgt) | ~cmask).all(1)
        has_unk = (cmask & ~known).any(1)
        for r, i in enumerate(rows):
            res[i] = ((lp[r] * use[r]).sum().item(), int(use[r].sum()), bool(ok[r]), bool(has_unk[r]))
    return res


def run_bench(model, vocab, prepared, device, amp, max_len):
    """Zero-shot scores: acc / acc_norm (norm = by byte length of the choice) for multiple choice, accuracy + perplexity of the last word for LAMBADA."""
    out = {}
    for name, task in prepared.items():
        items = task["items"]
        if task["type"] == "mc":
            seqs = [(it["ctx"] + c, len(c)) for it in items for c in it["conts"]]
            res = _score(model, vocab, seqs, device, amp, max_len)
            hit = hit_n = 0
            pos = 0
            for it in items:
                k = len(it["conts"])
                sc = np.array([r[0] for r in res[pos: pos + k]])
                nb = np.array(it["nbytes"], dtype=float)
                hit += int(sc.argmax() == it["label"])
                hit_n += int((sc / nb).argmax() == it["label"])
                pos += k
            out[f"{name}"] = 100 * hit / len(items)
            out[f"{name}_norm"] = 100 * hit_n / len(items)
        else:
            res = _score(model, vocab, [(it["ctx"] + it["conts"][0], len(it["conts"][0])) for it in items], device, amp, max_len)
            good = [r for r in res if not r[3]]
            out[f"{name}_acc"] = 100 * sum(r[2] for r in good) / max(1, len(good))
            out[f"{name}_ppl"] = math.exp(-sum(r[0] for r in good) / max(1, len(good)))
    return out


# ------------------------------------------------------------------ checkpoints, sampling, export
def save_checkpoint(path, model, optimizer, step, vocab_json, scaler=None, **extra):
    tmp = path + ".tmp"   # write-then-rename: a disconnect mid-save cannot corrupt the previous checkpoint
    torch.save({"model": model.state_dict(), "config": model.cfg.to_dict(), "vocab": vocab_json,
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None, "step": step, **extra}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, device, vocab_json=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if vocab_json is not None and "vocab" in ckpt:
        assert ckpt["vocab"]["teacher_ids"] == vocab_json["teacher_ids"], f"{path} was trained with a DIFFERENT vocab"
    model = StudentLM(Config.from_dict(ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt


@torch.no_grad()
def sample_text(model, vocab, tokenizer, prompt, device, amp, max_new_tokens=80, temperature=0.8, top_k=50,
                repetition_penalty=1.0):
    was_training = model.training
    model.eval()
    ids = vocab.to_student(tokenizer(prompt)["input_ids"])
    with amp():
        out = model.generate(torch.tensor([ids], device=device), max_new_tokens, temperature, top_k, stop_id=vocab.eos_id,
                             repetition_penalty=repetition_penalty)[0, len(ids):].tolist()
    if vocab.eos_id in out:
        out = out[: out.index(vocab.eos_id)]
    model.train(was_training)
    return tokenizer.decode(vocab.to_teacher(out))


def export_final(ckpt_path, out_path):
    from safetensors.torch import save_file
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tensors = {k: v.half().contiguous() for k, v in ck["model"].items() if k != "lm_head.weight"}   # tied to embed.weight
    meta = {"format": "distil-student-v2", "config": json.dumps(ck["config"]), "vocab": json.dumps(ck["vocab"]),
            "step": str(ck["step"]), "prompt_template": PROMPT_TEMPLATE}
    save_file(tensors, out_path, metadata=meta)
    print(f"saved {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")


def read_json(path, default=None):
    try:
        return json.load(open(path))
    except Exception:
        return default


# ------------------------------------------------------------------ history + dashboard
class History:
    """Train / eval / bench records of one run; persisted so plots continue across resumed sessions."""

    def __init__(self, path, start_step=0):
        h = read_json(path, {})
        # a resumed run replays from its last checkpoint: drop anything recorded after it
        self.h = {k: [r for r in h.get(k, []) if r["step"] <= start_step] for k in ("train", "eval", "bench")}
        self.path = path

    def add(self, kind, rec):
        self.h[kind].append({k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in rec.items()})

    def save(self):
        json.dump(self.h, open(self.path + ".tmp", "w"))
        os.replace(self.path + ".tmp", self.path)


def plot_history(h, png, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr, ev, be = h["train"], h["eval"], h["bench"]
    if len(tr) < 2:
        return False
    col = lambda rs, k: np.array([r.get(k, np.nan) for r in rs], dtype=float)
    fig, ax = plt.subplots(2, 3, figsize=(17, 8))
    s = col(tr, "step")
    for k in ("loss", "ce", "kl"):
        if not np.all(np.isnan(col(tr, k))) and np.nanmax(col(tr, k)) > 0:
            ax[0, 0].plot(s, col(tr, k), label=k)
    ax[0, 0].set_title("training loss"); ax[0, 0].legend()
    if ev:
        t = col(ev, "tokens") / 1e6
        unit = "bpb" if any(k.endswith("/bpb") for r in ev for k in r) else "ce"
        for k in sorted({k for r in ev for k in r if k.endswith("/" + unit)}):
            ax[0, 1].plot(t, col(ev, k), marker="o", label=k[: -len(unit) - 1])
        for k in sorted({k for r in ev for k in r if k.endswith("/t_" + unit)}):
            ax[0, 1].axhline(np.nanmean(col(ev, k)), ls=":", c="gray", lw=1)
        ax[0, 1].set_title(f"held-out {'bits/byte' if unit == 'bpb' else 'CE'} (dotted: teacher)"); ax[0, 1].set_xlabel("M tokens"); ax[0, 1].legend()
        for k in ("cosmo/kl", "cosmo/agree"):
            if any(k in r for r in ev):
                ax[0, 2].plot(t, col(ev, k), marker="o", label=k)
        ax[0, 2].set_title("agreement with the teacher"); ax[0, 2].set_xlabel("M tokens")
        if ax[0, 2].get_legend_handles_labels()[0]:
            ax[0, 2].legend()
    if be:
        t = col(be, "tokens") / 1e6
        for k in sorted({k for r in be for k in r if k not in ("step", "tokens", "t") and not k.endswith("_ppl")}):
            ax[1, 0].plot(t, col(be, k), marker="o", label=k)
        ax[1, 0].set_title("zero-shot benchmarks (%)"); ax[1, 0].set_xlabel("M tokens"); ax[1, 0].legend(fontsize=7)
    ax[1, 1].plot(s, col(tr, "lr"), label="lr"); ax[1, 1].set_title("learning rate / grad norm")
    ax2 = ax[1, 1].twinx(); ax2.plot(s, col(tr, "gnorm"), c="tab:red", alpha=0.6)
    ax[1, 2].plot(s, col(tr, "tok_s") / 1e3); ax[1, 2].set_title("throughput (k tokens/s, all GPUs)")
    for a in ax.flat:
        a.grid(alpha=0.3)
    ax[0, 0].set_xlabel("step"); ax[1, 1].set_xlabel("step"); ax[1, 2].set_xlabel("step")
    fig.suptitle(f"{title}  step {int(s[-1])}")
    fig.tight_layout()
    fig.savefig(png, dpi=100)
    plt.close(fig)
    return True
