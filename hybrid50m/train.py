"""Training entry point: one process, or one process per GPU under torchrun (DDP).

    torchrun --nproc_per_node=2 train.py --cfg cfg.json --stage pretrain     # main run (resumes from its checkpoint)
    torchrun --nproc_per_node=2 train.py --cfg cfg.json --stage pilot        # equal-wall-clock arms (KD vs CE, Muon vs AdamW)
    torchrun --nproc_per_node=2 train.py --cfg cfg.json --stage sft          # instruction tuning

Every rank holds its own copy of the student and the teacher and reads its own slice of each global batch; gradients are
averaged by DDP. Rank 0 evaluates, checkpoints and logs. The data position is a pure function of the optimizer step.
"""
import argparse
import json
import math
import os
import signal
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (History, Teacher, TokenFile, VocabMap, WindowSampler, evaluate, export_final, hms, kd_loss, load_teacher_checked,
                  plot_history, prep_bench, read_json, run_bench, sample_text, save_checkpoint, split_batch, teacher_token_bytes,
                  PROMPT_TEMPLATE)
from data import paths
from model import Config, StudentLM
from optim import build_optimizer, lr_at

STOP = {"flag": False}


def _on_signal(*_):
    STOP["flag"] = True


class Env:
    def __init__(self, C):
        self.world = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        self.local = int(os.environ.get("LOCAL_RANK", 0))
        if self.world > 1:
            dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", timeout=timedelta(minutes=120))
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local)
            self.device = torch.device("cuda", self.local)
            torch.backends.cuda.matmul.allow_tf32 = True
        elif torch.backends.mps.is_available() and self.world == 1:
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.main = self.rank == 0
        prec = C.PRECISION
        if self.device.type == "cuda" and prec == "bf16" and torch.cuda.get_device_capability()[0] < 8:
            prec = "fp16"
        self.amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[prec] if self.device.type == "cuda" else None
        self.teacher_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[prec]
        if self.device.type == "cpu":
            self.teacher_dtype = torch.float32
        elif self.device.type == "mps":
            self.teacher_dtype = torch.float16

    def log(self, *a):
        if self.main:
            print(*a, flush=True)

    def amp(self):
        return torch.autocast("cuda", dtype=self.amp_dtype) if self.amp_dtype else nullcontext()

    def scaler(self):
        return torch.amp.GradScaler("cuda", enabled=self.amp_dtype == torch.float16)

    def barrier(self):
        if self.world > 1:
            dist.barrier()

    def any(self, flag):
        # True on every rank if it is True on any rank (stop decisions must be collective)
        if self.world == 1:
            return bool(flag)
        t = torch.tensor([1.0 if flag else 0.0], device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item() > 0)

    def gpu_mem(self):
        return f"{torch.cuda.max_memory_allocated() / 1e9:.1f}GB" if self.device.type == "cuda" else "n/a"


class Context:
    """Everything the stages share: config, device, vocabulary, tokenizer, held-out sets, benchmarks, (lazy) teachers."""

    def __init__(self, C, env):
        from transformers import AutoTokenizer
        self.C, self.env = C, env
        self.P = paths(C.DATA_DIR)
        self.tok = AutoTokenizer.from_pretrained(C.TEACHER_NAME)
        self.vocab_json = json.load(open(C.VOCAB_PATH))
        self.vocab = VocabMap(self.vocab_json)
        self.vocab.set_bytes(teacher_token_bytes(self.tok, self.vocab_json["teacher_vocab_size"]))
        files = {n: TokenFile(self.P[f"{n}_val"], C.SEQ_LEN) for n in ("cosmo", "web", "wiki") if os.path.exists(self.P[f"{n}_val"])}
        self.sets = {n: f.fixed(C.VAL_WINDOWS) for n, f in files.items() if f.n_windows > 0}
        raw = read_json(self.P["bench"])
        self.bench = prep_bench(raw, self.tok) if raw and C.BENCH_EVERY > 0 else {}
        self._teachers = {}

    def teacher(self, name):
        if name not in self._teachers:
            self._teachers.clear()
            if self.env.device.type == "cuda":
                torch.cuda.empty_cache()
            probe = [self.sets["cosmo"][:2]]
            self._teachers[name] = load_teacher_checked(name, self.vocab, self.env.device, self.env.teacher_dtype, probe, self.env.log)
        return self._teachers[name]

    def new_model(self):
        return StudentLM(Config(vocab_size=self.vocab.size, **self.C.STUDENT))


# ------------------------------------------------------------------ the training loop (shared by all stages)
def run_job(ctx, J):
    """J: name, model, opt, scaler, teacher, alpha, get_batch(m), accum, tokens_per_step, total_steps, lr_fn, start_step, best,
    elapsed0, eval_fn(step, final), eval_every, save_every, milestone_every, minutes, hist, extra (saved in checkpoints)."""
    C, env = ctx.C, ctx.env
    model, opt, scaler = J.model, J.opt, J.scaler
    ddp = model
    if env.world > 1:
        ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[env.local] if env.device.type == "cuda" else None)
    latest, best_path = (os.path.join(C.CKPT_DIR, f"{J.name}_{k}.pt") for k in ("latest", "best"))
    step, best, win, skipped, bad = J.start_step, J.best, [], 0, 0
    t0 = t_win = time.time()
    last_logged = step
    elapsed = J.elapsed = lambda: J.elapsed0 + time.time() - t0
    reason = None
    if env.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    env.log(f"[{J.name}] steps {step}->{J.total_steps} | {J.tokens_per_step:,} tokens/step ({J.accum} micro-steps x {env.world} GPU) "
            f"| {'KD alpha ' + str(J.alpha) if J.teacher is not None else 'CE only (no teacher)'} | lr {J.lr_fn(step):.2e}")

    def checkpoint(path):
        if env.main:
            save_checkpoint(path, model, opt, step, ctx.vocab_json, scaler, best_val=best, elapsed=elapsed(), **J.extra)

    while step < J.total_steps:
        lr = J.lr_fn(step)
        for g in opt.param_groups:
            g["lr"] = lr
        sums = torch.zeros(3, device=env.device)
        for a in range(J.accum):
            xt, xs, y = J.get_batch(step * J.accum + a)
            with (ddp.no_sync() if env.world > 1 and a < J.accum - 1 else nullcontext()):
                t_logits = J.teacher.logits(xt) if J.teacher is not None else None
                with env.amp():
                    s_logits = ddp(xs)
                loss, kl, ce = kd_loss(s_logits, t_logits, y, C.TEMPERATURE, J.alpha if J.teacher is not None else 0.0)
                scaler.scale(loss / J.accum).backward()
            sums += torch.stack([loss.detach(), kl.detach(), ce.detach()]) / J.accum
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP).item()
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        skipped += int(scaler.get_scale() < scale_before)
        if env.world > 1:
            dist.all_reduce(sums)
            sums /= env.world
        loss_v, kl_v, ce_v = sums.tolist()
        bad = 0 if math.isfinite(loss_v) else bad + 1
        if bad >= 5:
            raise FloatingPointError("loss was NaN/inf for 5 steps in a row. The last checkpoint is intact; try PRECISION='fp32' "
                                     "(or 'bf16' on an Ampere+ GPU) or a lower LR")
        if math.isfinite(loss_v):
            win.append((loss_v, kl_v, ce_v, gnorm))

        if step in (1, 5) or step % C.LOG_EVERY == 0 or step == J.total_steps:
            now = time.time()
            w = np.nanmean(np.array(win), axis=0) if win else np.full(4, np.nan)
            tok_s = (step - last_logged) * J.tokens_per_step / max(1e-9, now - t_win)
            rem = (J.total_steps - step) * J.tokens_per_step / max(1.0, tok_s)
            if env.main:
                J.hist.add("train", dict(step=step, tokens=step * J.tokens_per_step, t=elapsed(), loss=w[0], kl=w[1], ce=w[2],
                                         gnorm=w[3], lr=lr, tok_s=tok_s, scale=scaler.get_scale()))
                print(f"step {step:6d}/{J.total_steps} ({100 * step / J.total_steps:4.1f}%) | {step * J.tokens_per_step / 1e6:7.1f}M tok "
                      f"| loss {w[0]:.3f} kl {w[1]:.3f} ce {w[2]:.3f} | lr {lr:.2e} | gnorm {w[3]:.2f} | {tok_s:,.0f} tok/s "
                      f"| elapsed {hms(elapsed())} | eta {hms(rem) if J.total_steps < 10 ** 8 else '-'} | mem {env.gpu_mem()}"
                      + (f" | loss-scale {scaler.get_scale():.0f} skipped {skipped}" if env.amp_dtype == torch.float16 else ""), flush=True)
            win, last_logged, t_win = [], step, now

        # collective stop decision: session time limit, a signal, or this arm's wall-clock budget
        over_session = (time.time() - C.NB_START) / 3600 > C.MAX_RUN_HOURS
        over_budget = J.minutes is not None and elapsed() / 60 >= J.minutes
        stop = env.any(over_session or over_budget or STOP["flag"])
        if stop and step < J.total_steps:
            reason = "budget" if over_budget and not (over_session or STOP["flag"]) else ("signal" if STOP["flag"] else "time")
        final = step == J.total_steps or reason is not None

        if step % J.eval_every == 0 or final:
            if env.main:
                metrics, lines, score = J.eval_fn(step, final)
                J.hist.add("eval", dict(step=step, tokens=step * J.tokens_per_step, t=elapsed(), **metrics))
                print("=" * 100 + f"\nEVAL @ step {step}/{J.total_steps}  " + " | ".join(f"{k} {v:.3f}" for k, v in metrics.items() if k.endswith(("/ce", "/bpb")))
                      + "".join("\n  " + ln for ln in lines) + "\n" + "=" * 100, flush=True)
                if score < best:
                    best = score
                    save_checkpoint(best_path, model, None, step, ctx.vocab_json, best_val=best, **J.extra)
                J.hist.save()
                if plot_history(J.hist.h, os.path.join(C.PERSIST_DIR, f"dashboard_{J.name}.png"), J.name):
                    print("DASHBOARD " + os.path.join(C.PERSIST_DIR, f"dashboard_{J.name}.png"), flush=True)
            env.barrier()
            t_win = time.time()
        if step % J.save_every == 0 or final:
            checkpoint(latest)
            if env.main:
                J.hist.save()
        if J.milestone_every and step % J.milestone_every == 0 and env.main:
            save_checkpoint(os.path.join(C.CKPT_DIR, f"{J.name}_step{step}.pt"), model, None, step, ctx.vocab_json, **J.extra)
        if reason is not None:
            break

    status = "paused" if reason in ("time", "signal") else "done"
    if env.main:
        J.hist.save()
        write_state(ctx, J.name, status, step, J.total_steps, best, elapsed(), J.tokens_per_step, reason)
        print(f"[{J.name}] {status.upper()} at step {step}/{J.total_steps}" + (f" ({reason})" if reason else "") + f" | best val bpb {best:.4f}", flush=True)
    env.barrier()
    return status


def write_state(ctx, name, status, step, total, best, elapsed, tokens_per_step, reason=None):
    if ctx.env.main:
        json.dump(dict(name=name, status=status, reason=reason, step=step, total_steps=total, best=best, elapsed=elapsed,
                       tokens=step * tokens_per_step, world=ctx.env.world, tokens_per_step=tokens_per_step),
                  open(os.path.join(ctx.C.CKPT_DIR, f"state_{name}.json"), "w"))


def resume_or_new(ctx, name, muon, lr, from_ckpt=None):
    """Model + optimizer + scaler, restored from `<name>_latest.pt` if it exists (else optionally initialised from `from_ckpt`)."""
    C, env = ctx.C, ctx.env
    latest = os.path.join(C.CKPT_DIR, f"{name}_latest.pt")
    model, ck = ctx.new_model(), None
    start, best, elapsed0 = 0, float("inf"), 0.0
    src = latest if os.path.exists(latest) else from_ckpt
    if src:
        ck = torch.load(src, map_location="cpu", weights_only=False)
        assert ck["vocab"]["teacher_ids"] == ctx.vocab_json["teacher_ids"], f"{src} was trained with a different vocab"
        model = StudentLM(Config.from_dict(ck["config"]))
        model.load_state_dict(ck["model"])
    model.to(env.device)
    opt, scaler = build_optimizer(model, lr, C.WEIGHT_DECAY, muon), env.scaler()
    if src == latest:
        if ck.get("muon", muon) != muon:
            raise RuntimeError(f"{latest} was trained with USE_MUON={ck.get('muon')}; the config says {muon}. Change the config back.")
        opt.load_state_dict(ck["optimizer"])
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start, best, elapsed0 = ck["step"], ck.get("best_val", best), ck.get("elapsed", 0.0)
        env.log(f"[{name}] RESUMING from step {start} ({ck.get('elapsed', 0) / 3600:.1f} h of training so far; best val bpb {best:.4f})")
    else:
        env.log(f"[{name}] starting from " + (f"{src} (step {ck['step']})" if src else "scratch") + f" | {model.num_params() / 1e6:.2f}M params")
    return model, opt, scaler, start, best, elapsed0, ck


# ------------------------------------------------------------------ stage 1: pretraining (also the pilot arms)
def run_pretrain(ctx, name, kd, muon, minutes=None, total_steps=None, sched=None, eval_every=None):
    C, env = ctx.C, ctx.env
    total = total_steps or C.PRETRAIN_MAX_STEPS
    accum = C.TOKENS_PER_STEP // (C.BATCH_SIZE * C.SEQ_LEN * env.world)
    assert accum * C.BATCH_SIZE * C.SEQ_LEN * env.world == C.TOKENS_PER_STEP, \
        "TOKENS_PER_STEP must be a multiple of BATCH_SIZE * SEQ_LEN * number of GPUs"
    latest = os.path.join(C.CKPT_DIR, f"{name}_latest.pt")
    if C.REQUIRE_CHECKPOINT and not os.path.exists(latest):
        raise FileNotFoundError(f"{latest} not found and REQUIRE_CHECKPOINT=True: attach the previous run's output as an input")
    model, opt, scaler, start, best, elapsed0, _ = resume_or_new(ctx, name, muon, C.LR)
    if start >= total:
        env.log(f"[{name}] already finished (step {start}/{total})")
        write_state(ctx, name, "done", start, total, best, elapsed0, C.TOKENS_PER_STEP)
        return "done"
    teacher = ctx.teacher(C.TEACHER_NAME) if kd else None
    train = TokenFile(ctx.P["train"], C.SEQ_LEN)
    sampler = WindowSampler(train.n_windows, C.SEED)
    kind = sched or C.SCHEDULE
    hist = History(os.path.join(C.CKPT_DIR, f"history_{name}.json"), start)
    env.log(f"[{name}] {train.n_windows:,} training windows ({train.n_tokens / 1e6:.0f}M tokens); {total * C.TOKENS_PER_STEP / 1e6:,.0f}M tokens "
            f"planned = {total * C.TOKENS_PER_STEP / train.n_tokens:.2f} epochs | schedule {kind} | optimizer {'Muon+AdamW' if muon else 'AdamW'}")
    if env.main:
        json.dump(dict(params=model.num_params(), config=model.cfg.to_dict(), world=env.world, device=str(env.device),
                       gpu=torch.cuda.get_device_name() if env.device.type == "cuda" else env.device.type,
                       precision=C.PRECISION, kd=kd, muon=muon, tokens_per_step=C.TOKENS_PER_STEP, total_steps=total),
                  open(os.path.join(C.CKPT_DIR, f"info_{name}.json"), "w"))

    def get_batch(m):
        idx = sampler.take((m * env.world + env.rank) * C.BATCH_SIZE, C.BATCH_SIZE)
        return split_batch(train.get(idx), ctx.vocab, env.device)

    def eval_fn(step, final):
        m, lines = {}, []
        for nm, w in ctx.sets.items():
            r = evaluate(model, ctx.vocab, w, env.device, env.amp, teacher, C.EVAL_BS)
            m.update({f"{nm}/{k}": v for k, v in r.items()})
            lines.append(f"{nm:6s} CE {r['ce']:.3f} ppl {r['ppl']:.1f} bpb {r['bpb']:.4f}"
                         + (f" | teacher CE {r['t_ce']:.3f} bpb {r['t_bpb']:.4f} | KL {r['kl']:.3f} | top-1 agree {r['agree']:.1f}%" if teacher is not None else ""))
        if ctx.bench and (final or step % C.BENCH_EVERY == 0):
            b = run_bench(model, ctx.vocab, ctx.bench, env.device, env.amp, C.SEQ_LEN - 1)
            J.hist.add("bench", dict(step=step, tokens=step * C.TOKENS_PER_STEP, t=J.elapsed(), **b))
            lines.append("bench  " + " | ".join(f"{k} {v:.1f}" for k, v in b.items()))
        for p in C.SAMPLE_PROMPTS:
            lines.append(f"> {p!r} -> {sample_text(model, ctx.vocab, ctx.tok, p, env.device, env.amp, 48, 0.7, 40, 1.15)!r}")
        score = float(np.mean([m[f"{nm}/bpb"] for nm in ("cosmo", "web") if f"{nm}/bpb" in m]))
        return m, lines, score

    J = SimpleNamespace(name=name, model=model, opt=opt, scaler=scaler, teacher=teacher, alpha=C.ALPHA, get_batch=get_batch, accum=accum,
                        tokens_per_step=C.TOKENS_PER_STEP, total_steps=total, lr_fn=lambda s: lr_at(s, total, C.LR, C.WARMUP, kind, C.DECAY_FRAC),
                        start_step=start, best=best, elapsed0=elapsed0, eval_fn=eval_fn, eval_every=eval_every or C.EVAL_EVERY,
                        save_every=C.SAVE_EVERY, milestone_every=C.MILESTONE_EVERY if name == "pretrain" else 0, minutes=minutes,
                        hist=hist, extra=dict(muon=muon, kd=kd))
    status = run_job(ctx, J)
    del model, opt
    return status


def run_pilot(ctx):
    C, env = ctx.C, ctx.env
    for arm in C.PILOT_ARMS:
        name = f"pilot_{arm['name']}"
        if os.path.exists(os.path.join(C.CKPT_DIR, f"state_{name}.json")) and read_json(os.path.join(C.CKPT_DIR, f"state_{name}.json"))["status"] == "done":
            env.log(f"[{name}] already done")
            continue
        status = run_pretrain(ctx, name, arm["kd"], arm["muon"], minutes=C.PILOT_MINUTES, total_steps=10 ** 9, sched="constant",
                              eval_every=C.PILOT_EVAL_EVERY)
        if env.device.type == "cuda":
            torch.cuda.empty_cache()
        if status == "paused":
            break


# ------------------------------------------------------------------ stage 2: instruction tuning
def load_sft(ctx):
    C, env = ctx.C, ctx.env
    z = np.load(ctx.P["sft"])
    ids, plen, tlen = z["ids"], z["plen"].astype(np.int64), z["tlen"].astype(np.int64)
    starts = np.cumsum(tlen) - tlen
    unk = (ctx.vocab.t2s[torch.from_numpy(ids.astype(np.int64))] == ctx.vocab.unk_id).numpy().astype(np.int64)
    cs = np.concatenate([[0], np.cumsum(unk)])
    frac = (cs[starts + tlen] - cs[starts + plen]) / np.maximum(1, tlen - plen)
    keep = np.where(frac <= C.SFT_MAX_UNK_FRAC)[0]
    env.log(f"[sft] {len(keep):,} of {len(plen):,} examples kept (<= {C.SFT_MAX_UNK_FRAC:.0%} unknown tokens in the answer)")
    rng = np.random.RandomState(C.SEED)
    keep = keep[rng.permutation(len(keep))]
    n_val = max(1, min(2000, len(keep) // 50))
    return ids, starts, plen, tlen, keep[:n_val], keep[n_val:]


def run_sft(ctx):
    C, env = ctx.C, ctx.env
    ids, starts, plen, tlen, val_idx, train_idx = load_sft(ctx)
    eos, B = ctx.vocab_json["eos_teacher_id"], C.SFT_BATCH_SIZE

    def collate(idx):
        T = int(max(tlen[i] for i in idx)) - 1
        x = torch.full((len(idx), T), eos, dtype=torch.long)
        y = torch.full((len(idx), T), -100, dtype=torch.long)
        for k, i in enumerate(idx):
            full = torch.from_numpy(ids[starts[i]: starts[i] + tlen[i]].astype(np.int64))
            x[k, : len(full) - 1] = full[:-1]
            y[k, plen[i] - 1: len(full) - 1] = full[plen[i]:]
        x, y = x.to(env.device), y.to(env.device)
        lab = ctx.vocab.to_student(y.clamp(min=0))
        lab[(y == -100) | (lab == ctx.vocab.unk_id)] = -100
        return x, ctx.vocab.to_student(x), lab

    order = train_idx[np.argsort(tlen[train_idx], kind="stable")]       # length-bucketed batches: little padding
    chunks = [order[i: i + B] for i in range(0, len(order) - B + 1, B)]
    n_chunks = len(chunks)
    accum = C.SFT_GRAD_ACCUM
    total = n_chunks * C.SFT_EPOCHS // (accum * env.world)
    avg_len = float(np.mean(tlen[train_idx]))
    tokens_per_step = int(B * accum * env.world * avg_len)
    perms = {}

    def get_batch(m):
        p = m * env.world + env.rank
        ep, i = divmod(p, n_chunks)
        if ep not in perms:
            perms.clear()
            perms[ep] = np.random.RandomState(C.SEED + ep).permutation(n_chunks)
        return collate(chunks[perms[ep][i]])

    pre_best = os.path.join(C.CKPT_DIR, "pretrain_best.pt")
    pre = torch.load(pre_best, map_location="cpu", weights_only=False) if os.path.exists(pre_best) else {}
    muon = pre.get("muon", C.USE_MUON)
    model, opt, scaler, start, best, elapsed0, _ = resume_or_new(ctx, "sft", muon, C.SFT_LR, from_ckpt=pre_best)
    if start >= total:
        env.log(f"[sft] already finished (step {start}/{total})")
        write_state(ctx, "sft", "done", start, total, best, elapsed0, tokens_per_step)
        return "done"
    teacher = ctx.teacher(C.SFT_TEACHER) if C.SFT_ALPHA > 0 else None
    env.log(f"[sft] {len(train_idx):,} train / {len(val_idx)} val examples, {n_chunks:,} batches of {B}; {total:,} steps over {C.SFT_EPOCHS} epochs "
            f"(~{total * tokens_per_step / 1e6:.0f}M tokens)")
    hist = History(os.path.join(C.CKPT_DIR, "history_sft.json"), start)

    def eval_fn(step, final):
        model.eval()
        ce_s = kl_s = 0.0
        nb = 0
        with torch.no_grad():
            for i in range(0, len(val_idx), B):
                xt, xs, y = collate(val_idx[i: i + B])
                with env.amp():
                    s = model(xs)
                _, kl, ce = kd_loss(s, teacher.logits(xt) if teacher is not None else None, y, 1.0, 1.0 if teacher is not None else 0.0)
                ce_s += ce.item(); kl_s += kl.item(); nb += 1
        model.train()
        m = {"sft/ce": ce_s / nb, "sft/kl": kl_s / nb}
        lines = [f"sft val CE {m['sft/ce']:.3f} ppl {math.exp(m['sft/ce']):.1f}" + (f" | KL {m['sft/kl']:.3f}" if teacher is not None else "")]
        for q in C.SAMPLE_QUERIES:
            a = sample_text(model, ctx.vocab, ctx.tok, PROMPT_TEMPLATE.format(query=q), env.device, env.amp, 80, 0.6, 40, 1.15).strip()
            lines.append(f"Q: {q}\n    A: {a!r}")
        if ctx.bench and final:
            b = run_bench(model, ctx.vocab, ctx.bench, env.device, env.amp, C.SEQ_LEN - 1)
            hist.add("bench", dict(step=step, tokens=step * tokens_per_step, t=J.elapsed(), **b))
            lines.append("bench  " + " | ".join(f"{k} {v:.1f}" for k, v in b.items()))
        return m, lines, m["sft/ce"]

    J = SimpleNamespace(name="sft", model=model, opt=opt, scaler=scaler, teacher=teacher, alpha=C.SFT_ALPHA, get_batch=get_batch, accum=accum,
                        tokens_per_step=tokens_per_step, total_steps=total, lr_fn=lambda s: lr_at(s, total, C.SFT_LR, C.SFT_WARMUP, "cosine"),
                        start_step=start, best=best, elapsed0=elapsed0, eval_fn=eval_fn, eval_every=C.SFT_EVAL_EVERY, save_every=C.SFT_SAVE_EVERY,
                        milestone_every=0, minutes=None, hist=hist, extra=dict(muon=muon, kd=teacher is not None))
    return run_job(ctx, J)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--stage", required=True, choices=["pretrain", "pilot", "sft"])
    args = ap.parse_args()
    C = SimpleNamespace(**json.load(open(args.cfg)))
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    env = Env(C)
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    env.log(f"device {env.device} | {env.world} process(es) | student autocast {env.amp_dtype} | teacher {env.teacher_dtype}")
    ctx = Context(C, env)
    if args.stage == "pretrain":
        run_pretrain(ctx, "pretrain", C.KD, C.USE_MUON)
    elif args.stage == "pilot":
        run_pilot(ctx)
    else:
        run_sft(ctx)
    if env.world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
