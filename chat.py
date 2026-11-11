"""Chat with the exported 51M student on your own machine (CPU, no GPU needed).

    python chat.py                  # instruction-tuned model
    python chat.py --model pretrained   # raw text continuation (type the start of a text, it continues it)

Commands inside the chat:  /temp 0.4   /topk 40   /rep 1.15   /len 200   /quit
The model is single-turn: every message is answered on its own, with no memory of earlier ones.
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "hybrid50m"))

import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer   # not `transformers`: it drags in scikit-learn/scipy, which are broken in the conda env

import core as K
from model import Config, StudentLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["chat", "pretrained"], default="chat")
ap.add_argument("--dir", default=os.path.join(HERE, "weights"))
ap.add_argument("--temp", type=float, default=0.6)
ap.add_argument("--topk", type=int, default=40)
ap.add_argument("--rep", type=float, default=1.15)
ap.add_argument("--len", type=int, default=200)
args = ap.parse_args()

cached = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/*/tokenizer.json"))
tok = Tokenizer.from_file(cached[0]) if cached else Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
vocab = K.VocabMap(os.path.join(args.dir, "vocab.json"))
tensors = {}
with safe_open(os.path.join(args.dir, f"student_{args.model}.safetensors"), "pt") as f:
    meta = f.metadata()
    tensors = {k: f.get_tensor(k).float() for k in f.keys()}
assert json.loads(meta["vocab"])["teacher_ids"] == vocab.raw["teacher_ids"], "vocab.json does not match this model"
model = StudentLM(Config.from_dict(json.loads(meta["config"])))
model.load_state_dict(tensors, strict=False)   # lm_head is tied to the embedding and not stored
model.eval()


@torch.no_grad()
def stream(prompt):
    """Sample one token at a time and print the text as it arrives."""
    ids = vocab.to_student(tok.encode(prompt).ids)
    idx = torch.tensor([ids])
    out, shown = [], ""
    for _ in range(args.len):
        logits = model(idx[:, -model.cfg.max_seq_len:])[:, -1, :].float()
        if args.rep != 1.0:
            seen = torch.gather(logits, 1, idx[:, -256:])
            logits.scatter_(1, idx[:, -256:], torch.where(seen > 0, seen / args.rep, seen * args.rep))
        if args.temp <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / args.temp
            if args.topk:
                v, _ = torch.topk(logits, min(args.topk, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        t = int(nxt)
        if t == vocab.eos_id:
            break
        idx = torch.cat([idx, nxt], dim=1)
        out.append(t)
        text = tok.decode(vocab.to_teacher(out))
        if not text.endswith("�"):   # a multi-byte character is still incomplete
            print(text[len(shown):], end="", flush=True)
            shown = text
    print()


print(f"{args.model} model, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, step {meta['step']}. "
      f"/temp /topk /rep /len to adjust, empty line or /quit to exit.\n")
while True:
    try:
        line = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not line or line == "/quit":
        break
    if line.startswith("/"):
        cmd, _, val = line[1:].partition(" ")
        try:
            {"temp": lambda v: setattr(args, "temp", float(v)), "topk": lambda v: setattr(args, "topk", int(v)),
             "rep": lambda v: setattr(args, "rep", float(v)), "len": lambda v: setattr(args, "len", int(v))}[cmd](val)
            print(f"  temp {args.temp} | topk {args.topk} | rep {args.rep} | len {args.len}")
        except (KeyError, ValueError):
            print("  usage: /temp 0.4  /topk 40  /rep 1.15  /len 200  /quit")
        continue
    print("bot> ", end="", flush=True)
    stream(K.PROMPT_TEMPLATE.format(query=line) if args.model == "chat" else line)
    print()
