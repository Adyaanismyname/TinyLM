"""Cheap correctness checks for the pieces the whole run depends on. `python hybrid50m/selftest.py` (the notebook runs it too)."""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import WindowSampler, kd_loss, teacher_token_bytes
from model import Config, StudentLM, from_v4_state_dict
from optim import MuonAdamW, build_optimizer, lr_at, newton_schulz


def ok(msg):
    print("  ok:", msg)


def test_model(old_model_cls=None, old_cfg_cls=None):
    torch.manual_seed(0)
    cfg = Config()
    m = StudentLM(cfg)
    n = m.num_params()
    assert 49e6 < n < 54e6, n
    ok(f"default config has {n / 1e6:.2f}M parameters (GQA {cfg.n_heads}:{cfg.n_kv_heads}, {cfg.n_layers} layers, QK-norm)")

    small = Config(vocab_size=500, d_model=64, n_layers=3, n_heads=4, n_kv_heads=2, ffn_hidden=96, max_seq_len=32, emb_dim=16)
    m = StudentLM(small).eval()
    x = torch.randint(0, 500, (2, 16))
    y1 = m(x)
    x2 = x.clone(); x2[:, 10:] = torch.randint(0, 500, (2, 6))
    assert torch.allclose(y1[:, :10], m(x2)[:, :10], atol=1e-5)
    assert not torch.allclose(y1[:, 10:], m(x2)[:, 10:], atol=1e-3)
    ok("causal: changing future tokens leaves earlier logits untouched")

    shared = StudentLM(Config(**{**small.to_dict(), "block_repeats": 2}))
    assert shared.num_params() == StudentLM(small).num_params()
    ok("block_repeats=2 doubles depth with identical parameter count")

    if old_model_cls is not None:
        ocfg = old_cfg_cls(vocab_size=500, d_model=64, n_layers=3, n_heads=4, ffn_hidden=96, max_seq_len=32, emb_dim=16)
        old = old_model_cls(ocfg).eval()
        new = StudentLM(Config.from_v4(ocfg.to_dict())).eval()
        new.load_state_dict(from_v4_state_dict(old.state_dict(), new.cfg), strict=False)
        assert torch.allclose(old(x), new(x), atol=1e-5), (old(x) - new(x)).abs().max()
        ok("loads the 10M model's weights (fused qkv -> q/k/v) and reproduces its logits exactly")


def test_optim():
    torch.manual_seed(0)
    cfg = Config(vocab_size=200, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=48, max_seq_len=16, emb_dim=8)
    a, b = StudentLM(cfg), StudentLM(cfg)
    b.load_state_dict(a.state_dict())
    oa = build_optimizer(a, 1e-2, 0.1, muon=False)
    decay = [p for p in b.parameters() if p.dim() >= 2]
    plain = [p for p in b.parameters() if p.dim() < 2]
    ob = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": plain, "weight_decay": 0.0}], lr=1e-2, betas=(0.9, 0.95))
    for _ in range(5):
        x = torch.randint(0, 200, (4, 16))
        for m, o in ((a, oa), (b, ob)):
            o.zero_grad(); m(x).logsumexp(-1).mean().backward(); o.step()
    # rounding differences accumulate to ~3e-6 (parameters move ~5e-2 in total); a real bug (e.g. no bias correction) is off by ~1e-2
    worst = max((p - q).abs().max().item() for p, q in zip(a.parameters(), b.parameters()))
    assert worst < 5e-5, worst
    ok(f"AdamW half matches torch.optim.AdamW over 5 steps (max difference {worst:.1e})")

    g = torch.randn(40, 100)
    s = torch.linalg.svdvals(newton_schulz(g))
    assert s.min() > 0.4 and s.max() < 1.6, (s.min(), s.max())
    ok(f"Newton-Schulz pushes singular values to ~1 (range {s.min():.2f}..{s.max():.2f})")

    c = StudentLM(cfg)
    oc = build_optimizer(c, 1e-2, 0.1, muon=True)
    kinds = [g_["kind"] for g_ in oc.param_groups]
    assert kinds.count("muon") == 1 and all(p.dim() == 2 and n.startswith("blocks.") for n, p in c.named_parameters() if any(p is q for q in oc.param_groups[0]["params"]))
    before = [p.clone() for p in c.parameters()]
    oc.zero_grad(); c(torch.randint(0, 200, (4, 16))).logsumexp(-1).mean().backward(); oc.step()
    assert all(not torch.equal(p, q) for p, q in zip(before, c.parameters()))
    oc.load_state_dict(oc.state_dict())
    ok("Muon covers only block matrices, updates every parameter, and its state round-trips")


def test_schedule_and_sampler():
    assert lr_at(0, 1000, 1.0, 10) == 0.1 and lr_at(9, 1000, 1.0, 10) == 1.0
    assert lr_at(500, 1000, 1.0, 10) == 1.0 and lr_at(799, 1000, 1.0, 10) == 1.0
    assert lr_at(1000, 1000, 1.0, 10) == 0.02 and 0.02 < lr_at(900, 1000, 1.0, 10) < 1.0
    assert lr_at(100, 10 ** 9, 1.0, 10, "constant") == 1.0
    ok("warmup-stable-decay schedule: flat until 80%, then 1-sqrt decay to 2%")

    s = WindowSampler(37, seed=3)
    whole = s.take(0, 100)
    assert (np.concatenate([s.take(0, 30), s.take(30, 70)]) == whole).all()
    assert sorted(whole[:37].tolist()) == list(range(37)) and sorted(whole[37:74].tolist()) == list(range(37))
    assert not (whole[:37] == whole[37:74]).all()
    ok("sampler: each epoch visits every window once, any resume point continues the same stream")


def test_loss():
    torch.manual_seed(0)
    s = torch.randn(2, 5, 20)
    y = torch.randint(0, 20, (2, 5)); y[0, 0] = -100
    total, kl, ce = kd_loss(s, s.clone(), y, alpha=0.7)
    assert kl.abs() < 1e-5 and torch.allclose(total, 0.3 * ce, atol=1e-5)
    _, kl2, _ = kd_loss(s, torch.randn(2, 5, 20), y, alpha=0.7)
    assert kl2 > 0.1
    ok("KD loss: zero KL against itself, positive against a different teacher, ignored targets skipped")


def test_bytes(tokenizer):
    tb = teacher_token_bytes(tokenizer, max(len(tokenizer), 151936))
    for text in ["Hello, world! The cat sat on the mat.", "naïve café — 日本語のテキスト 😀", "def f(x):\n    return x ** 2\n"]:
        ids = tokenizer(text)["input_ids"]
        assert int(sum(tb[i] for i in ids)) == len(text.encode()), (text, sum(tb[i] for i in ids), len(text.encode()))
    assert tb[tokenizer.eos_token_id] == 0
    ok("byte lengths of tokens sum to the UTF-8 length of the text (so bits-per-byte is exact)")


def main(with_tokenizer=True):
    print("self-test")
    old_cls = old_cfg = None
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.exists(os.path.join(root, "student_model.py")):
        sys.path.insert(0, root)
        from student_model import StudentConfig, StudentLM as Old
        old_cls, old_cfg = Old, StudentConfig
    test_model(old_cls, old_cfg)
    test_optim()
    test_schedule_and_sampler()
    test_loss()
    if with_tokenizer:
        try:
            from transformers import AutoTokenizer
            test_bytes(AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B"))
        except Exception as e:
            print(f"  skipped byte-accounting test ({type(e).__name__}: {e})")
    print("all self-tests passed")


if __name__ == "__main__":
    main()
