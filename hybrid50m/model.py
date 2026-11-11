"""~50M-parameter student: deep-and-narrow Llama-style decoder (MobileLLM recipe).

GQA (few key/value heads), QK-norm (fp16 stability), RoPE, RMSNorm, SwiGLU and a factorized tied embedding.
Optional `block_repeats` runs every stored block several times in a row (MobileLLM-LS): more depth, same parameters.
Also loads the weights of the earlier 10M model (fused qkv, no QK-norm) through `from_v4_state_dict`.
"""
import math
from dataclasses import dataclass, asdict, fields
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 32768
    d_model: int = 512
    n_layers: int = 18
    n_heads: int = 8
    n_kv_heads: int = 2
    ffn_hidden: int = 1280
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    dropout: float = 0.0
    emb_dim: Optional[int] = 128   # factorized tied embedding: vocab * emb_dim, projected up/down (None: plain tied)
    qk_norm: bool = True
    block_repeats: int = 1         # >1: weight sharing across depth (compute grows, parameters do not)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_v4(cls, d):
        # the 10M model: plain multi-head attention, no QK-norm
        return cls.from_dict({**d, "n_kv_heads": d["n_heads"], "qk_norm": False, "block_repeats": 1})


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()  # statistics in fp32 so fp16 activations cannot overflow in the square
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


def rope_cache(seq_len, head_dim, theta, device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len, device=device).float(), inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, D); rotate interleaved pairs in fp32, hand back x's dtype
    xf = x.float()
    x1, x2 = xf[..., 0::2], xf[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0 and cfg.n_heads % cfg.n_kv_heads == 0
        self.h, self.kv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.d_model // cfg.n_heads
        self.wq = nn.Linear(cfg.d_model, self.h * self.hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.kv * self.hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, self.kv * self.hd, bias=False)
        self.out = nn.Linear(self.h * self.hd, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(self.hd) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.hd) if cfg.qk_norm else nn.Identity()
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        q, k = apply_rope(self.q_norm(q), cos, sin), apply_rope(self.k_norm(k), cos, sin)
        if self.kv != self.h:  # share each key/value head between h/kv query heads
            k, v = k.repeat_interleave(self.h // self.kv, dim=1), v.repeat_interleave(self.h // self.kv, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(B, T, self.h * self.hd))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        x = x + self.drop(self.attn(self.norm1(x), cos, sin))
        return x + self.drop(self.mlp(self.norm2(x)))


class StudentLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        e = cfg.emb_dim or cfg.d_model
        self.embed = nn.Embedding(cfg.vocab_size, e)
        self.proj_in = nn.Linear(e, cfg.d_model, bias=False) if cfg.emb_dim else nn.Identity()
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.proj_out = nn.Linear(cfg.d_model, e, bias=False) if cfg.emb_dim else nn.Identity()
        self.lm_head = nn.Linear(e, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # tied

        cos, sin = rope_cache(cfg.max_seq_len, cfg.d_model // cfg.n_heads, cfg.rope_theta, "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        if cfg.emb_dim:  # narrow embeddings need a larger std so the projected signal is not tiny
            nn.init.normal_(self.embed.weight, mean=0.0, std=0.1)
        depth = cfg.n_layers * cfg.block_repeats
        for name, p in self.named_parameters():  # GPT-2 style scaled init for the residual-branch outputs
            if name.endswith("attn.out.weight") or name.endswith("mlp.down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * depth))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx):
        T = idx.shape[1]
        assert T <= self.cfg.max_seq_len, f"sequence length {T} > max_seq_len {self.cfg.max_seq_len}"
        x = self.proj_in(self.embed(idx))
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for block in self.blocks:
            for _ in range(self.cfg.block_repeats):
                x = block(x, cos, sin)
        return self.lm_head(self.proj_out(self.norm(x)))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, stop_id=None, repetition_penalty=1.0):
        # no KV cache on purpose: this is for samples during training. Inference gets its own cached path at deployment.
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.cfg.max_seq_len:])[:, -1, :].float()
            if repetition_penalty != 1.0:
                seen = torch.gather(logits, 1, idx[:, -256:])
                logits.scatter_(1, idx[:, -256:], torch.where(seen > 0, seen / repetition_penalty, seen * repetition_penalty))
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if stop_id is not None and (nxt == stop_id).all():
                break
        return idx


def from_v4_state_dict(sd, cfg: Config):
    """Map a 10M-model state dict (fused `qkv`) onto this architecture (needs n_kv_heads == n_heads, qk_norm False)."""
    d = cfg.d_model
    out = {}
    for k, v in sd.items():
        if k == "lm_head.weight":
            continue
        if k.endswith("attn.qkv.weight"):
            pre = k[: -len("qkv.weight")]
            out[pre + "wq.weight"], out[pre + "wk.weight"], out[pre + "wv.weight"] = v[:d], v[d:2 * d], v[2 * d:]
        else:
            out[k] = v
    return out
