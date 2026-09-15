"""
LoRA (Low-Rank Adaptation) for the GPT model in model.py.

Instead of fine-tuning every weight, LoRA freezes the base model and adds a
small trainable low-rank update to a chosen linear layer:

    y = base(x) + scaling * (x @ A^T @ B^T)

where A is (r, in_features) and B is (out_features, r), with r << in/out
features. B is zero-initialized so a freshly-wrapped model computes exactly
the same outputs as the unwrapped one until training moves A/B away from
init. Only A and B are trainable -- for r=8 on this model's 128-dim
attention/FF layers, that's a small fraction of the full parameter count.
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base_linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)

        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scaling * lora_out

    def extra_repr(self):
        return f"r={self.r}, alpha={self.scaling * self.r:.1f}"


def add_lora(model, r=8, alpha=16, dropout=0.0, target_ff=True):
    """
    Freezes every parameter in `model`, then wraps each transformer block's
    attention projections (qkv_proj, out_proj), and optionally its
    feed-forward linears, with LoRALinear. Mutates `model` in place and
    returns it for convenience.
    """
    for p in model.parameters():
        p.requires_grad = False

    for block in model.blocks:
        block.attn.qkv_proj = LoRALinear(block.attn.qkv_proj, r, alpha, dropout)
        block.attn.out_proj = LoRALinear(block.attn.out_proj, r, alpha, dropout)
        if target_ff:
            block.ff.net[0] = LoRALinear(block.ff.net[0], r, alpha, dropout)
            block.ff.net[2] = LoRALinear(block.ff.net[2], r, alpha, dropout)

    return model


def trainable_parameters(model):
    """Returns (trainable_count, total_count) -- the numbers that make a
    PEFT-vs-full-finetune comparison apples-to-apples."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def lora_state_dict(model):
    """
    Just the LoRA A/B matrices -- the small adapter you'd actually want to
    save/share, instead of a full copy of the (unchanged, frozen) base model.
    """
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if name.endswith("lora_A") or name.endswith("lora_B")
    }


if __name__ == "__main__":
    # Quick smoke test: wrapping should not change outputs (B starts at
    # zero), and should freeze everything except the new LoRA params.
    from model import GPT

    torch.manual_seed(0)
    model = GPT(vocab_size=50, block_size=16, embed_dim=32, num_heads=4, num_layers=2)
    model.eval()

    x = torch.randint(0, 50, (2, 8))
    logits_before, _, _ = model(x)

    add_lora(model, r=4, alpha=8)
    logits_after, _, _ = model(x)

    print("LoRA-wrapped output matches base (B initialized to zero):", torch.allclose(logits_before, logits_after))

    trainable, total = trainable_parameters(model)
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    adapter = lora_state_dict(model)
    print(f"adapter tensors saved: {len(adapter)}")
