"""
A minimal GPT-style decoder-only transformer, built from scratch on top of
your own BPETokenizer (see BPE.py / dataset.py).

The big picture:

    tokens (ints) --> embed --> [ Block x N ] --> layernorm --> linear --> logits

Each Block does:

    x = x + SelfAttention(LayerNorm(x))   # tokens "look at" each other
    x = x + FeedForward(LayerNorm(x))     # each token thinks on its own

This is "decoder-only" because attention is causal: token i can only look at
tokens 0..i, never at the future. That's what lets you train on next-token
prediction and then generate text one token at a time at inference time.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention, masked so a position can't attend to future
    positions (hence "causal").

    Intuition: for every token, attention builds a custom "summary" of all
    the tokens before it, weighted by how relevant they are. Concretely,
    each token produces three vectors:

        query (Q): "what am I looking for?"
        key   (K): "what do I contain?"
        value (V): "what info do I pass along if picked?"

    A token's new representation is a weighted sum of every earlier token's
    value vector, where the weight is how well that token's query matches
    the earlier token's key (via a dot product, i.e. Q @ K^T).

    Splitting into multiple "heads" lets different heads specialize in
    different kinds of relationships (e.g. one head might track subject-verb
    agreement, another might track nearby words) since each head has its
    own, smaller Q/K/V projections that all run in parallel.
    """

    def __init__(self, embed_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must divide evenly across heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # One linear layer produces Q, K, and V all at once (faster than three
        # separate layers); we split the output into three chunks afterwards.
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)

        # Recombines the concatenated head outputs back into embed_dim.
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # A lower-triangular matrix of 1s/0s used to zero out "future" attention
        # scores. Registered as a buffer (not a parameter) so it moves with
        # .to(device) but isn't trained or saved as a learnable weight.
        causal_mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, block_size, block_size))

    def forward(self, x, kv_cache=None, use_cache=False):
        """
        kv_cache: optional (past_k, past_v), each (batch, num_heads, past_len,
        head_dim), from a previous call -- lets generation attend over
        already-seen tokens without recomputing their keys/values.
        use_cache: if True, also return this call's (possibly cache-extended)
        keys/values so the caller can pass them into the next step.
        """
        batch_size, seq_len, embed_dim = x.shape

        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * embed_dim)
        q, k, v = qkv.split(embed_dim, dim=-1)

        # Reshape (batch, seq_len, embed_dim) -> (batch, num_heads, seq_len, head_dim)
        # so each head gets its own slice of the embedding to work with.
        def split_heads(t):
            return t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_cache = (k, v) if use_cache else None

        # Raw attention scores: how much does each query match each key?
        # Scaling by sqrt(head_dim) keeps the dot products (and the softmax
        # that follows) from growing too large as head_dim increases.
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if kv_cache is None:
            # Full self-attention over a fresh sequence (training, or a
            # cache "prefill" step): mask out future positions so after
            # softmax those positions get a weight of ~0 — the model
            # literally cannot see ahead.
            attn_scores = attn_scores.masked_fill(
                self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
            )
        # else: decoding new token(s) against a cache — every cached
        # position is already causally before the new one(s), so there's
        # nothing to mask (this assumes seq_len == 1 during cached decoding,
        # which is all GPT.generate ever does).

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of value vectors -> each token's new representation.
        out = attn_weights @ v  # (batch, num_heads, seq_len, head_dim)

        # Merge heads back together: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, embed_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        return self.resid_dropout(self.out_proj(out)), new_cache


class FeedForward(nn.Module):
    """
    A plain per-token MLP: Linear -> GELU -> Linear.

    Attention lets tokens exchange information; this is where the model
    actually "computes" with that information, independently for each
    token. The hidden layer is widened (commonly 4x) to give the network
    more room to combine features before projecting back down.
    """

    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        hidden_dim = 4 * embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    One transformer block: attention, then feed-forward, each wrapped in
    a residual connection and preceded by LayerNorm ("pre-norm").

    Residual connections (x + sublayer(x)) let gradients flow directly
    through the addition during backprop, which is what makes it possible
    to stack many of these blocks without training becoming unstable.

    Pre-norm (normalize *before* the sublayer, rather than after) is the
    modern GPT convention — it tends to train more stably than the original
    "post-norm" Transformer design.
    """

    def __init__(self, embed_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, dropout)

    def forward(self, x, kv_cache=None, use_cache=False):
        attn_out, new_cache = self.attn(self.ln1(x), kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, new_cache


def _sample_next_token(logits, temperature, top_k):
    """logits: (batch, vocab_size) -- the last position's logits. Shared by
    both branches of GPT.generate so the sampling logic isn't duplicated."""
    logits = logits / temperature

    if top_k is not None:
        # Zero out (via -inf) every logit outside the top-k, so sampling
        # can't pick a very unlikely token — a common trick for keeping
        # generation coherent.
        v, _ = torch.topk(logits, top_k)
        logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class GPT(nn.Module):
    """
    The full decoder-only model.

    Pipeline:
      1. Token embedding: look up a learned vector per token id.
      2. Positional embedding: look up a learned vector per position, since
         attention itself has no notion of order (it just weighs pairs of
         tokens) — without this, "cat sat on mat" and "mat sat on cat" would
         look identical to the model.
      3. Add the two embeddings together and run through N transformer Blocks.
      4. Final LayerNorm, then a linear layer projecting back to vocab_size
         logits — a score per vocabulary token for "what comes next."
    """

    def __init__(
        self,
        vocab_size,
        block_size=128,     # max sequence length the model can attend over
        embed_dim=128,      # size of each token's vector representation
        num_heads=4,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, block_size, dropout)
            for _ in range(num_layers)
        ])

        self.ln_final = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: reuse the token embedding matrix as the output
        # projection. This is a standard GPT trick — the intuition is that
        # "map id -> vector" and "map vector -> logits over ids" are
        # naturally each other's inverse, and sharing the weights halves
        # the embedding parameter count while usually helping quality.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None, kv_cache=None, use_cache=False):
        """
        idx: (batch, seq_len) integer token ids -- the full sequence during
             training/prefill, or just the new token(s) when decoding
             against a kv_cache.
        targets: optional (batch, seq_len) integer ids to compute loss against
                 (the ground-truth "next token" for each position).
        kv_cache: optional list of length num_layers of (past_k, past_v),
                  one pair per block, from a previous call.
        use_cache: if True, also return the (extended) per-layer kv_cache
                   for the caller to feed into the next step.

        Returns (logits, loss, new_cache). new_cache is None unless
        use_cache=True.
        """
        batch_size, seq_len = idx.shape
        past_len = kv_cache[0][0].shape[2] if kv_cache is not None else 0
        assert past_len + seq_len <= self.block_size, (
            f"sequence length {past_len + seq_len} exceeds block_size {self.block_size}"
        )

        positions = torch.arange(past_len, past_len + seq_len, device=idx.device)

        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.dropout(x)

        new_cache = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            block_cache = kv_cache[i] if kv_cache is not None else None
            x, block_new_cache = block(x, kv_cache=block_cache, use_cache=use_cache)
            if use_cache:
                new_cache.append(block_new_cache)

        x = self.ln_final(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        loss = None
        if targets is not None:
            # Compare every predicted position's distribution over the vocab
            # against the actual next token, averaged over the whole batch.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss, new_cache

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=False):
        """
        Autoregressive generation: repeatedly predict the next token, append
        it, and feed it back in. This is why the model is causal — at
        inference time there's no "future" to peek at anyway.

        use_cache=False (default): recomputes attention over the whole
        growing sequence every step (truncated to the last block_size
        tokens if it overflows) -- simple, and O(n^2) in tokens generated.

        use_cache=True: keeps each layer's past keys/values around (see
        CausalSelfAttention) so every step only attends the one new token
        against the cached past -- O(n) overall, the standard KV-cache
        speedup. There's no cache eviction here, so prompt length +
        max_new_tokens must fit within block_size.
        """
        self.eval()

        if not use_cache:
            for _ in range(max_new_tokens):
                # Only the last block_size tokens fit in the model's context window.
                idx_cond = idx[:, -self.block_size:]
                logits, _, _ = self(idx_cond)
                next_id = _sample_next_token(logits[:, -1, :], temperature, top_k)
                idx = torch.cat([idx, next_id], dim=1)
            return idx

        # Prefill: run the whole prompt once to seed the per-layer kv_cache.
        logits, _, cache = self(idx, use_cache=True)
        next_id = _sample_next_token(logits[:, -1, :], temperature, top_k)
        idx = torch.cat([idx, next_id], dim=1)

        # Decode: each step only feeds the one newest token; the cache holds
        # everything before it.
        for _ in range(max_new_tokens - 1):
            logits, _, cache = self(next_id, kv_cache=cache, use_cache=True)
            next_id = _sample_next_token(logits[:, -1, :], temperature, top_k)
            idx = torch.cat([idx, next_id], dim=1)

        return idx


if __name__ == "__main__":
    # Quick smoke test with a tiny vocab, no training — just checking shapes
    # and that a forward pass + generation run end to end.
    vocab_size = 1000
    model = GPT(vocab_size, block_size=32, embed_dim=64, num_heads=4, num_layers=2)

    dummy_input = torch.randint(0, vocab_size, (2, 16))   # batch of 2, seq_len 16
    dummy_targets = torch.randint(0, vocab_size, (2, 16))

    logits, loss, _ = model(dummy_input, dummy_targets)
    print("logits shape:", logits.shape)  # (2, 16, vocab_size)
    print("loss:", loss.item())

    generated = model.generate(dummy_input[:, :1], max_new_tokens=10)
    print("generated shape:", generated.shape)  # (2, 11)

    # KV-cache sanity check: caching only changes how attention is computed
    # (incrementally vs from scratch each step), not the math, so with the
    # same seed the sampled tokens should match exactly.
    torch.manual_seed(0)
    gen_nocache = model.generate(dummy_input[:, :1], max_new_tokens=10, use_cache=False)
    torch.manual_seed(0)
    gen_cache = model.generate(dummy_input[:, :1], max_new_tokens=10, use_cache=True)
    print("kv-cache output matches no-cache:", torch.equal(gen_nocache, gen_cache))

    num_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {num_params:,}")
