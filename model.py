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

# Cross-entropy's own default ignore_index is already -100, so this doesn't
# change any existing behavior -- it's named here so data.py's EpochBatcher
# (which masks instruction-tuning header tokens by writing -100 into the
# target) and this file agree on the convention explicitly instead of by
# coincidentally relying on a library default.
IGNORE_INDEX = -100


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

    attn_impl selects how the softmax(QK^T/sqrt(d))V computation itself is
    carried out:
      "sdpa"   -- torch.nn.functional.scaled_dot_product_attention, which
                  dispatches to a fused kernel (flash attention on CUDA when
                  available). Mathematically the same computation as
                  "manual" below; faster and lower peak memory, especially
                  at longer context lengths, because it never materializes
                  the full (seq_len, seq_len) score matrix.
      "manual" -- the explicit Q @ K^T, mask, softmax, @ V steps below.
                  Kept as a reference implementation and for bench_train.py
                  (Experiment 4), which specifically wants to attribute
                  training-speed differences to LoRA vs. full fine-tuning,
                  not to which attention kernel is in use.
    """

    def __init__(self, embed_dim, num_heads, block_size, dropout=0.1, attn_impl="sdpa"):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must divide evenly across heads"
        assert attn_impl in ("sdpa", "manual"), f"unknown attn_impl: {attn_impl}"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout
        self.attn_impl = attn_impl

        # One linear layer produces Q, K, and V all at once (faster than three
        # separate layers); we split the output into three chunks afterwards.
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)

        # Recombines the concatenated head outputs back into embed_dim.
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # A lower-triangular matrix of 1s/0s used to zero out "future" attention
        # scores. Registered as a buffer (not a parameter) so it moves with
        # .to(device) but isn't trained or saved as a learnable weight. Only
        # used by the "manual" attn_impl -- "sdpa" gets causality for free
        # via is_causal=True.
        causal_mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, block_size, block_size))

    def _split_heads(self, t, batch_size, seq_len):
        return t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _project_qkv(self, x):
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * embed_dim)
        q, k, v = qkv.split(embed_dim, dim=-1)
        return (
            self._split_heads(q, batch_size, seq_len),
            self._split_heads(k, batch_size, seq_len),
            self._split_heads(v, batch_size, seq_len),
        )

    def _attend(self, q, k, v, is_causal):
        """Runs softmax(QK^T/sqrt(d))V, dispatching to whichever attn_impl
        this module was built with. `is_causal=True` means q's positions are
        exactly k/v's trailing positions with nothing already-seen omitted
        (a fresh sequence or a cache prefill); `is_causal=False` means every
        key/value position is already causally before every query position
        (decoding one new token against an existing cache), so no masking
        is needed either way."""
        dropout_p = self.dropout_p if self.training else 0.0

        if self.attn_impl == "sdpa":
            return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=dropout_p)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if is_causal:
            seq_len, kv_len = q.shape[-2], k.shape[-2]
            attn_scores = attn_scores.masked_fill(
                self.causal_mask[:, :, :seq_len, :kv_len] == 0, float("-inf")
            )
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        return attn_weights @ v

    def _merge_heads(self, out, batch_size, seq_len, embed_dim):
        return out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

    def forward(self, x, kv_cache=None, use_cache=False):
        """
        kv_cache: optional (past_k, past_v), each (batch, num_heads, past_len,
        head_dim), from a previous call -- lets generation attend over
        already-seen tokens without recomputing their keys/values. Grows by
        concatenation every call (see forward_prealloc for the alternative,
        write-in-place cache used by Experiment 5's latency comparison).
        use_cache: if True, also return this call's (possibly cache-extended)
        keys/values so the caller can pass them into the next step.
        """
        batch_size, seq_len, embed_dim = x.shape
        q, k, v = self._project_qkv(x)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_cache = (k, v) if use_cache else None

        # is_causal=True only for a fresh sequence or a cache prefill (no
        # past yet); once there's a past, every cached position already
        # precedes every new query position, so nothing needs masking.
        out = self._attend(q, k, v, is_causal=(kv_cache is None))
        out = self._merge_heads(out, batch_size, seq_len, embed_dim)

        return self.resid_dropout(self.out_proj(out)), new_cache

    def forward_prealloc(self, x, cache_k, cache_v, pos):
        """
        Same computation as forward(), but the cache is a preallocated
        (batch, num_heads, block_size, head_dim) buffer (see
        GPT.init_prealloc_cache) that this call writes into in place at
        [pos : pos + seq_len], instead of growing a new tensor every step
        via torch.cat. The buffer is allocated once per generation call and
        reused for every step, trading that one-time allocation for
        avoiding per-step reallocation -- what Experiment 5 (bench_kvcache.py)
        measures against the original concat-based cache.

        Returns just the attention output (no cache object to thread through
        -- cache_k/cache_v are mutated in place and `pos` is tracked by the
        caller, GPT.generate_prealloc).
        """
        batch_size, seq_len, embed_dim = x.shape
        q, k, v = self._project_qkv(x)

        cache_k[:, :, pos : pos + seq_len, :] = k
        cache_v[:, :, pos : pos + seq_len, :] = v
        total_len = pos + seq_len
        k_full = cache_k[:, :, :total_len, :]
        v_full = cache_v[:, :, :total_len, :]

        # is_causal=True only during prefill (pos == 0, seq_len == prompt
        # length); a single-token decode step (seq_len == 1) needs no mask
        # since the one new query trivially comes after every cached key.
        out = self._attend(q, k_full, v_full, is_causal=(pos == 0 and seq_len > 1))
        out = self._merge_heads(out, batch_size, seq_len, embed_dim)
        return self.resid_dropout(self.out_proj(out))


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
    modern GPT convention -- it tends to train more stably than the original
    "post-norm" Transformer design.
    """

    def __init__(self, embed_dim, num_heads, block_size, dropout=0.1, attn_impl="sdpa"):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, block_size, dropout, attn_impl=attn_impl)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, dropout)

    def forward(self, x, kv_cache=None, use_cache=False):
        attn_out, new_cache = self.attn(self.ln1(x), kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, new_cache

    def forward_prealloc(self, x, cache_k, cache_v, pos):
        attn_out = self.attn.forward_prealloc(self.ln1(x), cache_k, cache_v, pos)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


def _sample_next_token(logits, temperature, top_k):
    """logits: (batch, vocab_size) -- the last position's logits. Shared by
    every generation path so the sampling logic isn't duplicated. A
    temperature of exactly 0 means greedy decoding (argmax), used by the
    latency benchmarks (bench_kvcache.py) where deterministic, comparable
    output across cache implementations matters more than diversity."""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None:
        # Zero out (via -inf) every logit outside the top-k, so sampling
        # can't pick a very unlikely token -- a common trick for keeping
        # generation coherent. Clamped to the vocab size so a caller's
        # default top_k (e.g. 50) doesn't crash torch.topk on a tokenizer
        # with a smaller vocabulary than that (small synthetic vocabs show
        # up throughout this codebase's tests and tiny ad-hoc runs).
        top_k = min(top_k, logits.shape[-1])
        v, _ = torch.topk(logits, top_k)
        logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _apply_eos(next_id, finished, eos_id):
    """Once a sequence has emitted eos_id, keep clamping its output to
    eos_id instead of letting it keep generating -- so a batch can mix
    sequences that finish early with ones that don't, and decode() just
    truncates at the first eos_id per row afterward. Returns (next_id,
    updated_finished)."""
    if eos_id is None:
        return next_id, finished
    next_id = torch.where(finished.unsqueeze(1), torch.full_like(next_id, eos_id), next_id)
    finished = finished | (next_id.squeeze(1) == eos_id)
    return next_id, finished


class GPT(nn.Module):
    """
    The full decoder-only model.

    Pipeline:
      1. Token embedding: look up a learned vector per token id.
      2. Positional embedding: look up a learned vector per position, since
         attention itself has no notion of order (it just weighs pairs of
         tokens) -- without this, "cat sat on mat" and "mat sat on cat" would
         look identical to the model.
      3. Add the two embeddings together and run through N transformer Blocks.
      4. Final LayerNorm, then a linear layer projecting back to vocab_size
         logits -- a score per vocabulary token for "what comes next."
    """

    def __init__(
        self,
        vocab_size,
        block_size=128,     # max sequence length the model can attend over
        embed_dim=128,      # size of each token's vector representation
        num_heads=4,
        num_layers=4,
        dropout=0.1,
        attn_impl="sdpa",
    ):
        super().__init__()
        self.block_size = block_size
        self.attn_impl = attn_impl

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, block_size, dropout, attn_impl=attn_impl)
            for _ in range(num_layers)
        ])

        self.ln_final = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: reuse the token embedding matrix as the output
        # projection. This is a standard GPT trick -- the intuition is that
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

    def forward(self, idx, targets=None, kv_cache=None, use_cache=False, ignore_index=IGNORE_INDEX):
        """
        idx: (batch, seq_len) integer token ids -- the full sequence during
             training/prefill, or just the new token(s) when decoding
             against a kv_cache.
        targets: optional (batch, seq_len) integer ids to compute loss against
                 (the ground-truth "next token" for each position). Any
                 position equal to `ignore_index` is excluded from the loss
                 -- data.py's EpochBatcher sets this for masked
                 (instruction-tuning header) positions.
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
            # against the actual next token, averaged over the whole batch
            # (positions where targets == ignore_index don't contribute).
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=ignore_index,
            )

        return logits, loss, new_cache

    def init_prealloc_cache(self, batch_size, device, dtype=torch.float32):
        """Allocates one (batch, num_heads, block_size, head_dim) key and
        value buffer per layer, for generate_prealloc. Call once per
        generation and reuse across every step of that generation."""
        head_dim = self.blocks[0].attn.head_dim
        num_heads = self.blocks[0].attn.num_heads
        shape = (batch_size, num_heads, self.block_size, head_dim)
        return [
            {"k": torch.zeros(shape, device=device, dtype=dtype), "v": torch.zeros(shape, device=device, dtype=dtype)}
            for _ in self.blocks
        ]

    def forward_prealloc(self, idx, cache, pos):
        """One step of the preallocated-cache path: idx is (batch, seq_len)
        new tokens (the whole prompt on the first, prefill call; one token
        per call after that), cache is the list returned by
        init_prealloc_cache, pos is the number of tokens already written
        into the cache before this call. Returns logits for the positions
        in `idx` only (not the whole cached history)."""
        batch_size, seq_len = idx.shape
        assert pos + seq_len <= self.block_size, (
            f"sequence length {pos + seq_len} exceeds block_size {self.block_size}"
        )

        positions = torch.arange(pos, pos + seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.dropout(x)

        for i, block in enumerate(self.blocks):
            x = block.forward_prealloc(x, cache[i]["k"], cache[i]["v"], pos)

        x = self.ln_final(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=False, eos_id=None):
        """
        Autoregressive generation: repeatedly predict the next token, append
        it, and feed it back in. This is why the model is causal -- at
        inference time there's no "future" to peek at anyway.

        use_cache=False (default): recomputes attention over the whole
        growing sequence every step (truncated to the last block_size
        tokens if it overflows) -- simple, and O(n^2) in tokens generated.

        use_cache=True: keeps each layer's past keys/values around (see
        CausalSelfAttention) so every step only attends the one new token
        against the cached past -- O(n) overall, the standard KV-cache
        speedup. There's no cache eviction here, so prompt length +
        max_new_tokens must fit within block_size.

        eos_id: if given, a batch row stops "really" generating once it
        emits eos_id (later positions are clamped to eos_id -- see
        _apply_eos) and the whole call returns early once every row in the
        batch has finished, rather than always running max_new_tokens steps.
        Decode with tokenizer.decode(ids[: ids.index(eos_id)]) per row to
        drop the clamped trailing eos_id repeats.
        """
        self.eval()
        batch_size = idx.shape[0]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=idx.device) if eos_id is not None else None

        if not use_cache:
            for _ in range(max_new_tokens):
                # Only the last block_size tokens fit in the model's context window.
                idx_cond = idx[:, -self.block_size:]
                logits, _, _ = self(idx_cond)
                next_id = _sample_next_token(logits[:, -1, :], temperature, top_k)
                next_id, finished = _apply_eos(next_id, finished, eos_id)
                idx = torch.cat([idx, next_id], dim=1)
                if finished is not None and finished.all():
                    break
            return idx

        # Prefill: run the whole prompt once to seed the per-layer kv_cache.
        logits, _, cache = self(idx, use_cache=True)
        next_id = _sample_next_token(logits[:, -1, :], temperature, top_k)
        next_id, finished = _apply_eos(next_id, finished, eos_id)
        idx = torch.cat([idx, next_id], dim=1)

        # Decode: each step only feeds the one newest token; the cache holds
        # everything before it.
        for _ in range(max_new_tokens - 1):
            if finished is not None and finished.all():
                break
            logits, _, cache = self(next_id, kv_cache=cache, use_cache=True)
            next_id = _sample_next_token(logits[:, -1, :], temperature, top_k)
            next_id, finished = _apply_eos(next_id, finished, eos_id)
            idx = torch.cat([idx, next_id], dim=1)

        return idx

    @torch.no_grad()
    def generate_prealloc(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
        """
        Generation backed by the preallocated, write-in-place KV cache
        (forward_prealloc) instead of the growing-by-concatenation cache
        `generate(use_cache=True)` uses. Produces the same distribution over
        outputs as the other two generation paths (see the model.py smoke
        test); exists so Experiment 5 can measure whether avoiding per-step
        reallocation actually matters at these model sizes, separately from
        whether caching helps at all.
        """
        self.eval()
        batch_size = idx.shape[0]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=idx.device) if eos_id is not None else None
        cache = self.init_prealloc_cache(batch_size, idx.device, dtype=self.token_embedding.weight.dtype)

        prompt_len = idx.shape[1]
        logits = self.forward_prealloc(idx, cache, pos=0)
        pos = prompt_len
        next_id = _sample_next_token(logits[:, -1, :], temperature, top_k)
        next_id, finished = _apply_eos(next_id, finished, eos_id)
        idx = torch.cat([idx, next_id], dim=1)

        for _ in range(max_new_tokens - 1):
            if finished is not None and finished.all():
                break
            logits = self.forward_prealloc(next_id, cache, pos=pos)
            pos += 1
            next_id = _sample_next_token(logits[:, -1, :], temperature, top_k)
            next_id, finished = _apply_eos(next_id, finished, eos_id)
            idx = torch.cat([idx, next_id], dim=1)

        return idx


if __name__ == "__main__":
    # Quick smoke test with a tiny vocab, no training -- just checking shapes
    # and that a forward pass + generation run end to end, for both
    # attention implementations.
    vocab_size = 1000

    for attn_impl in ("manual", "sdpa"):
        print(f"\n--- attn_impl={attn_impl} ---")
        model = GPT(vocab_size, block_size=32, embed_dim=64, num_heads=4, num_layers=2, attn_impl=attn_impl)
        model.eval()

        dummy_input = torch.randint(0, vocab_size, (2, 16))   # batch of 2, seq_len 16
        dummy_targets = torch.randint(0, vocab_size, (2, 16))

        logits, loss, _ = model(dummy_input, dummy_targets)
        print("logits shape:", logits.shape)  # (2, 16, vocab_size)
        print("loss:", loss.item())

        generated = model.generate(dummy_input[:, :1], max_new_tokens=10)
        print("generated shape:", generated.shape)  # (2, 11)

        # KV-cache sanity check: caching only changes how attention is computed
        # (incrementally vs from scratch each step), not the math, so with the
        # same seed the sampled tokens should match exactly across all three
        # generation paths (no cache, growing cache, preallocated cache).
        torch.manual_seed(0)
        gen_nocache = model.generate(dummy_input[:, :1], max_new_tokens=10, use_cache=False)
        torch.manual_seed(0)
        gen_cache = model.generate(dummy_input[:, :1], max_new_tokens=10, use_cache=True)
        torch.manual_seed(0)
        gen_prealloc = model.generate_prealloc(dummy_input[:, :1], max_new_tokens=10)
        print("cache == no-cache:", torch.equal(gen_nocache, gen_cache))
        print("prealloc == no-cache:", torch.equal(gen_nocache, gen_prealloc))

        # ignore_index sanity check: a target of IGNORE_INDEX should be
        # excluded from the loss -- masking out an entire batch row should
        # match computing the loss over just the other row alone.
        targets_masked = dummy_targets.clone()
        targets_masked[1, :] = IGNORE_INDEX
        _, loss_masked, _ = model(dummy_input, targets_masked)
        _, loss_row0_only, _ = model(dummy_input[:1], dummy_targets[:1])
        print("masked-row loss == single-row loss:", torch.allclose(loss_masked, loss_row0_only, atol=1e-5))

        # eos_id sanity check: with a tiny 2-token vocab, a random-init model
        # emits "eos" (token 1) roughly every other step, so generation
        # should almost always stop well before max_new_tokens -- and every
        # token after the first eos in a row should be clamped to eos_id.
        tiny_model = GPT(2, block_size=64, embed_dim=32, num_heads=2, num_layers=2, attn_impl=attn_impl)
        tiny_model.eval()
        torch.manual_seed(0)
        eos_id = 1
        start = torch.zeros(4, 1, dtype=torch.long)
        gen_eos = tiny_model.generate(start, max_new_tokens=50, use_cache=True, eos_id=eos_id)
        stopped_early = gen_eos.shape[1] < 51
        rows_with_eos = (gen_eos == eos_id).any(dim=1)
        clamped_correctly = all(
            (row[(row == eos_id).float().argmax():] == eos_id).all().item()
            for row, has_eos in zip(gen_eos, rows_with_eos) if has_eos.item()
        )
        print(f"eos stopped generation early: {stopped_early} (final length {gen_eos.shape[1]} of 51)")
        print(f"every row past its first eos stays clamped to eos_id: {clamped_correctly}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nparameters: {num_params:,}")
