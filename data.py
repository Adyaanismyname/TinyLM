"""
Seeded, reproducible batch sampling over token-id streams.

The old approach (train.py's original get_batch) drew a fresh uniformly
random block_size-window on every step, with no seed anywhere. That has two
problems for anything you want to *compare*:

  1. Coverage is uneven -- with replacement, some stretches of text get
     sampled many times in a run while others are never seen at all.
  2. Comparisons aren't paired -- LoRA r=4 and full fine-tuning saw
     completely different sequences of batches, so any gap between them is
     partly "different training method" and partly "different random data",
     with no way to separate the two.

This module fixes both: a TokenStream partitions its data into fixed-length,
non-overlapping windows once, then hands them out shuffled by an explicit
seed. One epoch = every window exactly once. Two streams (or two training
runs) built with the same seed produce byte-identical batches in identical
order, so "same data, different method" comparisons are actually true.

Works over two kinds of storage:
  - a uint16 memmap on disk (real data, doesn't need to fit in RAM)
  - a plain in-memory array/tensor (small runs, tests)
"""

import numpy as np
import torch


class TokenStream:
    """
    A 1D array of token ids, sliced into non-overlapping (block_size + 1)
    windows (block_size input tokens + 1 shifted-by-one target token, same
    convention as the old train.py get_batch).

    `data` just needs to support `len()` and integer-slice indexing that
    returns something numpy can turn into an array -- a numpy array, a
    memmap, or a 1D tensor's `.numpy()` all work.

    `mask`, if given, is a same-length 0/1 array (e.g. uint8): position i's
    mask value gates whether *token i's own loss* should count in training
    (1 = compute loss here, 0 = ignore, e.g. an instruction-tuning header
    that the model isn't asked to predict). It's applied to the batch's
    target array in EpochBatcher.next_batch, not here.
    """

    def __init__(self, data, block_size, mask=None):
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        if mask is not None and len(mask) != len(data):
            raise ValueError(
                f"mask length {len(mask)} does not match data length {len(data)}"
            )

        self.data = data
        self.mask = mask
        self.block_size = block_size
        self.num_windows = max(0, (len(data) - 1) // block_size)

        if self.num_windows == 0:
            raise ValueError(
                f"stream has only {len(data)} tokens -- too short for even one "
                f"block_size={block_size} window (need at least block_size + 1)"
            )

    def __len__(self):
        return self.num_windows

    def window(self, window_idx):
        """Returns (x, y[, mask_y]) for one window -- input, shifted-by-one
        target, and (if a mask was given) the mask aligned to the target
        positions (mask[i] gates whether target token i counts in the loss)."""
        start = window_idx * self.block_size
        x = np.asarray(self.data[start : start + self.block_size])
        y = np.asarray(self.data[start + 1 : start + 1 + self.block_size])
        if self.mask is None:
            return x, y, None
        m = np.asarray(self.mask[start + 1 : start + 1 + self.block_size])
        return x, y, m

    def epoch_batches(self, batch_size, seed, drop_last=True):
        """
        Yields (x, y, mask_or_None) numpy-array batches covering one full
        epoch -- every window exactly once, in an order determined entirely
        by `seed` (np.random.default_rng(seed).permutation).

        drop_last=True (the default) drops a final short batch so every
        yielded batch has exactly `batch_size` rows, which is what a
        training loop with a fixed-shape model wants. Set it False if you
        need every window represented (e.g. exhaustive evaluation).
        """
        order = np.random.default_rng(seed).permutation(self.num_windows)

        num_full_batches = len(order) // batch_size
        num_batches = num_full_batches if drop_last else -(-len(order) // batch_size)

        for b in range(num_batches):
            idxs = order[b * batch_size : (b + 1) * batch_size]
            if len(idxs) == 0:
                continue
            xs, ys, ms = [], [], []
            for i in idxs:
                x, y, m = self.window(int(i))
                xs.append(x)
                ys.append(y)
                if m is not None:
                    ms.append(m)
            batch_mask = np.stack(ms) if ms else None
            yield np.stack(xs), np.stack(ys), batch_mask


class EpochBatcher:
    """
    The object train_model actually calls: an infinite, step-based iterator
    over a TokenStream's epochs.

    Each epoch uses seed = base_seed + epoch_index, so epoch 0, 1, 2, ...
    are each a *different* shuffle of the windows (not the same order
    repeated), while the whole sequence of batches across every epoch is
    still fully determined by base_seed -- rerun with the same base_seed and
    you get the identical sequence of batches, forever.

    next_batch() returns (x, y) tensors on `device`, with masked-out target
    positions (mask == 0) set to `ignore_index` so they're excluded from
    cross-entropy automatically (model.py's forward passes ignore_index
    through to F.cross_entropy).
    """

    def __init__(self, stream: TokenStream, batch_size, seed, device, ignore_index=-100):
        if batch_size > len(stream):
            raise ValueError(
                f"batch_size={batch_size} exceeds the stream's {len(stream)} total "
                f"windows -- every batch would need to drop_last down to nothing"
            )
        self.stream = stream
        self.batch_size = batch_size
        self.base_seed = seed
        self.device = device
        self.ignore_index = ignore_index
        self.epoch = 0
        self._iter = None
        self._advance_epoch()

    def _advance_epoch(self):
        self._iter = self.stream.epoch_batches(self.batch_size, seed=self.base_seed + self.epoch)
        self.epoch += 1

    def next_batch(self):
        try:
            x, y, m = next(self._iter)
        except StopIteration:
            self._advance_epoch()
            x, y, m = next(self._iter)

        x = torch.as_tensor(x.astype(np.int64), device=self.device)
        y = torch.as_tensor(y.astype(np.int64), device=self.device)
        if m is not None:
            y = y.clone()
            y[torch.as_tensor(m == 0, device=self.device)] = self.ignore_index
        return x, y


def load_memmap(path, dtype=np.uint16):
    return np.memmap(path, dtype=dtype, mode="r")


def stream_from_bin(token_path, block_size, mask_path=None, dtype=np.uint16):
    """Builds a TokenStream backed by on-disk memmaps -- the normal path for
    real training data written by prepare_data.py."""
    data = load_memmap(token_path, dtype=dtype)
    mask = load_memmap(mask_path, dtype=np.uint8) if mask_path else None
    return TokenStream(data, block_size, mask=mask)


def stream_from_array(token_ids, block_size):
    """Builds a TokenStream over an in-memory sequence (list/np.array/1D
    tensor) of token ids -- for small ad-hoc runs and tests where writing a
    .bin file to disk first would be pointless."""
    if torch.is_tensor(token_ids):
        arr = token_ids.numpy()
    else:
        arr = np.asarray(token_ids)
    return TokenStream(arr, block_size)
