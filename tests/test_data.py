"""
Tests for data.py -- the seeded epoch-based batch sampler that everything
else (train.py, the E1/E2/E3 experiments) is built on. These are checked
first and in isolation because a bug here would silently corrupt every
downstream comparison.
"""

import numpy as np
import pytest
import torch

from data import EpochBatcher, TokenStream, stream_from_array


def make_stream(n=1000, block_size=10):
    data = np.arange(n, dtype=np.int64)  # token i == value i, easy to check
    return TokenStream(data, block_size)


def test_window_shapes_and_shift():
    stream = make_stream(n=100, block_size=10)
    x, y, m = stream.window(0)
    assert x.tolist() == list(range(0, 10))
    assert y.tolist() == list(range(1, 11))
    assert m is None


def test_too_short_raises():
    with pytest.raises(ValueError):
        TokenStream(np.arange(5), block_size=10)


def test_mask_length_mismatch_raises():
    with pytest.raises(ValueError):
        TokenStream(np.arange(100), block_size=10, mask=np.ones(50))


def test_epoch_covers_every_window_exactly_once():
    stream = make_stream(n=1000, block_size=10)  # 99 windows
    seen = []
    for x, y, m in stream.epoch_batches(batch_size=9, seed=0, drop_last=False):
        seen.extend((x[i, 0]) for i in range(x.shape[0]))
    # window i starts at token i*block_size, so its first token identifies it
    expected_starts = {i * 10 for i in range(len(stream))}
    assert set(seen) == expected_starts
    assert len(seen) == len(stream)  # no duplicates, no drops


def test_drop_last_true_drops_partial_batch():
    stream = make_stream(n=1000, block_size=10)  # 99 windows
    total = sum(x.shape[0] for x, y, m in stream.epoch_batches(batch_size=10, seed=0, drop_last=True))
    assert total == (99 // 10) * 10 == 90  # last 9 windows (99 - 90) dropped


def test_same_seed_same_batches():
    stream_a = make_stream(n=5000, block_size=16)
    stream_b = make_stream(n=5000, block_size=16)

    batches_a = list(stream_a.epoch_batches(batch_size=8, seed=42))
    batches_b = list(stream_b.epoch_batches(batch_size=8, seed=42))

    assert len(batches_a) == len(batches_b)
    for (xa, ya, _), (xb, yb, _) in zip(batches_a, batches_b):
        assert np.array_equal(xa, xb)
        assert np.array_equal(ya, yb)


def test_different_seeds_different_order():
    stream = make_stream(n=5000, block_size=16)
    batches_1 = list(stream.epoch_batches(batch_size=8, seed=1))
    batches_2 = list(stream.epoch_batches(batch_size=8, seed=2))
    # first batch's first window almost certainly differs
    assert batches_1[0][0][0, 0] != batches_2[0][0][0, 0]


def test_successive_epochs_reshuffle_not_repeat():
    stream = make_stream(n=2000, block_size=10)  # 199 windows
    batcher = EpochBatcher(stream, batch_size=9, seed=0, device=torch.device("cpu"))

    epoch0_firsts = []
    for _ in range(199 // 9):
        x, y = batcher.next_batch()
        epoch0_firsts.append(x[0, 0].item())
    epoch0_epoch_field = batcher.epoch

    x, y = batcher.next_batch()  # triggers epoch 2 (0-indexed epoch becomes 1 -> 2)
    assert batcher.epoch > epoch0_epoch_field
    # the reshuffled epoch's first window shouldn't just replay epoch 0's order
    # (not a hard guarantee, but overwhelmingly likely with a real RNG)
    assert x[0, 0].item() != epoch0_firsts[0]


def test_epoch_batcher_reproducible_across_instances():
    stream_a = make_stream(n=3000, block_size=12)
    stream_b = make_stream(n=3000, block_size=12)
    device = torch.device("cpu")

    batcher_a = EpochBatcher(stream_a, batch_size=6, seed=7, device=device)
    batcher_b = EpochBatcher(stream_b, batch_size=6, seed=7, device=device)

    for _ in range(50):
        xa, ya = batcher_a.next_batch()
        xb, yb = batcher_b.next_batch()
        assert torch.equal(xa, xb)
        assert torch.equal(ya, yb)


def test_mask_sets_ignore_index_on_target():
    n = 200
    data = np.arange(n, dtype=np.int64)
    mask = np.ones(n, dtype=np.uint8)
    mask[50:60] = 0  # a "header" span whose loss should be ignored
    stream = TokenStream(data, block_size=10, mask=mask)
    batcher = EpochBatcher(stream, batch_size=1, seed=0, device=torch.device("cpu"), ignore_index=-100)

    found_ignored = False
    for _ in range(len(stream)):
        x, y = batcher.next_batch()
        # target token value == y itself (since data[i] == i); masked ones become -100
        for val in y[0].tolist():
            if val == -100:
                found_ignored = True
            else:
                assert 0 <= val < n
    assert found_ignored


def test_batch_size_larger_than_stream_raises():
    stream = make_stream(n=100, block_size=10)  # 9 windows
    with pytest.raises(ValueError):
        EpochBatcher(stream, batch_size=20, seed=0, device=torch.device("cpu"))


def test_stream_from_array_accepts_list_and_tensor():
    ids = list(range(500))
    s1 = stream_from_array(ids, block_size=10)
    s2 = stream_from_array(torch.tensor(ids, dtype=torch.long), block_size=10)
    assert len(s1) == len(s2) == 49


def test_memmap_backed_stream(tmp_path):
    n = 10_000
    arr = np.arange(n, dtype=np.uint16) % 1000  # fits in uint16, wraps like real token ids
    path = tmp_path / "toy.bin"
    arr.tofile(path)

    from data import stream_from_bin

    stream = stream_from_bin(str(path), block_size=32)
    assert len(stream) == (n - 1) // 32

    batcher = EpochBatcher(stream, batch_size=4, seed=123, device=torch.device("cpu"))
    x, y = batcher.next_batch()
    assert x.shape == (4, 32)
    assert x.dtype == torch.int64
