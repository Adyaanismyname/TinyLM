"""
Tests for bench_train.py / bench_kvcache.py's pure helper functions. The
actual benchmarks are exercised via --quick smoke tests (they need a real
device to time), not unit tests here.
"""

import bench_kvcache
import bench_train
from lora import trainable_parameters
from model import GPT


def test_bench_train_build_model_lora_has_fewer_trainable_params():
    device = "cpu"
    full = bench_train.build_model(128, "full", None, device)
    lora = bench_train.build_model(128, "lora", 8, device)

    full_trainable, full_total = trainable_parameters(full)
    lora_trainable, lora_total = trainable_parameters(lora)

    assert full_trainable == full_total  # full fine-tuning: everything trainable
    assert lora_trainable < lora_total   # LoRA: only a slice trainable
    assert lora_total > full_total       # LoRA adds A/B parameters on top


def test_bench_train_build_model_head_dim_is_64():
    for width in bench_train.WIDTHS:
        model = bench_train.build_model(width, "full", None, "cpu")
        assert model.blocks[0].attn.head_dim == 64


def test_analytical_kv_bytes_scales_with_batch_and_seq_len():
    base = bench_kvcache.analytical_kv_bytes(batch_size=1, seq_len=100)
    double_batch = bench_kvcache.analytical_kv_bytes(batch_size=2, seq_len=100)
    double_seq = bench_kvcache.analytical_kv_bytes(batch_size=1, seq_len=200)

    assert double_batch == base * 2
    assert double_seq == base * 2


def test_analytical_kv_bytes_matches_hand_computation():
    # 2 (K+V) * layers * seq_len * embed_dim * batch * 4 bytes
    expected = 2 * bench_kvcache.NUM_LAYERS * 50 * bench_kvcache.EMBED_DIM * 3 * 4
    assert bench_kvcache.analytical_kv_bytes(batch_size=3, seq_len=50) == expected
