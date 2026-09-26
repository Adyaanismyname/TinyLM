"""
Tests for dataset.py's pure text-processing pieces (Instruct example
parsing, prompt formatting, dev split) -- no network access, so these run
without hitting HuggingFace. Live-download smoke checks against the real
datasets live in tests/test_prepare_data_live.py (opt-in, requires network).
"""

import pytest

from dataset import (
    _parse_instruct_example,
    format_instruct_prompt,
    load_tinystories_instruct_examples,
    preprocess_text,
    split_dev,
)


def test_preprocess_collapses_whitespace_and_trims():
    assert preprocess_text("  Hello   world.\n\nNew line.  ") == "Hello world. New line."


def test_parse_instruct_example_all_fields():
    lines = [
        "Features: Dialogue, BadEnding",
        "Words: quit, oak, gloomy",
        "Summary: Sara and Ben find a gloomy oak tree.",
        "Random sentence: The wind howled.",
        "Story:",
        "",
        "Once upon a time, Sara and Ben saw a gloomy oak tree.",
        "They decided to quit looking and go home.",
    ]
    parsed = _parse_instruct_example(lines)
    assert parsed["features"] == ["Dialogue", "BadEnding"]
    assert parsed["words"] == ["quit", "oak", "gloomy"]
    assert parsed["summary"] == "Sara and Ben find a gloomy oak tree."
    assert parsed["random_sentence"] == "The wind howled."
    assert "Once upon a time" in parsed["story"]
    assert "quit looking" in parsed["story"]


def test_parse_instruct_example_missing_optional_fields():
    lines = [
        "Words: cat, dog",
        "Summary: A cat meets a dog.",
        "Story: The cat met the dog and they were friends.",
    ]
    parsed = _parse_instruct_example(lines)
    assert parsed["features"] == []
    assert parsed["random_sentence"] is None
    assert parsed["story"] == "The cat met the dog and they were friends."


def test_parse_instruct_example_story_inline_after_colon():
    lines = ["Words: sun", "Story: The sun was bright."]
    parsed = _parse_instruct_example(lines)
    assert parsed["story"] == "The sun was bright."


def test_format_instruct_prompt_header_ends_with_story_marker():
    example = {
        "words": ["cat", "dog"],
        "features": ["Dialogue"],
        "summary": "A cat and a dog play.",
        "random_sentence": None,
        "story": "The cat and the dog played all day.",
    }
    header, story = format_instruct_prompt(example)
    assert header.endswith("Story:")
    assert "Words: cat, dog" in header
    assert "Features: Dialogue" in header
    assert story.startswith(" ")  # deliberate leading space joining header + story
    assert "played all day" in story


def test_format_instruct_prompt_omits_absent_fields():
    example = {"words": [], "features": [], "summary": "", "random_sentence": None, "story": "A story."}
    header, story = format_instruct_prompt(example)
    assert header == "Story:"


def test_split_dev_no_overlap_and_correct_sizes():
    items = list(range(100))
    train_pool, dev = split_dev(items, num_dev=20, seed=0)
    assert len(dev) == 20
    assert len(train_pool) == 80
    assert set(train_pool).isdisjoint(dev)
    assert set(train_pool) | set(dev) == set(items)


def test_split_dev_deterministic():
    items = list(range(50))
    a = split_dev(items, num_dev=10, seed=42)
    b = split_dev(items, num_dev=10, seed=42)
    assert a == b


def test_split_dev_too_large_raises():
    with pytest.raises(ValueError):
        split_dev(list(range(5)), num_dev=10, seed=0)


class _FakeRows:
    """Mimics enough of a HF Dataset's row-iteration protocol (iterating
    yields one {"text": ...} dict per row) for load_tinystories_instruct_
    examples' row-by-row loop, without needing the real `datasets` package
    machinery in these offline tests."""

    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return ({"text": r} for r in self._rows)


def _fake_instruct_dataset(monkeypatch, rows):
    fake_dataset = {"train": _FakeRows(rows)}

    def fake_load_dataset(name):
        assert name == "roneneldan/TinyStoriesInstruct"
        return fake_dataset

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)


def test_load_tinystories_instruct_examples_groups_by_endoftext(monkeypatch):
    rows = [
        "Words: cat, dog",
        "Summary: A cat meets a dog.",
        "Story: The cat met the dog.",
        "<|endoftext|>",
        "Words: sun, moon",
        "Summary: The sun and moon.",
        "Story: The sun and moon shared the sky.",
        "<|endoftext|>",
    ]
    _fake_instruct_dataset(monkeypatch, rows)
    examples = load_tinystories_instruct_examples("train")
    assert len(examples) == 2
    assert examples[0]["words"] == ["cat", "dog"]
    assert examples[1]["words"] == ["sun", "moon"]
    assert "shared the sky" in examples[1]["story"]


def test_load_tinystories_instruct_examples_drops_leading_boundary_fragment(monkeypatch):
    """Reproduces the real TinyStoriesInstruct validation split's first row:
    a story continuation with no header before it, no "Story:" line at all,
    ending at the first <|endoftext|> -- should be dropped rather than kept
    as an example with an empty story."""
    rows = [
        'ooks. He said, "Yes, please. Read, please."',            # orphaned story tail
        "His mom picked up the book and continued to read.",       # more orphaned tail
        "Summary: Sam's tower of books falls down.",               # a field, but no Story: ever appears
        "<|endoftext|>",
        "Words: cat, dog",
        "Story: The cat met the dog.",
        "<|endoftext|>",
    ]
    _fake_instruct_dataset(monkeypatch, rows)
    examples = load_tinystories_instruct_examples("train")
    assert len(examples) == 1
    assert examples[0]["words"] == ["cat", "dog"]


def test_load_tinystories_instruct_examples_respects_num_examples(monkeypatch):
    rows = []
    for i in range(5):
        rows += [f"Words: w{i}", f"Story: story {i}.", "<|endoftext|>"]
    _fake_instruct_dataset(monkeypatch, rows)
    examples = load_tinystories_instruct_examples("train", num_examples=2)
    assert len(examples) == 2
