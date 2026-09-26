"""
Loads and preprocesses TinyStories and TinyStoriesInstruct.

This module only loads and structures text -- it doesn't touch the
tokenizer or write any files. prepare_data.py is the pipeline that calls
these functions, trains the tokenizer, and writes the .bin files everything
else trains on.
"""

import random
import re
import unicodedata


def preprocess_text(text):
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters except whitespace
    text = "".join(
        char for char in text
        if char.isprintable() or char.isspace()
    )

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_tinystories_texts(split, num_examples=None):
    """
    Loads a TinyStories split ("train" or "validation"), preprocessed.
    num_examples=None loads the whole split; an int takes a fixed prefix
    (not shuffled, so it's reproducible without a seed).
    """
    from datasets import load_dataset

    dataset = load_dataset("roneneldan/TinyStories")
    texts = dataset[split]["text"]
    if num_examples is not None:
        texts = texts[:num_examples]
    return [preprocess_text(t) for t in texts]


_INSTRUCT_FIELD_RE = re.compile(r"^(Words|Features|Summary|Random sentence|Story):\s*(.*)$")


def _parse_instruct_example(lines):
    """
    Parses one TinyStoriesInstruct example's accumulated lines into its
    fields. Each line is "Field: value", except the story, which may span
    several lines after "Story:" (everything from there to the end of the
    example is story text).
    """
    fields = {"words": [], "features": [], "summary": "", "random_sentence": None, "story": ""}
    story_lines = []
    in_story = False

    for line in lines:
        if not in_story:
            match = _INSTRUCT_FIELD_RE.match(line)
            if match:
                key, value = match.group(1), match.group(2)
                if key == "Words":
                    fields["words"] = [w.strip() for w in value.split(",") if w.strip()]
                elif key == "Features":
                    fields["features"] = [f.strip() for f in value.split(",") if f.strip()]
                elif key == "Summary":
                    fields["summary"] = value.strip()
                elif key == "Random sentence":
                    fields["random_sentence"] = value.strip()
                elif key == "Story":
                    in_story = True
                    if value.strip():
                        story_lines.append(value)
                continue
            # A blank/unrecognized line before "Story:" -- just a separator, skip it.
        else:
            story_lines.append(line)

    fields["story"] = "\n".join(story_lines).strip()
    return fields


def load_tinystories_instruct_examples(split, num_examples=None):
    """
    Loads TinyStoriesInstruct. The HuggingFace dataset stores one *line* of
    text per row, with examples separated by a row that reads
    "<|endoftext|>" -- this groups consecutive rows back into examples and
    parses each into its instruction fields.

    The very first accumulated "example" of a split is sometimes a
    boundary artifact rather than a real example: row 0 can be the tail end
    of a story whose header lines aren't present in this split's row array
    at all (confirmed against the real dataset -- TinyStoriesInstruct's
    validation split starts mid-sentence, with no "Words:"/"Story:" line
    before its first "<|endoftext|>"). _parse_instruct_example has nothing
    to attach that orphaned text to, so it comes back with an empty story;
    such examples are dropped here since they're unusable for either
    pretraining-style text or instruction fine-tuning either way.

    Returns a list of dicts: {"words", "features", "summary",
    "random_sentence", "story"}. `summary`/`random_sentence`/`story` are
    preprocessed with preprocess_text; `words`/`features` are left as short
    lists of raw strings (they're prompt-construction inputs, not prose).
    """
    from datasets import load_dataset

    dataset = load_dataset("roneneldan/TinyStoriesInstruct")
    # Iterate row-by-row instead of materializing dataset[split]["text"] (a
    # 21.7M-element Python list for the train split) up front -- with
    # num_examples set, the loop below can stop as soon as it has enough,
    # never touching most of the split.
    rows = dataset[split]

    examples = []
    current = []
    for row in rows:
        line = row["text"]
        if line.strip() == "<|endoftext|>":
            if current:
                examples.append(_parse_instruct_example(current))
                current = []
            if num_examples is not None and len(examples) >= num_examples:
                break
        else:
            current.append(line)
    if current and (num_examples is None or len(examples) < num_examples):
        examples.append(_parse_instruct_example(current))

    for ex in examples:
        ex["summary"] = preprocess_text(ex["summary"])
        if ex["random_sentence"]:
            ex["random_sentence"] = preprocess_text(ex["random_sentence"])
        ex["story"] = preprocess_text(ex["story"])

    before = len(examples)
    examples = [ex for ex in examples if ex["story"]]
    dropped = before - len(examples)
    if dropped:
        print(f"load_tinystories_instruct_examples({split}): dropped {dropped} example(s) with no story text")

    return examples


def format_instruct_prompt(example):
    """
    Renders one parsed Instruct example back into the "header + story" text
    the model actually trains on, in the same field order TinyStoriesInstruct
    itself uses. Returns (header_text, story_text) separately so callers
    (prepare_data.py) can mark header tokens as loss-masked and story tokens
    as not.
    """
    lines = []
    if example["features"]:
        lines.append(f"Features: {', '.join(example['features'])}")
    if example["words"]:
        lines.append(f"Words: {', '.join(example['words'])}")
    if example["summary"]:
        lines.append(f"Summary: {example['summary']}")
    if example["random_sentence"]:
        lines.append(f"Random sentence: {example['random_sentence']}")
    header = preprocess_text(" ".join(lines) + " Story:")
    story = " " + example["story"] if example["story"] else ""
    return header, story


def split_dev(items, num_dev, seed=0):
    """
    Splits `items` into (train_pool, dev), moving a random `num_dev`-sized
    sample out for dev (e.g. picking a fine-tuning learning rate) so dev
    never silently overlaps whatever subset of train_pool an experiment
    later trains on. Deterministic given `seed`; the official validation
    split is untouched by this and used separately as the held-out test set.
    """
    if num_dev > len(items):
        raise ValueError(f"num_dev={num_dev} exceeds available items={len(items)}")
    indices = list(range(len(items)))
    random.Random(seed).shuffle(indices)
    dev_idx = set(indices[:num_dev])
    dev = [items[i] for i in sorted(dev_idx)]
    train_pool = [items[i] for i in range(len(items)) if i not in dev_idx]
    return train_pool, dev


if __name__ == "__main__":
    print("loading a small TinyStories sample...")
    texts = load_tinystories_texts("train", num_examples=20)
    print(f"loaded {len(texts)} stories, e.g.:\n  {texts[0][:120]}...")

    train_pool, dev = split_dev(texts, num_dev=5, seed=0)
    print(f"split_dev: {len(train_pool)} train / {len(dev)} dev (no overlap: {set(train_pool).isdisjoint(dev)})")

    print("\nloading a small TinyStoriesInstruct sample...")
    examples = load_tinystories_instruct_examples("train", num_examples=5)
    print(f"loaded {len(examples)} examples")
    for ex in examples[:2]:
        header, story = format_instruct_prompt(ex)
        print(f"  words={ex['words']} features={ex['features']}")
        print(f"  header: {header[:100]}...")
        print(f"  story:  {story[:100]}...")
