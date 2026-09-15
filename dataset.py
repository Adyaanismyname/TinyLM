import re
import unicodedata
from BPE import train_bpe, BPETokenizer


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



def build_dataset(vocab_size=1000, num_train=8000, num_val=2000):
    """
    Loads TinyStories, preprocesses it, trains a BPE tokenizer on the
    training split, and returns everything needed to train a model:

        tokenizer      - a BPETokenizer built from the learned vocab/merges
        train_token_ids - list of token-id sequences (one per training story)
        val_token_ids   - list of token-id sequences (one per validation story)
    """
    from datasets import load_dataset

    dataset = load_dataset("roneneldan/TinyStories")

    train_texts = [preprocess_text(t) for t in dataset["train"]["text"][:num_train]]
    val_texts = [preprocess_text(t) for t in dataset["validation"]["text"][:num_val]]
    
    # train_texts = [preprocess_text(t) for t in dataset["train"]["text"]]
    # val_texts = [preprocess_text(t) for t in dataset["validation"]["text"]]

    # train_bpe mutates its input in place, replacing each story's char list
    # with its final merged tokens — so train_stories doubles as the encoded
    # training set once training finishes.
    # Reserve one vocab slot for the <unk> token (added below by
    # BPETokenizer) so the final tokenizer vocab still totals `vocab_size`.
    train_stories = [list(t) for t in train_texts]
    vocab, merges = train_bpe(train_stories, vocab_size=vocab_size - 1)

    tokenizer = BPETokenizer(vocab, merges)

    train_token_ids = [
        [tokenizer.token_to_id[tok] for tok in story]
        for story in train_stories
    ]
    val_token_ids = [tokenizer.encode(text) for text in val_texts]

    return tokenizer, train_token_ids, val_token_ids


if __name__ == "__main__":
    tokenizer, train_token_ids, val_token_ids = build_dataset()

    print("vocab size:", len(tokenizer.vocab))
    print("num merges:", len(tokenizer.merges))
    print("sample vocab:", list(tokenizer.vocab)[:50])
    print("sample merges:", tokenizer.merges[:20])
    print("train stories:", len(train_token_ids))
    print("val stories:", len(val_token_ids))


