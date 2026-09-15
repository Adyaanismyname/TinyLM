def merge_pair(tokens, pair):
    new_tokens = []
    
    i = 0 
    
    while i < (len(tokens) - 1):
        if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            new_tokens.append(pair[0] + pair[1])
            i+=2
        else:
            new_tokens.append(tokens[i])
            i+=1
    
    if i < len(tokens):
        new_tokens.append(tokens[i])

    return new_tokens

def get_pair_counts(tokens):
    pair_counts = {}

    for sequence in tokens:
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    return pair_counts

def train_bpe(corpus, vocab_size, log_every=25):
    import time

    vocab = set()

    for tokens in corpus:
        vocab.update(tokens)

    merges = []
    starting_vocab_size = len(vocab)
    merges_needed = vocab_size - starting_vocab_size
    start_time = time.time()

    print(
        f"train_bpe: starting vocab {starting_vocab_size} -> target {vocab_size} "
        f"({merges_needed} merges needed, corpus of {len(corpus)} sequences)"
    )

    while len(vocab) < vocab_size:

        pair_counts = get_pair_counts(corpus)

        if not pair_counts:
            print("train_bpe: no more pairs left to merge -- stopping early")
            break

        best_pair = max(pair_counts, key=pair_counts.get)

        new_token = best_pair[0] + best_pair[1]

        vocab.add(new_token)

        for i in range(len(corpus)):
            corpus[i] = merge_pair(corpus[i], best_pair)

        merges.append((best_pair, new_token))

        done = len(merges)
        if done % log_every == 0 or len(vocab) >= vocab_size:
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            eta = (merges_needed - done) / rate if rate > 0 else float("inf")
            print(
                f"train_bpe: merge {done}/{merges_needed} | vocab {len(vocab)}/{vocab_size} "
                f"| {elapsed:.1f}s elapsed | ~{eta:.1f}s remaining"
            )

    print(f"train_bpe: done -- {len(merges)} merges, final vocab {len(vocab)}, {time.time() - start_time:.1f}s total")

    return vocab, merges

class BPETokenizer:
    def __init__(self, vocab, merges, unk_token="<unk>"):
        """
        vocab:
            Set of learned tokens.

        merges:
            Ordered list of:
            ((token1, token2), new_token)

        unk_token:
            Special token used for any character/token encountered at encode
            time that never showed up in the training corpus (e.g. a rare
            unicode character that only appears in the validation set). Added
            to vocab if it isn't already present.
        """

        self.vocab = set(vocab)
        self.merges = merges
        self.unk_token = unk_token
        self.vocab.add(unk_token)

        # token -> integer ID
        self.token_to_id = {
            token: i for i, token in enumerate(self.vocab)
        }
        self.unk_id = self.token_to_id[unk_token]

        # integer ID -> token
        self.id_to_token = {
            i: token for token, i in self.token_to_id.items()
        }

    def tokenize(self, text):
        """
        Convert raw text into BPE tokens.
        """

        # Start with individual characters
        tokens = list(text)

        # Replay the merges in the order they were learned
        for pair, new_token in self.merges:
            tokens = merge_pair(tokens, pair)

        return tokens

    def encode(self, text):
        """
        Convert text into integer token IDs.
        """

        tokens = self.tokenize(text)

        ids = [
            self.token_to_id.get(token, self.unk_id)
            for token in tokens
        ]

        return ids

    def decode(self, ids):
        """
        Convert integer token IDs back into text.
        """

        tokens = [
            self.id_to_token[i]
            for i in ids
        ]

        return "".join(tokens)