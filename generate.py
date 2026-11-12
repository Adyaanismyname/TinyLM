"""
Generate text from a saved checkpoint.

    python generate.py --checkpoint checkpoints/e2_base_layers8_seed0.pt \
        --prompt "Once upon a time" -n 3

`sample()` is also what generate_samples.py uses to build the side-by-side
comparison of fine-tuning methods, so a sample you produce here goes through
exactly the same encode/generate/decode path as the ones in the report.
"""

import argparse

import torch

from dataset import preprocess_text
from metrics import load_checkpoint


def sample(model, tokenizer, prompt, device, max_new_tokens=300, temperature=0.8, top_k=50, seed=None):
    """
    Generates a continuation of `prompt` and returns (text, hit_eos):
    `text` is only the newly generated part (the prompt is not repeated) and
    is cut at the first end-of-story token; `hit_eos` says whether the model
    ended the story itself rather than running out of `max_new_tokens`.

    The prompt goes through the same preprocess_text normalization as the
    training data (whitespace collapsed, NFKC), so free-form prompts are
    tokenized the way the model saw text during training. An empty prompt
    starts from the end-of-story token, which is what precedes every story
    in the training stream.

    max_new_tokens is clamped so prompt + generation fits the model's
    context window (block_size); a prompt that already fills it raises.
    """
    if seed is not None:
        torch.manual_seed(seed)

    prompt_ids = tokenizer.encode(preprocess_text(prompt)) if prompt.strip() else []
    start_ids = prompt_ids or [tokenizer.eos_id]

    room = model.block_size - len(start_ids)
    if room < 1:
        raise ValueError(
            f"prompt is {len(start_ids)} tokens, which leaves no room to generate "
            f"within the model's {model.block_size}-token context"
        )

    ids = torch.tensor([start_ids], device=device)
    out = model.generate(
        ids, max_new_tokens=min(max_new_tokens, room), temperature=temperature,
        top_k=top_k, use_cache=True, eos_id=tokenizer.eos_id,
    )

    new_ids = out[0, len(start_ids):].tolist()
    hit_eos = tokenizer.eos_id in new_ids
    if hit_eos:
        new_ids = new_ids[: new_ids.index(tokenizer.eos_id)]
    return tokenizer.decode(new_ids), hit_eos


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="", help="Text to continue. Empty = start a fresh story.")
    parser.add_argument("-n", "--num-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 = greedy.")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model, tokenizer, device = load_checkpoint(args.checkpoint)
    for i in range(args.num_samples):
        text, hit_eos = sample(
            model, tokenizer, args.prompt, device, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, seed=args.seed + i,
        )
        print(f"--- sample {i + 1}{'' if hit_eos else ' (hit token limit)'} ---")
        print(f"{args.prompt}{text}" if args.prompt else text.lstrip())
        print()


if __name__ == "__main__":
    main()
