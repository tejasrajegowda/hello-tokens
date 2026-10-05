"""Speculative decoding (Leviathan et al., 2023; Chen et al., 2023).

A small draft model guesses `k` tokens one at a time, which is cheap. The big (target) model then reads
all of them in one pass, about the cost of writing one token, and keeps the guesses it agrees with:

- Guess i is kept with probability min(1, p(x) / q(x)), where p is the target's distribution and q the
  draft's, both after temperature/top-k/top-p.
- At the first rejection, a replacement is drawn from max(0, p - q), renormalised: the part of the
  target's distribution the draft under-estimated. The round ends there.
- If every guess is kept, the target's pass already gives the distribution for one more token: a free
  extra token.

The rule makes the output follow the target's distribution exactly, as if the draft did not exist; the
draft only changes the speed. With temperature 0 both distributions are one-hot, so a guess is kept
exactly when it equals the target's top token, and the text is identical to ordinary greedy writing.

Both models keep a KV cache. Rejected guesses are forgotten by rolling each cache back.
"""

import time
from collections.abc import Iterator

import torch

from hello_tokens.generation.sampling import Step, filtered_probabilities
from hello_tokens.model.gpt import GPT


@torch.no_grad()
def speculative_stream(
    target: GPT,
    draft: GPT,
    ids: list[int],
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float | None = None,
    stop_id: int | None = None,
    generator: torch.Generator | None = None,
    k: int = 4,
    stats: dict | None = None,
) -> Iterator[Step]:
    """Yield new tokens until stop_id (not yielded) or max_new_tokens, k drafted tokens per round.

    `stats`, if given, collects "rounds", "drafted" and "accepted" (acceptance rate = accepted / drafted).
    """
    target.eval(), draft.eval()
    device = next(target.parameters()).device
    context = min(target.config.context, draft.config.context)
    keep = context * 3 // 4
    tokens = list(ids)
    # Each cache holds a stretch of the text: tokens[base : base + cache.length]. Whatever comes after
    # that is "pending": read at the start of the next round.
    target_cache, draft_cache = target.new_cache(), draft.new_cache()
    base = max(0, len(tokens) - context)  # both caches start at the same place
    counts = stats if stats is not None else {}
    for key in ("rounds", "drafted", "accepted"):
        counts.setdefault(key, 0)

    def dials(logits: torch.Tensor) -> torch.Tensor:
        return filtered_probabilities(logits, temperature, top_k, top_p)

    written = 0
    while written < max_new_tokens:
        started = time.perf_counter()
        # This round reads the pending tokens plus k guesses. If a cache can't fit that, start both
        # again from the last 3/4 of the text, as plain cached writing does.
        if len(tokens) - base + k > context:
            target_cache.rollback(0), draft_cache.rollback(0)
            base = len(tokens) - keep
        target_pending = tokens[base + target_cache.length :]
        draft_pending = tokens[base + draft_cache.length :]
        before = len(tokens)

        # 1. The draft guesses k tokens, one at a time.
        guesses, draft_dists = [], []
        feed = draft_pending
        for _ in range(k):
            q = dials(draft(torch.tensor([feed], device=device), draft_cache)[0, -1])
            guess = int(torch.multinomial(q, 1, generator=generator))
            guesses.append(guess)
            draft_dists.append(q)
            feed = [guess]

        # 2. The target reads its pending tokens and all k guesses in one pass. The last k + 1
        # positions give its distribution after each guess (and one more after the last guess).
        logits = target(torch.tensor([target_pending + guesses], device=device), target_cache)[0]
        target_dists = [dials(row) for row in logits[-(k + 1):]]

        # 3. Keep guesses while the target agrees; replace the first one it rejects.
        new: list[int] = []
        for guess, p, q in zip(guesses, target_dists, draft_dists):
            if torch.rand(1, generator=generator, device=device) < torch.clamp(p[guess] / q[guess], max=1.0):
                new.append(guess)
                continue
            residual = torch.clamp(p - q, min=0)
            new.append(int(torch.multinomial(residual / residual.sum(), 1, generator=generator)))
            break
        else:  # every guess kept: the target's last distribution gives one more token
            new.append(int(torch.multinomial(target_dists[k], 1, generator=generator)))
        accepted = len(new) - 1 if len(new) <= k else k
        counts["rounds"] += 1
        counts["drafted"] += k
        counts["accepted"] += accepted

        # 4. Forget what wasn't kept. The target read every guess, the draft all but the last one; of
        # those, only the kept guesses are now text. The last new token is read next round.
        target_cache.rollback(before + accepted - base)
        draft_cache.rollback(before + min(accepted, k - 1) - base)
        seconds = (time.perf_counter() - started) / len(new)
        for token in new:
            if token == stop_id:
                return
            tokens.append(token)
            written += 1
            yield Step(token, seconds)
            if written == max_new_tokens:
                return
