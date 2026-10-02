"""Draw the next token from logits.

Greedy, temperature, top-k, and top-p are applied in that order so a test can
turn each one on by itself. A per-request torch.Generator is what makes a seed
repeat: the global RNG is left alone.
"""

import torch

from tinyserve.engine.sequence import SamplingParams


def sample_token(
    logits: torch.Tensor,
    params: SamplingParams,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick one token id per row.

    logits: [batch, vocab] or [vocab]. Returns a long tensor of shape [batch]
    (or a 0-dim tensor when the input was 1-d). Greedy does not use `generator`.
    """
    squeeze = logits.ndim == 1
    if squeeze:
        logits = logits.unsqueeze(0)  # [1, vocab]
    if params.temperature == 0.0:
        chosen = torch.argmax(logits, dim=-1)
        return chosen[0] if squeeze else chosen

    # Temperature flattens the distribution. Values below 0 are not meaningful.
    scaled = logits / params.temperature
    scaled = _apply_top_k(scaled, params.top_k)
    scaled = _apply_top_p(scaled, params.top_p)
    probs = torch.softmax(scaled, dim=-1)  # [batch, vocab]
    chosen = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    return chosen[0] if squeeze else chosen


def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep the k largest logits. 0, -1, and values past the vocab leave the row unchanged."""
    if top_k is None or top_k <= 0:
        return logits
    k = min(top_k, logits.shape[-1])
    threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]  # [batch, 1]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filter: drop tokens after the cumulative probability crosses top_p.

    The token that crosses the threshold is kept. Sorting is undone so the
    caller can softmax in the original vocab order.
    """
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_index = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)  # [batch, vocab]
    # Shift right so the token that crosses top_p is not removed.
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    filtered = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(dim=-1, index=sorted_index, src=filtered)
