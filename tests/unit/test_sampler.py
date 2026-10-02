"""Greedy, temperature, top-k, top-p, and seeded sampling."""

import torch

from tinyserve.engine.sampler import sample_token
from tinyserve.engine.sequence import SamplingParams


def _logits() -> torch.Tensor:
    # Index 1 is the unique maximum. Index 2 is second.
    scores = torch.full((8,), -10.0)
    scores[1] = 5.0
    scores[2] = 4.0
    scores[3] = 0.0
    return scores


def test_sampling_params_defaults_match_spec():
    params = SamplingParams()
    assert params.temperature == 1.0
    assert params.top_p == 1.0
    assert params.top_k == -1
    assert params.max_tokens == 256
    assert params.seed is None
    assert params.stop_token_ids == []


def test_greedy_is_argmax():
    token = sample_token(_logits(), SamplingParams(temperature=0.0))
    assert int(token) == 1


def test_top_k_never_leaves_the_kept_set():
    logits = _logits()
    generator = torch.Generator().manual_seed(0)
    params = SamplingParams(temperature=1.0, top_k=2, top_p=1.0)
    drawn = {int(sample_token(logits, params, generator)) for _ in range(40)}
    assert drawn <= {1, 2}
    assert 1 in drawn and 2 in drawn


def test_top_p_drops_the_tail():
    # Two huge logits and a tiny one. top_p just above the first token drops the rest
    # only after the cumulative mass crosses, so both large tokens can appear, the tiny one cannot.
    logits = torch.tensor([0.0, 10.0, 10.0, -50.0])
    generator = torch.Generator().manual_seed(1)
    params = SamplingParams(temperature=1.0, top_k=-1, top_p=0.6)
    drawn = {int(sample_token(logits, params, generator)) for _ in range(30)}
    assert 3 not in drawn
    assert drawn <= {1, 2}


def test_seed_is_reproducible_and_independent_of_global_rng():
    logits = torch.randn(32)
    params = SamplingParams(temperature=0.8, top_k=10, top_p=0.9)
    first = torch.Generator().manual_seed(123)
    second = torch.Generator().manual_seed(123)
    torch.randn(4)  # advance the global RNG; the sampler must not use it
    a = [int(sample_token(logits, params, first)) for _ in range(5)]
    b = [int(sample_token(logits, params, second)) for _ in range(5)]
    assert a == b


def test_batch_rows_are_independent():
    logits = torch.stack([_logits(), _logits()])
    logits[1, 1] = -20.0
    logits[1, 4] = 9.0
    chosen = sample_token(logits, SamplingParams(temperature=0.0))
    assert chosen.tolist() == [1, 4]
