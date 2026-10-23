"""RoPE reference matches the model helper. The in-place path writes those values."""

import torch

from tinyserve.kernels.reference import apply_rope
from tinyserve.kernels.rope import algorithm_bytes, apply_rope_
from tinyserve.model.rope import apply_rotary, compute_inv_freq, rotary_cos_sin


def _model_rotate(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """The path Attention uses: precomputed cos and sin, then rotate_half."""
    cos, sin = rotary_cos_sin(inv_freq, positions.view(1, -1))
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)
    packed = x.transpose(0, 1).unsqueeze(0)  # [1, heads, tokens, head_dim]
    rotated, _key = apply_rotary(packed, packed, cos, sin)
    return rotated.squeeze(0).transpose(0, 1)


def test_reference_matches_model_rope_including_position_zero():
    torch.manual_seed(0)
    inv_freq = compute_inv_freq(16, 500000.0)
    positions = torch.tensor([0, 1, 17, 8193], dtype=torch.long)
    query = torch.randn(4, 4, 16)
    key = torch.randn(4, 2, 16)
    got_q = apply_rope(query, positions, inv_freq)
    got_k = apply_rope(key, positions, inv_freq)
    assert torch.allclose(got_q, _model_rotate(query, positions, inv_freq), atol=1e-6, rtol=1e-6)
    assert torch.allclose(got_k, _model_rotate(key, positions, inv_freq), atol=1e-6, rtol=1e-6)
    # Position 0 is angle 0, so that token is unchanged.
    assert torch.allclose(got_q[0], query[0], atol=1e-6, rtol=1e-6)


def test_inplace_writes_the_reference_and_keeps_storage():
    torch.manual_seed(1)
    inv_freq = compute_inv_freq(16, 10000.0)
    positions = torch.tensor([3, 8], dtype=torch.long)
    query = torch.randn(2, 4, 16)
    key = torch.randn(2, 2, 16)
    expected_q = apply_rope(query, positions, inv_freq)
    expected_k = apply_rope(key, positions, inv_freq)
    query_id = query.data_ptr()
    out_q, out_k = apply_rope_(query, key, positions, inv_freq)
    assert out_q.data_ptr() == query_id
    assert out_k.data_ptr() == key.data_ptr()
    assert torch.allclose(out_q, expected_q, atol=1e-6, rtol=1e-6)
    assert torch.allclose(out_k, expected_k, atol=1e-6, rtol=1e-6)


def test_empty_tokens_stay_empty():
    inv_freq = compute_inv_freq(16, 10000.0)
    positions = torch.zeros(0, dtype=torch.long)
    query = torch.randn(0, 4, 16)
    key = torch.randn(0, 2, 16)
    out_q, out_k = apply_rope_(query, key, positions, inv_freq)
    assert out_q.shape == (0, 4, 16)
    assert out_k.shape == (0, 2, 16)


def test_algorithm_bytes_count_q_and_k_once_each_way():
    # bf16 Q 2*4*16 and K 2*2*16, read and write, plus 2 int64 positions and 8 float32 freqs.
    query = 2 * 4 * 16 * 2
    key = 2 * 2 * 16 * 2
    positions = 2 * 8
    inv_freq = 8 * 4
    assert algorithm_bytes(2, 4, 2, 16, torch.bfloat16) == 2 * (query + key) + positions + inv_freq
