"""Rotary position embeddings, including Llama 3 frequency scaling.

RoPE multiplies each pair of query and key channels by an angle that grows with
the token position, so attention can tell positions apart without a learned
embedding. Llama 3 then stretches the long wavelengths (low frequencies) so the
same model can run past the context it was trained on. Getting that stretch
wrong does not crash; it quietly damages long-context quality, which is why
the frequencies are checked against Hugging Face past the original context.
"""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RopeScaling:
    """Llama 3 RoPE scaling parameters, read from the checkpoint config."""

    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position_embeddings: int

    @staticmethod
    def from_config(data: dict | None) -> "RopeScaling | None":
        """Build scaling from a HF `rope_scaling` dict. `default` means no scaling."""
        if not data:
            return None
        rope_type = data.get("rope_type", "default")
        if rope_type == "default":
            return None
        if rope_type != "llama3":
            raise ValueError(f"unsupported rope_type {rope_type!r}")
        return RopeScaling(
            factor=float(data["factor"]),
            low_freq_factor=float(data["low_freq_factor"]),
            high_freq_factor=float(data["high_freq_factor"]),
            original_max_position_embeddings=int(data["original_max_position_embeddings"]),
        )


def compute_inv_freq(
    head_dim: int,
    rope_theta: float,
    scaling: RopeScaling | None = None,
) -> torch.Tensor:
    """Inverse frequencies, shape [head_dim / 2], in float32.

    Without scaling, frequency i is theta ** (-2i / head_dim). Llama 3 leaves
    short wavelengths alone, divides long wavelengths by `factor`, and blends
    the band between `high_freq_factor` and `low_freq_factor`.
    """
    # [head_dim / 2]
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    inv_freq = 1.0 / (rope_theta**exponents)
    if scaling is None:
        return inv_freq

    old_context = scaling.original_max_position_embeddings
    low_freq_wavelen = old_context / scaling.low_freq_factor
    high_freq_wavelen = old_context / scaling.high_freq_factor
    wavelen = 2 * math.pi / inv_freq  # [head_dim / 2]

    # Long waves (low frequency) are divided by the scale factor.
    scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / scaling.factor, inv_freq)
    smooth = (old_context / wavelen - scaling.low_freq_factor) / (
        scaling.high_freq_factor - scaling.low_freq_factor
    )
    blended = (1 - smooth) * scaled / scaling.factor + smooth * scaled
    # True on the closed band between the two wavelengths.
    medium = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    return torch.where(medium, blended, scaled)


def rotary_cos_sin(
    inv_freq: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos and sin for the given positions.

    inv_freq: [head_dim / 2]
    position_ids: [batch, seq]
    returns cos, sin: [batch, seq, head_dim]

    Each angle is repeated on both halves of the head so it lines up with
    `rotate_half`, which swaps those halves.
    """
    # [batch, seq, head_dim / 2]
    freqs = torch.einsum("f,bs->bsf", inv_freq, position_ids.to(dtype=torch.float32))
    emb = torch.cat((freqs, freqs), dim=-1)  # [batch, seq, head_dim]
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the first half.

    x: [..., head_dim]
    """
    half = x.shape[-1] // 2
    first = x[..., :half]
    second = x[..., half:]
    return torch.cat((-second, first), dim=-1)


def apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate queries and keys in place of a learned position embedding.

    query: [batch, num_heads, seq, head_dim]
    key: [batch, num_kv_heads, seq, head_dim]
    cos, sin: [batch, seq, head_dim]
    """
    # [batch, 1, seq, head_dim], so the same angle broadcasts over heads.
    cos_b = cos[:, None, :, :]
    sin_b = sin[:, None, :, :]
    rotated_q = query * cos_b + rotate_half(query) * sin_b
    rotated_k = key * cos_b + rotate_half(key) * sin_b
    return rotated_q, rotated_k
