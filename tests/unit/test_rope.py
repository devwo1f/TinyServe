"""RoPE matches Hugging Face, including Llama 3 scaling past the original context."""

import json
from pathlib import Path

import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

from tinyserve.model.rope import RopeScaling, apply_rotary, compute_inv_freq, rotary_cos_sin

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_llama.json"
POSITIONS = [0, 1, 17, 8191, 8192, 8193, 20000]


def _hf_and_ours(scaling: dict | None):
    raw = json.loads(FIXTURE.read_text())
    hf_cfg = LlamaConfig(
        hidden_size=raw["hidden_size"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=raw["head_dim"],
        max_position_embeddings=max(POSITIONS) + 1,
        rope_theta=raw["rope_theta"],
        rope_scaling=scaling,
    )
    hf = LlamaRotaryEmbedding(hf_cfg)
    ours = compute_inv_freq(
        raw["head_dim"],
        raw["rope_theta"],
        RopeScaling.from_config(scaling),
    )
    return hf, ours


def _compare(scaling: dict | None):
    hf, inv_freq = _hf_and_ours(scaling)
    position_ids = torch.tensor([POSITIONS], dtype=torch.long)  # [1, seq]
    dummy = torch.zeros(1, len(POSITIONS), 16)  # [batch, seq, head_dim]
    hf_cos, hf_sin = hf(dummy, position_ids)
    cos, sin = rotary_cos_sin(inv_freq, position_ids)  # [1, seq, head_dim]
    torch.testing.assert_close(cos, hf_cos, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(sin, hf_sin, atol=1e-5, rtol=1e-5)

    query = torch.randn(1, 4, len(POSITIONS), 16)  # [batch, heads, seq, head_dim]
    key = torch.randn(1, 2, len(POSITIONS), 16)  # [batch, kv_heads, seq, head_dim]
    hf_q, hf_k = apply_rotary_pos_emb(query, key, hf_cos, hf_sin)
    our_q, our_k = apply_rotary(query, key, cos, sin)
    torch.testing.assert_close(our_q, hf_q, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(our_k, hf_k, atol=1e-5, rtol=1e-5)


def test_default_rope_matches_hf():
    _compare(None)


def test_llama3_rope_matches_hf_past_original_context():
    raw = json.loads(FIXTURE.read_text())
    _compare(raw["rope_scaling"])
    assert max(POSITIONS) > raw["rope_scaling"]["original_max_position_embeddings"]
