"""Llama decoder with a contiguous KV cache.

This is the readable baseline: grouped-query attention, SwiGLU, and RoPE, one
dense cache per batch. Paged memory and continuous batching replace the cache
later; the math stays. Parameter names match Hugging Face so a checkpoint
loads without renaming.

Shapes on tensor lines use batch B, sequence S, hidden H, heads Nq, KV heads Nk,
head dim D, and intermediate I.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from tinyserve.model.rope import RopeScaling, apply_rotary, compute_inv_freq, rotary_cos_sin


@dataclass
class LlamaModelConfig:
    """Architecture fields read from a Hugging Face `config.json`."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    max_position_embeddings: int = 2048
    rope_scaling: dict | None = None
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False

    @staticmethod
    def from_json(path: str | Path) -> "LlamaModelConfig":
        """Load the fields TinyServe needs and ignore the rest of the HF config."""
        data = json.loads(Path(path).read_text())
        hidden = int(data["hidden_size"])
        num_heads = int(data["num_attention_heads"])
        return LlamaModelConfig(
            vocab_size=int(data["vocab_size"]),
            hidden_size=hidden,
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=num_heads,
            num_key_value_heads=int(data["num_key_value_heads"]),
            head_dim=int(data.get("head_dim") or hidden // num_heads),
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-5)),
            rope_theta=float(data.get("rope_theta", 10_000.0)),
            max_position_embeddings=int(data.get("max_position_embeddings", 2048)),
            rope_scaling=data.get("rope_scaling"),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", False)),
            attention_bias=bool(data.get("attention_bias", False)),
            mlp_bias=bool(data.get("mlp_bias", False)),
        )

    @property
    def num_kv_groups(self) -> int:
        """How many query heads share one KV head."""
        return self.num_attention_heads // self.num_key_value_heads


class RMSNorm(nn.Module):
    """Root-mean-square norm. Variance is computed in float32 so bf16 squares do not overflow."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_f = x.float()  # [B, S, H]
        variance = x_f.pow(2).mean(dim=-1, keepdim=True)  # [B, S, 1]
        x_f = x_f * torch.rsqrt(variance + self.eps)
        return self.weight * x_f.to(in_dtype)


class ContiguousKVCache:
    """Dense KV cache. One slot per position, up to a fixed max length.

    Paging exists to avoid this waste; the naive engine keeps the dense layout
    so the attention math can be checked before the block table exists.
    k and v: [num_layers, B, max_len, Nk, D]. `length` is how many positions are valid.
    """

    def __init__(self, config: LlamaModelConfig, batch: int, max_len: int, dtype, device):
        shape = (
            config.num_hidden_layers,
            batch,
            max_len,
            config.num_key_value_heads,
            config.head_dim,
        )
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.length = 0

    def write(self, layer: int, key: torch.Tensor, value: torch.Tensor, start: int) -> None:
        """Store a new chunk. key and value are [B, Nk, S, D]."""
        seq = key.shape[2]
        self.k[layer, :, start : start + seq] = key.transpose(1, 2)  # [B, S, Nk, D]
        self.v[layer, :, start : start + seq] = value.transpose(1, 2)


def _causal_mask(q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
    """True where a query may attend. Queries sit at the end of the key sequence.

    Shape [q_len, k_len]. Query local index i is absolute position (k_len - q_len + i).
    """
    q_pos = torch.arange(k_len - q_len, k_len, device=device)[:, None]  # [q_len, 1]
    k_pos = torch.arange(k_len, device=device)[None, :]  # [1, k_len]
    return k_pos <= q_pos


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat each KV head across the query heads that share it.

    x: [B, Nk, S, D] -> [B, Nq, S, D]. A no-op when the model is not using GQA.
    """
    if n_rep == 1:
        return x
    batch, n_kv, seq, dim = x.shape
    expanded = x[:, :, None, :, :].expand(batch, n_kv, n_rep, seq, dim)
    return expanded.reshape(batch, n_kv * n_rep, seq, dim)


class Attention(nn.Module):
    """Grouped-query attention. Several query heads share one KV head, which shrinks the cache."""

    def __init__(self, config: LlamaModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = config.num_kv_groups
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: ContiguousKVCache | None,
        layer_idx: int,
        start: int,
    ) -> torch.Tensor:
        batch, seq, _ = hidden.shape
        query = self.q_proj(hidden).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden).view(batch, seq, self.num_kv_heads, self.head_dim)
        value = value.transpose(1, 2)
        # query: [B, Nq, S, D], key/value: [B, Nk, S, D]
        query, key = apply_rotary(query, key, cos, sin)
        if cache is not None:
            cache.write(layer_idx, key, value, start)
            end = start + seq
            key = cache.k[layer_idx, :, :end].transpose(1, 2)  # [B, Nk, end, D]
            value = cache.v[layer_idx, :, :end].transpose(1, 2)
        key = repeat_kv(key, self.n_rep)  # [B, Nq, Sk, D]
        value = repeat_kv(value, self.n_rep)
        # When the key sequence is longer than the query (decode), is_causal on this
        # PyTorch lets query 0 see only key 0. The mask below aligns the queries to
        # the end of the cache instead.
        q_len = query.shape[2]
        k_len = key.shape[2]
        if q_len == k_len:
            out = nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
        else:
            out = nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=_causal_mask(q_len, k_len, query.device)
            )
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)  # [B, S, Nq * D]
        return self.o_proj(out)


class MLP(nn.Module):
    """SwiGLU: silu(gate(x)) * up(x), then down. Two up-projections, gated, instead of one."""

    def __init__(self, config: LlamaModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, H] -> [B, S, I] -> [B, S, H]
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """Pre-norm residual block: attention, then MLP. The residual is what lets many layers train."""

    def __init__(self, config: LlamaModelConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: ContiguousKVCache | None,
        layer_idx: int,
        start: int,
    ) -> torch.Tensor:
        residual = hidden
        hidden = self.self_attn(self.input_layernorm(hidden), cos, sin, cache, layer_idx, start)
        hidden = residual + hidden
        residual = hidden
        hidden = self.mlp(self.post_attention_layernorm(hidden))
        return residual + hidden


class LlamaModel(nn.Module):
    """Token embeddings, decoder stack, and final norm. Does not produce logits."""

    def __init__(self, config: LlamaModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # Not a parameter: recomputed from the config, and checked against HF in the RoPE tests.
        self.inv_freq = compute_inv_freq(
            config.head_dim,
            config.rope_theta,
            RopeScaling.from_config(config.rope_scaling),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: ContiguousKVCache | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """input_ids: [B, S]. Returns hidden states [B, S, H]."""
        batch, seq = input_ids.shape
        hidden = self.embed_tokens(input_ids)  # [B, S, H]
        start = 0 if cache is None else cache.length
        if position_ids is None:
            positions = torch.arange(start, start + seq, device=input_ids.device)
            position_ids = positions.unsqueeze(0).expand(batch, -1)  # [B, S]
        cos, sin = rotary_cos_sin(self.inv_freq.to(device=hidden.device), position_ids)
        cos = cos.to(dtype=hidden.dtype)
        sin = sin.to(dtype=hidden.dtype)
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, cos, sin, cache, index, start)
        if cache is not None:
            cache.length = start + seq
        return self.norm(hidden)


class LlamaForCausalLM(nn.Module):
    """Full model. Tied embeddings share one matrix for the input embedding and the LM head."""

    def __init__(self, config: LlamaModelConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: ContiguousKVCache | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """input_ids: [B, S]. Returns logits [B, S, vocab]."""
        hidden = self.model(input_ids, cache=cache, position_ids=position_ids)  # [B, S, H]
        return self.lm_head(hidden)  # [B, S, vocab]
