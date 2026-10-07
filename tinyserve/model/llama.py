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

from tinyserve.kernels.kv_store import write_layer_kv
from tinyserve.kernels.reference import _causal_mask, paged_attention_flat, repeat_kv
from tinyserve.kv.cache import PagedKVCache
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
        # is_causal on this PyTorch is upper-left aligned: a short query only sees the
        # first keys, not the end of the cache. A single new token can see every key,
        # so it needs no mask. Materializing an all-true mask forces the math kernel
        # and the bf16 result drifts off the flash path Hugging Face uses for decode.
        q_len = query.shape[2]
        k_len = key.shape[2]
        if q_len == k_len:
            out = nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
        elif q_len == 1:
            out = nn.functional.scaled_dot_product_attention(query, key, value, is_causal=False)
        else:
            out = nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=_causal_mask(q_len, k_len, query.device)
            )
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)  # [B, S, Nq * D]
        return self.o_proj(out)

    def forward_paged(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slots: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> torch.Tensor:
        """Attention over a flattened batch. K and V are written before the scores.

        ``hidden`` is ``[num_tokens, hidden]``. The new tokens are already
        assigned slots, and ``seq_lens`` counts them. Writing first is what
        lets a query at the end of a chunk see its own key.
        """
        num_tokens = hidden.shape[0]
        query = self.q_proj(hidden).view(num_tokens, self.num_heads, self.head_dim)
        key = self.k_proj(hidden).view(num_tokens, self.num_kv_heads, self.head_dim)
        value = self.v_proj(hidden).view(num_tokens, self.num_kv_heads, self.head_dim)
        # apply_rotary wants a batch and a sequence axis. The flat batch is one row.
        query_r = query.unsqueeze(0).transpose(1, 2)  # [1, Nq, num_tokens, D]
        key_r = key.unsqueeze(0).transpose(1, 2)  # [1, Nk, num_tokens, D]
        query_r, key_r = apply_rotary(query_r, key_r, cos.unsqueeze(0), sin.unsqueeze(0))
        query = query_r.squeeze(0).transpose(0, 1)  # [num_tokens, Nq, D]
        key = key_r.squeeze(0).transpose(0, 1)  # [num_tokens, Nk, D]
        write_layer_kv(k_cache, v_cache, key, value, slots)
        out = paged_attention_flat(query, k_cache, v_cache, block_tables, seq_lens, query_start_loc)
        return self.o_proj(out.reshape(num_tokens, -1))  # [num_tokens, Nq * D]


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

    def forward_paged(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slots: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> torch.Tensor:
        """Same residual block as ``forward``, over the flattened paged batch."""
        residual = hidden
        normed = self.input_layernorm(hidden)  # [num_tokens, H]
        hidden = self.self_attn.forward_paged(
            normed, cos, sin, k_cache, v_cache, slots, block_tables, seq_lens, query_start_loc
        )
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

    def forward_paged(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cache: PagedKVCache,
        slots: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> torch.Tensor:
        """Flattened tokens. Returns hidden states ``[num_tokens, hidden]``.

        ``positions`` are the absolute positions of those tokens, not a range
        starting at 0, so a decode token keeps the position it has in the
        sequence. Every layer writes its own pages and then reads them back.
        """
        hidden = self.embed_tokens(input_ids)  # [num_tokens, H]
        pos = positions.view(1, -1)  # [1, num_tokens]
        cos, sin = rotary_cos_sin(self.inv_freq.to(device=hidden.device), pos)
        cos = cos[0].to(dtype=hidden.dtype)  # [num_tokens, head_dim]
        sin = sin[0].to(dtype=hidden.dtype)
        for index, layer in enumerate(self.layers):
            hidden = layer.forward_paged(
                hidden,
                cos,
                sin,
                cache.k[index],
                cache.v[index],
                slots,
                block_tables,
                seq_lens,
                query_start_loc,
            )
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

    def forward_paged(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cache: PagedKVCache,
        slots: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        logits_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Logits only at ``logits_indices``. Shape ``[num_indices, vocab]``.

        The LM head is the large vocabulary matrix. Sampling needs the last
        token of each sequence, so the other positions skip that multiply.
        """
        hidden = self.model.forward_paged(
            input_ids, positions, cache, slots, block_tables, seq_lens, query_start_loc
        )
        selected = hidden[logits_indices]  # [num_indices, H]
        return self.lm_head(selected)  # [num_indices, vocab]
