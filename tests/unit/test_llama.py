"""Tiny-model logit parity with Hugging Face, and contiguous-cache decode."""

import json
from pathlib import Path

import torch
from transformers import LlamaConfig as HFConfig
from transformers import LlamaForCausalLM as HFLlama

from tinyserve.model.llama import ContiguousKVCache, LlamaForCausalLM, LlamaModelConfig

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_llama.json"


def _pair(tie: bool) -> tuple[HFLlama, LlamaForCausalLM]:
    raw = json.loads(FIXTURE.read_text())
    hf_cfg = HFConfig(
        vocab_size=raw["vocab_size"],
        hidden_size=raw["hidden_size"],
        intermediate_size=raw["intermediate_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=raw["head_dim"],
        max_position_embeddings=raw["max_position_embeddings"],
        rms_norm_eps=raw["rms_norm_eps"],
        rope_theta=raw["rope_theta"],
        rope_scaling=raw["rope_scaling"],
        tie_word_embeddings=tie,
        attention_bias=False,
        mlp_bias=False,
        attn_implementation="eager",
    )
    hf = HFLlama(hf_cfg).float().eval()
    ours = LlamaForCausalLM(LlamaModelConfig.from_json(FIXTURE)).float().eval()
    if tie:
        ours.config.tie_word_embeddings = True
        ours.lm_head.weight = ours.model.embed_tokens.weight
    missing, unexpected = ours.load_state_dict(hf.state_dict(), strict=False)
    assert not missing, missing
    assert not unexpected, unexpected
    return hf, ours


def test_prefill_logits_match_hf():
    hf, ours = _pair(tie=False)
    ids = torch.randint(0, 256, (1, 7))
    with torch.no_grad():
        reference = hf(ids).logits  # [1, 7, vocab]
        logits = ours(ids)
    torch.testing.assert_close(logits, reference, atol=1e-4, rtol=0)


def test_tied_embeddings_match_hf():
    hf, ours = _pair(tie=True)
    assert ours.lm_head.weight is ours.model.embed_tokens.weight
    ids = torch.randint(0, 256, (1, 5))
    with torch.no_grad():
        reference = hf(ids).logits
        logits = ours(ids)
    torch.testing.assert_close(logits, reference, atol=1e-4, rtol=0)


def test_cache_decode_matches_full_forward():
    """A prefill plus one-token steps must match one forward of the whole sequence."""
    _, model = _pair(tie=False)
    ids = torch.randint(0, 256, (1, 6))
    with torch.no_grad():
        full = model(ids)  # [1, 6, vocab]
        cache = ContiguousKVCache(
            model.config, batch=1, max_len=6, dtype=torch.float32, device="cpu"
        )
        prefill = model(ids[:, :4], cache=cache)
        torch.testing.assert_close(prefill, full[:, :4], atol=1e-5, rtol=0)
        for index in range(4, 6):
            step = model(ids[:, index : index + 1], cache=cache)  # [1, 1, vocab]
            torch.testing.assert_close(step, full[:, index : index + 1], atol=1e-5, rtol=0)
    assert cache.length == 6
