"""Naive engine: one sequence at a time, greedy tokens match Hugging Face."""

import json
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig as HFConfig
from transformers import LlamaForCausalLM as HFLlama

from tinyserve.engine.engine import Engine
from tinyserve.engine.sequence import SamplingParams
from tinyserve.model.llama import LlamaForCausalLM, LlamaModelConfig

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_llama.json"


class _Ascii:
    """Maps each character to its code point. The tiny model vocab is 256, so ASCII fits."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return "".join(chr(token) for token in token_ids)


def _pair() -> tuple[HFLlama, LlamaForCausalLM]:
    torch.manual_seed(0)
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
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False,
        attn_implementation="eager",
    )
    hf = HFLlama(hf_cfg).float().eval()
    ours = LlamaForCausalLM(LlamaModelConfig.from_json(FIXTURE)).float().eval()
    missing, unexpected = ours.load_state_dict(hf.state_dict(), strict=False)
    assert not missing, missing
    assert not unexpected, unexpected
    return hf, ours


def _greedy_hf(model: HFLlama, prompt_ids: list[int], n_new: int) -> list[int]:
    """Argmax loop on the Hugging Face model. `generate` is not used: it can ban EOS."""
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model(ids, use_cache=True)
    next_logits = out.logits[0, -1]
    past = out.past_key_values
    generated: list[int] = []
    for _ in range(n_new):
        token = int(next_logits.argmax().item())
        generated.append(token)
        step = torch.tensor([[token]], dtype=torch.long, device=device)
        out = model(step, past_key_values=past, use_cache=True)
        next_logits = out.logits[0, -1]
        past = out.past_key_values
    return generated


def test_greedy_tokens_match_hf_one_prompt_at_a_time():
    hf, ours = _pair()
    torch.manual_seed(1)
    prompts = [
        torch.randint(0, 256, (5,)).tolist(),
        torch.randint(0, 256, (3,)).tolist(),
    ]
    params = SamplingParams(temperature=0.0, max_tokens=6, stop_token_ids=[])
    engine = Engine(ours, _Ascii())
    results = engine.generate_tokens(prompts, params)
    assert len(results) == 2
    for prompt, result in zip(prompts, results, strict=True):
        assert result.output_token_ids == _greedy_hf(hf, prompt, params.max_tokens)
        assert result.num_computed_tokens == len(prompt) + len(result.output_token_ids)
        assert result.prompt_token_ids == prompt
        assert result.ttft_s is not None
        assert result.ttft_s <= result.e2e_s
        assert result.itl_s is not None
        assert len(result.itl_s) == len(result.output_token_ids) - 1


def test_stop_token_is_not_emitted():
    _, ours = _pair()
    prompt = [4, 5, 6, 7]
    engine = Engine(ours, _Ascii())
    full = engine.generate_tokens([prompt], SamplingParams(temperature=0.0, max_tokens=4))
    assert len(full[0].output_token_ids) == 4
    first, second = full[0].output_token_ids[:2]
    stopped = engine.generate_tokens(
        [prompt], SamplingParams(temperature=0.0, max_tokens=4, stop_token_ids=[second])
    )
    assert stopped[0].output_token_ids == [first]
    assert stopped[0].num_computed_tokens == len(prompt) + 1


def test_seed_repeats_and_text_comes_from_the_tokenizer():
    _, ours = _pair()
    params = SamplingParams(temperature=0.8, top_k=8, top_p=0.9, max_tokens=5, seed=123)
    engine = Engine(ours, _Ascii())
    prompt = [10, 20, 30]
    first = engine.generate_tokens([prompt], params)
    second = engine.generate_tokens([prompt], params)
    assert first[0].output_token_ids == second[0].output_token_ids
    assert first[0].text == _Ascii().decode(first[0].output_token_ids)


def test_generate_encodes_the_prompt_string():
    _, ours = _pair()
    engine = Engine(ours, _Ascii())
    params = SamplingParams(temperature=0.0, max_tokens=3)
    text = "Ab"
    from_string = engine.generate([text], params)
    from_ids = engine.generate_tokens([[ord("A"), ord("b")]], params)
    assert from_string[0].output_token_ids == from_ids[0].output_token_ids
    assert from_string[0].prompt_token_ids == [ord("A"), ord("b")]


def test_empty_prompt_and_zero_max_tokens_raise():
    _, ours = _pair()
    engine = Engine(ours, _Ascii())
    with pytest.raises(ValueError, match="empty"):
        engine.generate([""], SamplingParams(temperature=0.0, max_tokens=2))
    with pytest.raises(ValueError, match="max_tokens"):
        engine.generate_tokens([[1, 2]], SamplingParams(temperature=0.0, max_tokens=0))
