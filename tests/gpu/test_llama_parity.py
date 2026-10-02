"""Greedy parity of TinyServe against Hugging Face on Llama-3.2-1B, bf16.

Spec Section 12: 64 new tokens on 10 fixed prompts. The first 32 tokens must
match on at least 9 prompts. bf16 is not exact across implementations, so the
test also records the max absolute logit difference. Both numbers are written
by this test; they are not typed in by hand.
"""

import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.env_info import collect_env_info
from tinyserve.model.llama import ContiguousKVCache, LlamaForCausalLM, LlamaModelConfig
from tinyserve.model.weights import load_hf_weights

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "models" / "Llama-3.2-1B-Instruct"
RESULT = REPO / "docs" / "results" / "phase1" / "2026-10-01_p1-4-greedy-parity.json"
NEW_TOKENS = 64
MATCH_PREFIX = 32
PROMPTS = [
    "The capital of France is",
    "Once upon a time there was",
    "def add(a, b):\n    return",
    "Explain gravity in one sentence:",
    "2 + 2 =",
    "The opposite of hot is",
    "In 1969, astronauts",
    "A haiku about rain:",
    "Translate hello to Spanish:",
    "The three primary colors are",
]


def _greedy_hf(model, input_ids: torch.Tensor, n_new: int) -> tuple[list[int], torch.Tensor]:
    """Argmax tokens from the Hugging Face forward pass.

    `generate(min_new_tokens=...)` bans the EOS id until that many tokens exist, so it
    is not plain greedy. This loop is the same rule TinyServe uses: always take argmax.
    """
    out = model(input_ids, use_cache=True)
    first = out.logits[0, -1].float().cpu()
    next_logits = out.logits[:, -1]
    past = out.past_key_values
    generated: list[int] = []
    for _ in range(n_new):
        token = int(next_logits.argmax(dim=-1).item())
        generated.append(token)
        step = torch.tensor([[token]], device=input_ids.device)
        out = model(step, past_key_values=past, use_cache=True)
        next_logits = out.logits[:, -1]
        past = out.past_key_values
    return generated, first


def _greedy_ours(
    model: LlamaForCausalLM, input_ids: torch.Tensor, n_new: int
) -> tuple[list[int], torch.Tensor]:
    """Return new token ids and the logits at the last prompt position. input_ids: [1, S]."""
    device = input_ids.device
    dtype = next(model.parameters()).dtype
    cache = ContiguousKVCache(
        model.config,
        batch=1,
        max_len=input_ids.shape[1] + n_new,
        dtype=dtype,
        device=device,
    )
    logits = model(input_ids, cache=cache)  # [1, S, vocab]
    first = logits[0, -1].float().cpu()
    generated: list[int] = []
    next_logits = logits[:, -1]
    for _ in range(n_new):
        token = int(next_logits.argmax(dim=-1).item())
        generated.append(token)
        step = torch.tensor([[token]], device=device)
        next_logits = model(step, cache=cache)[:, -1]
    return generated, first


@pytest.mark.gpu
@pytest.mark.slow
def test_1b_greedy_matches_hf():
    if not (MODEL_DIR / "model.safetensors").exists():
        pytest.skip("Llama-3.2-1B weights not downloaded")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    device = torch.device("cuda")

    hf = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).to(device).eval()
    hf_tokens: list[list[int]] = []
    hf_logits: list[torch.Tensor] = []
    with torch.no_grad():
        for prompt in PROMPTS:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            generated, logits = _greedy_hf(hf, ids, NEW_TOKENS)
            hf_logits.append(logits)
            hf_tokens.append(generated)
    del hf
    torch.cuda.empty_cache()

    ours = LlamaForCausalLM(LlamaModelConfig.from_json(MODEL_DIR / "config.json")).eval()
    load_hf_weights(ours, MODEL_DIR, device=device, dtype=torch.bfloat16)
    our_tokens: list[list[int]] = []
    max_logit_diff = 0.0
    with torch.no_grad():
        for prompt, reference in zip(PROMPTS, hf_logits, strict=True):
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            generated, logits = _greedy_ours(ours, ids, NEW_TOKENS)
            our_tokens.append(generated)
            max_logit_diff = max(max_logit_diff, (logits - reference).abs().max().item())

    prefix_matches = [
        a[:MATCH_PREFIX] == b[:MATCH_PREFIX] for a, b in zip(hf_tokens, our_tokens, strict=True)
    ]
    n_match = sum(prefix_matches)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(
        json.dumps(
            {
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "dtype": "bfloat16",
                "new_tokens": NEW_TOKENS,
                "match_prefix": MATCH_PREFIX,
                "prompts_matched": n_match,
                "n_prompts": len(PROMPTS),
                "max_abs_logit_diff": max_logit_diff,
                "per_prompt_prefix_match": prefix_matches,
                "env_info": collect_env_info(),
            },
            indent=2,
        )
        + "\n"
    )
    assert n_match >= 9, f"only {n_match}/10 prompts matched on the first {MATCH_PREFIX} tokens"
