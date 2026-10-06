"""One real prompt through the naive engine, bf16, against Hugging Face.

The 10-prompt check lives in test_llama_parity.py. This test only checks that
`Engine.generate` walks the same greedy path. Both sides use Tokenizer.encode,
which does not add a beginning-of-sequence token.
"""

import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM

from scripts.env_info import collect_env_info
from tinyserve.engine.engine import Engine
from tinyserve.engine.sequence import SamplingParams
from tinyserve.model.llama import LlamaForCausalLM, LlamaModelConfig
from tinyserve.model.tokenizer import Tokenizer
from tinyserve.model.weights import load_hf_weights

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "models" / "Llama-3.2-1B-Instruct"
RESULT = REPO / "docs" / "results" / "phase1" / "2026-10-02_p1-6-naive-engine.json"
PROMPT = "The capital of France is"
NEW_TOKENS = 16


def _greedy_hf(model, input_ids: torch.Tensor, n_new: int) -> list[int]:
    out = model(input_ids, use_cache=True)
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
    return generated


@pytest.mark.gpu
@pytest.mark.slow
def test_engine_greedy_matches_hf_on_1b():
    if not (MODEL_DIR / "model.safetensors").exists():
        pytest.skip("Llama-3.2-1B weights not downloaded")
    tokenizer = Tokenizer.from_pretrained(str(MODEL_DIR))
    prompt_ids = tokenizer.encode(PROMPT)
    device = torch.device("cuda")
    params = SamplingParams(temperature=0.0, max_tokens=NEW_TOKENS, stop_token_ids=[])

    hf = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).to(device).eval()
    with torch.inference_mode():
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        hf_tokens = _greedy_hf(hf, ids, NEW_TOKENS)
    del hf
    torch.cuda.empty_cache()

    ours = LlamaForCausalLM(LlamaModelConfig.from_json(MODEL_DIR / "config.json")).eval()
    load_hf_weights(ours, MODEL_DIR, device=device, dtype=torch.bfloat16)
    engine = Engine(ours, tokenizer)
    result = engine.generate([PROMPT], params)[0]

    matched = result.output_token_ids == hf_tokens
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(
        json.dumps(
            {
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "dtype": "bfloat16",
                "prompt": PROMPT,
                "new_tokens": NEW_TOKENS,
                "matched": matched,
                "output_text": result.text,
                "env_info": collect_env_info(),
            },
            indent=2,
        )
        + "\n"
    )
    assert matched
    assert result.text == tokenizer.decode(hf_tokens)
