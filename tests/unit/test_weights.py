"""Checkpoint loading, without the real weights. The GPU parity test is separate."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tinyserve.model.llama import LlamaForCausalLM, LlamaModelConfig
from tinyserve.model.weights import load_hf_weights

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_llama.json"
LLAMA_DIR = Path(__file__).resolve().parents[2] / "models" / "Llama-3.2-1B-Instruct"


def test_round_trip_through_safetensors(tmp_path: Path):
    source = LlamaForCausalLM(LlamaModelConfig.from_json(FIXTURE)).float().eval()
    save_file(source.state_dict(), tmp_path / "model.safetensors")
    loaded = LlamaForCausalLM(LlamaModelConfig.from_json(FIXTURE)).float().eval()
    with torch.no_grad():
        loaded.model.embed_tokens.weight.normal_()
    load_hf_weights(loaded, tmp_path, device="cpu", dtype=torch.float32)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)


def test_tied_checkpoint_without_lm_head(tmp_path: Path):
    config = LlamaModelConfig.from_json(FIXTURE)
    config.tie_word_embeddings = True
    source = LlamaForCausalLM(config).float().eval()
    tensors = {
        name: value for name, value in source.state_dict().items() if name != "lm_head.weight"
    }
    save_file(tensors, tmp_path / "model.safetensors")
    loaded = LlamaForCausalLM(config).float().eval()
    with torch.no_grad():
        loaded.model.embed_tokens.weight.normal_()
    load_hf_weights(loaded, tmp_path, device="cpu", dtype=torch.float32)
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
    torch.testing.assert_close(loaded.model.embed_tokens.weight, source.model.embed_tokens.weight)


def test_real_1b_config_shape():
    path = LLAMA_DIR / "config.json"
    if not path.exists():
        pytest.skip("Llama-3.2-1B config not downloaded")
    config = LlamaModelConfig.from_json(path)
    raw = json.loads(path.read_text())
    assert config.vocab_size == raw["vocab_size"]
    assert config.tie_word_embeddings is True
    assert config.head_dim == 64
    assert config.num_hidden_layers == 16
    assert config.num_key_value_heads == 8
