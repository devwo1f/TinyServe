"""Smoke tests for the repository skeleton.

These exist so that `uv run pytest` has something real to check before any engine code lands:
the package imports, and the tiny test model config matches the shape the spec requires
(Section 6), since every CPU unit test in later phases depends on it.
"""

import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_package_imports():
    import tinyserve
    import tinyserve.engine
    import tinyserve.kernels
    import tinyserve.kv
    import tinyserve.model
    import tinyserve.quant
    import tinyserve.server
    import tinyserve.spec

    assert tinyserve.__version__


def test_tiny_llama_fixture_matches_spec():
    cfg = json.loads((FIXTURES / "tiny_llama.json").read_text())
    assert cfg["num_hidden_layers"] == 2
    assert cfg["hidden_size"] == 64
    assert cfg["num_attention_heads"] == 4
    assert cfg["num_key_value_heads"] == 2
    assert cfg["head_dim"] == 16
    assert cfg["vocab_size"] == 256
    # GQA sanity: query heads split evenly across KV heads, and heads tile the hidden size.
    assert cfg["num_attention_heads"] % cfg["num_key_value_heads"] == 0
    assert cfg["num_attention_heads"] * cfg["head_dim"] == cfg["hidden_size"]
