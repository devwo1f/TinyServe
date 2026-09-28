"""Tests for tinyserve/config.py: defaults, overrides, validation, and CLI parsing."""

import argparse
import json

import pytest

from tinyserve.config import (
    TinyServeConfig,
    add_config_args,
    apply_overrides,
    config_from_args,
    iter_fields,
)


def test_defaults_match_spec():
    cfg = TinyServeConfig()
    assert cfg.cache.block_size == 16
    assert cfg.model.dtype == "auto"
    assert cfg.model.model.endswith("Llama-3.2-1B-Instruct")
    assert cfg.speculative.enabled is False
    assert cfg.benchmark.num_warmup_requests == 10
    assert cfg.benchmark.num_repeats == 3


def test_to_dict_is_json_serializable():
    d = TinyServeConfig().to_dict()
    assert set(d) == {"model", "cache", "scheduler", "speculative", "server", "benchmark"}
    json.dumps(d)  # result files embed this


def test_apply_overrides_parses_strings_by_type():
    cfg = apply_overrides(
        TinyServeConfig(),
        {
            "cache.block_size": "32",
            "cache.gpu_memory_utilization": "0.5",
            "cache.enable_prefix_caching": "false",
            "cache.num_gpu_blocks_override": "128",
            "model.dtype": "float32",
            "model.tokenizer": "none",
        },
    )
    assert cfg.cache.block_size == 32
    assert cfg.cache.gpu_memory_utilization == 0.5
    assert cfg.cache.enable_prefix_caching is False
    assert cfg.cache.num_gpu_blocks_override == 128
    assert cfg.model.dtype == "float32"
    assert cfg.model.tokenizer is None


def test_apply_overrides_accepts_typed_values_and_leaves_original_untouched():
    base = TinyServeConfig()
    cfg = apply_overrides(base, {"scheduler.max_num_seqs": 8})
    assert cfg.scheduler.max_num_seqs == 8
    assert base.scheduler.max_num_seqs == 64
    assert cfg.cache == base.cache


def test_unknown_field_raises():
    with pytest.raises(KeyError):
        apply_overrides(TinyServeConfig(), {"cache.blocksize": "32"})


@pytest.mark.parametrize(
    "overrides",
    [
        {"cache.block_size": "0"},
        {"cache.gpu_memory_utilization": "1.5"},
        {"model.dtype": "int4"},
        {"speculative.policy": "magic"},
        {"server.admission_policy": "lottery"},
        {"cache.enable_prefix_caching": "maybe"},
    ],
)
def test_invalid_values_raise(overrides):
    with pytest.raises(ValueError):
        apply_overrides(TinyServeConfig(), overrides)


def test_every_field_has_a_cli_flag():
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    flags = {a.dest for a in parser._actions}
    assert all(name in flags for name, _ in iter_fields(TinyServeConfig()))


def test_config_from_cli_args():
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    args = parser.parse_args(
        [
            "--cache.block_size",
            "8",
            "--speculative.enabled",
            "true",
            "--benchmark.request_rate",
            "4",
        ]
    )
    cfg = config_from_args(args)
    assert cfg.cache.block_size == 8
    assert cfg.speculative.enabled is True
    assert cfg.benchmark.request_rate == 4.0
    # Flags not passed keep their defaults.
    assert cfg.scheduler.max_num_batched_tokens == 2048
