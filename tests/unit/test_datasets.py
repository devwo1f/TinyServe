"""Filtering and fixed-seed subsets. No downloaded datasets required."""

import json

import pytest

from bench.datasets import (
    Sample,
    load_code,
    load_sharegpt,
    shared_prefix_workload,
    synthetic_workload,
    take_subset,
)


def _chars(text: str) -> list[int]:
    return [ord(ch) for ch in text]


def _sharegpt(tmp_path) -> object:
    rows = [
        {
            "id": "ok",
            "conversations": [
                {"from": "human", "value": "aa"},
                {"from": "gpt", "value": "bbb"},
                {"from": "human", "value": "ignored"},
                {"from": "gpt", "value": "also ignored"},
            ],
        },
        {
            "id": "long-prompt",
            "conversations": [
                {"from": "human", "value": "x" * 20},
                {"from": "gpt", "value": "y"},
            ],
        },
        {
            "id": "long-output",
            "conversations": [
                {"from": "human", "value": "z"},
                {"from": "gpt", "value": "w" * 20},
            ],
        },
        {
            "id": "no-reply",
            "conversations": [{"from": "human", "value": "only human"}],
        },
        {
            "id": "blank",
            "conversations": [
                {"from": "human", "value": "   "},
                {"from": "gpt", "value": "hi"},
            ],
        },
        {
            "id": "roles",
            "conversations": [
                {"from": "user", "value": "hi"},
                {"from": "assistant", "value": "yo"},
            ],
        },
    ]
    path = tmp_path / "share.json"
    path.write_text(json.dumps(rows))
    return path


def test_sharegpt_filters_length_and_keeps_the_first_turn(tmp_path):
    samples = load_sharegpt(
        _sharegpt(tmp_path),
        _chars,
        max_prompt_len=4,
        max_output_len=4,
        seed=0,
    )
    by_prompt = {sample.prompt: sample for sample in samples}
    assert set(by_prompt) == {"aa", "hi"}
    assert by_prompt["aa"].prompt_token_ids == [ord("a"), ord("a")]
    assert by_prompt["aa"].output_len == 3
    assert by_prompt["hi"].output_len == 2
    assert all(sample.workload == "sharegpt" for sample in samples)


def test_subset_is_stable_for_a_seed_and_changes_with_the_seed(tmp_path):
    path = _sharegpt(tmp_path)
    kwargs = {"max_prompt_len": 4, "max_output_len": 4, "num_requests": 1}
    first = load_sharegpt(path, _chars, seed=0, **kwargs)
    second = load_sharegpt(path, _chars, seed=0, **kwargs)
    other = load_sharegpt(path, _chars, seed=1, **kwargs)
    assert first == second
    assert len(first) == 1
    assert first != other


def test_code_jsonl_and_json_use_the_solution_length(tmp_path):
    rows = [
        {"prompt": "def a():\n", "canonical_solution": "  return 1\n"},
        {"prompt": "def b():\n", "completion": "pass"},
        {"prompt": "   ", "canonical_solution": "x"},
        {"prompt": "def long():\n", "canonical_solution": "y" * 30},
    ]
    jsonl = tmp_path / "code.jsonl"
    jsonl.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    as_json = tmp_path / "code.json"
    as_json.write_text(json.dumps(rows))
    from_jsonl = load_code(jsonl, _chars, max_prompt_len=16, max_output_len=16, seed=0)
    from_json = load_code(as_json, _chars, max_prompt_len=16, max_output_len=16, seed=0)
    assert [sample.prompt for sample in from_jsonl] == [sample.prompt for sample in from_json]
    kept = {sample.prompt: sample.output_len for sample in from_jsonl}
    assert kept == {"def a():": len(_chars("return 1")), "def b():": len(_chars("pass"))}


def test_shared_prefix_concatenates_token_ids():
    prefix = "SYSTEM "
    samples = shared_prefix_workload(prefix, ["alpha", "beta"], _chars, output_len=3, seed=0)
    prefix_ids = _chars(prefix)
    assert len(samples) == 2
    assert all(sample.prompt_token_ids[: len(prefix_ids)] == prefix_ids for sample in samples)
    assert all(sample.output_len == 3 for sample in samples)
    # Order is the hash order, not the input order. Both suffixes are present.
    assert {tuple(sample.prompt_token_ids[len(prefix_ids) :]) for sample in samples} == {
        tuple(_chars("alpha")),
        tuple(_chars("beta")),
    }


def test_shared_prefix_drops_a_prompt_over_the_cap():
    samples = shared_prefix_workload(
        "abc",
        ["d", "efghij"],
        _chars,
        output_len=1,
        max_prompt_len=5,
        seed=0,
    )
    assert [sample.prompt for sample in samples] == ["abcd"]


def test_synthetic_lengths_and_seed():
    first = synthetic_workload(num_requests=4, input_len=5, output_len=7, seed=3, vocab_size=11)
    again = synthetic_workload(num_requests=4, input_len=5, output_len=7, seed=3, vocab_size=11)
    other = synthetic_workload(num_requests=4, input_len=5, output_len=7, seed=4, vocab_size=11)
    assert first == again
    assert first != other
    assert all(len(sample.prompt_token_ids) == 5 for sample in first)
    assert all(sample.output_len == 7 for sample in first)
    assert all(sample.prompt == "" for sample in first)
    assert all(0 <= token < 11 for sample in first for token in sample.prompt_token_ids)


def test_take_subset_rejects_a_negative_count():
    sample = Sample(workload="synthetic", prompt="", prompt_token_ids=[1], output_len=1)
    with pytest.raises(ValueError, match="num_requests"):
        take_subset([sample], -1, seed=0)


def test_sharegpt_rejects_a_non_list(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"conversations": []}))
    with pytest.raises(ValueError, match="JSON list"):
        load_sharegpt(path, _chars)
