"""Benchmark workloads, filtered to fixed token lengths with a fixed seed.

The raw files stay as published (see `scripts/download_datasets.sh`). This module
is where a conversation becomes one request: a prompt, its token ids, and how
many tokens the benchmark should generate. Two engines then use that same
`output_len` and ignore end-of-sequence, so the comparison is the same work.

Token ids are what the benchmark must send. Re-encoding `prompt` can merge the
last token of a shared prefix with the first token of the suffix, and the
prefix cache would miss.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

Encode = Callable[[str], list[int]]

WORKLOAD_NAMES = ("sharegpt", "code", "shared_prefix", "synthetic")

_HUMAN_ROLES = frozenset({"human", "user"})
_ASSISTANT_ROLES = frozenset({"gpt", "assistant", "chatgpt"})


@dataclass(frozen=True)
class Sample:
    """One benchmark request.

    `output_len` is the number of tokens to generate, taken from the dataset's
    answer or from the synthetic sweep. It is not a count of tokens already
    produced.
    """

    workload: str
    prompt: str
    prompt_token_ids: list[int]
    output_len: int


def load_sharegpt(
    path: str | Path,
    encode: Encode,
    *,
    max_prompt_len: int = 1024,
    max_output_len: int = 1024,
    num_requests: int | None = None,
    seed: int = 0,
) -> list[Sample]:
    """First human turn plus the following assistant turn, then length-filter and subset.

    Later turns are ignored. A benchmark request is one prompt and one target
    length; a multi-turn transcript would mix those.
    """
    samples = []
    for prompt, answer in _sharegpt_pairs(Path(path)):
        prompt_ids = encode(prompt)
        output_len = len(encode(answer))
        samples.append(
            Sample(
                workload="sharegpt",
                prompt=prompt,
                prompt_token_ids=prompt_ids,
                output_len=output_len,
            )
        )
    return take_subset(
        _filter_lengths(samples, max_prompt_len=max_prompt_len, max_output_len=max_output_len),
        num_requests,
        seed,
    )


def load_code(
    path: str | Path,
    encode: Encode,
    *,
    max_prompt_len: int = 1024,
    max_output_len: int = 1024,
    num_requests: int | None = None,
    seed: int = 0,
) -> list[Sample]:
    """Code prompts from a JSON list or JSONL file.

    HumanEval rows use `prompt` and `canonical_solution`. `completion` is
    accepted as the solution field so a hand-built fixture can use either name.
    """
    samples = []
    for row in _read_records(Path(path)):
        prompt = row.get("prompt")
        solution = row.get("canonical_solution", row.get("completion"))
        if not isinstance(prompt, str) or not isinstance(solution, str):
            continue
        prompt = prompt.strip()
        solution = solution.strip()
        if not prompt or not solution:
            continue
        samples.append(
            Sample(
                workload="code",
                prompt=prompt,
                prompt_token_ids=encode(prompt),
                output_len=len(encode(solution)),
            )
        )
    return take_subset(
        _filter_lengths(samples, max_prompt_len=max_prompt_len, max_output_len=max_output_len),
        num_requests,
        seed,
    )


def shared_prefix_workload(
    prefix: str,
    suffixes: list[str],
    encode: Encode,
    *,
    output_len: int,
    max_prompt_len: int = 1024,
    num_requests: int | None = None,
    seed: int = 0,
) -> list[Sample]:
    """Requests that share one token prefix.

    Ids are `encode(prefix) + encode(suffix)`, not `encode(prefix + suffix)`.
    A byte-level tokenizer can glue those two pieces into a different first
    suffix token, and then the shared prefix is no longer a shared token span.
    """
    if output_len < 1:
        raise ValueError("output_len must be at least 1")
    prefix_ids = encode(prefix)
    if not prefix_ids:
        raise ValueError("prefix encodes to no tokens")
    samples = []
    for suffix in suffixes:
        suffix_ids = encode(suffix)
        samples.append(
            Sample(
                workload="shared_prefix",
                prompt=prefix + suffix,
                prompt_token_ids=prefix_ids + suffix_ids,
                output_len=output_len,
            )
        )
    kept = _filter_lengths(samples, max_prompt_len=max_prompt_len, max_output_len=output_len)
    return take_subset(kept, num_requests, seed)


def synthetic_workload(
    *,
    num_requests: int,
    input_len: int,
    output_len: int,
    seed: int,
    vocab_size: int,
) -> list[Sample]:
    """Fixed-length random token ids. `prompt` is empty; send `prompt_token_ids`.

    The ids come from `random.Random(seed)`, which is the seeded generator the
    rest of the benchmark uses. The same seed rebuilds the same requests.
    """
    if num_requests < 0:
        raise ValueError("num_requests must be >= 0")
    if input_len < 1 or output_len < 1:
        raise ValueError("input_len and output_len must be at least 1")
    if vocab_size < 1:
        raise ValueError("vocab_size must be at least 1")
    rng = random.Random(seed)
    samples = []
    for _ in range(num_requests):
        token_ids = [rng.randrange(vocab_size) for _ in range(input_len)]
        samples.append(
            Sample(
                workload="synthetic",
                prompt="",
                prompt_token_ids=token_ids,
                output_len=output_len,
            )
        )
    return samples


def take_subset(samples: list[Sample], num_requests: int | None, seed: int) -> list[Sample]:
    """Return up to `num_requests` samples in an order fixed by `seed`.

    The order is a hash, not `random.shuffle`. Shuffle's algorithm is a Python
    implementation detail; the hash of the sample contents is not.
    """
    if num_requests is not None and num_requests < 0:
        raise ValueError("num_requests must be >= 0")
    ordered = sorted(samples, key=lambda sample: _order_key(seed, sample))
    if num_requests is None:
        return ordered
    return ordered[:num_requests]


def _filter_lengths(
    samples: list[Sample], *, max_prompt_len: int, max_output_len: int
) -> list[Sample]:
    """Drop empty prompts and anything over the spec's 1024/1024 caps (or a test cap)."""
    if max_prompt_len < 1 or max_output_len < 1:
        raise ValueError("length caps must be at least 1")
    return [
        sample
        for sample in samples
        if 0 < len(sample.prompt_token_ids) <= max_prompt_len
        and 0 < sample.output_len <= max_output_len
    ]


def _order_key(seed: int, sample: Sample) -> bytes:
    payload = f"{seed}\n{sample.workload}\n{sample.output_len}\n{sample.prompt}\n" + ",".join(
        str(token) for token in sample.prompt_token_ids
    )
    return hashlib.sha256(payload.encode()).digest()


def _sharegpt_pairs(path: Path) -> list[tuple[str, str]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list of conversations")
    pairs = []
    for item in data:
        if not isinstance(item, dict):
            continue
        turns = item.get("conversations") or item.get("conversation") or []
        prompt = None
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from") or turn.get("role") or "").strip().lower()
            text = turn.get("value")
            if text is None:
                text = turn.get("content")
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue
            if prompt is None and role in _HUMAN_ROLES:
                prompt = text
            elif prompt is not None and role in _ASSISTANT_ROLES:
                pairs.append((prompt, text))
                break
    return pairs


def _read_records(path: Path) -> list[dict]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(row)
        return records
    data = json.loads(text)
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError(f"{path} must be a JSON list of objects")
    return data
