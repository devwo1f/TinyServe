"""All TinyServe settings in one place, as plain dataclasses.

Why this exists: model paths, dtypes, block sizes, memory fractions, and batch limits must never
be hard-coded (spec Section 6), because the same code runs on a CPU laptop with a tiny random
model, an 8 GB dev GPU, and an 80 GB benchmark GPU. Every benchmark result file stores
`config.to_dict()` so a run can be reproduced exactly.

Overrides use dotted names, `<section>.<field>`, both from code and from the CLI:

    cfg = apply_overrides(TinyServeConfig(), {"cache.block_size": "32"})
    python -m some_entrypoint --cache.block_size 32 --scheduler.max_num_seqs 128

This module does not import torch, so it can be used anywhere (CI, benchmark clients, scripts).
"""

import argparse
import dataclasses
import types
import typing
from dataclasses import dataclass, field
from typing import Any

DTYPES = ("auto", "float32", "float16", "bfloat16")


@dataclass
class ModelConfig:
    """Which checkpoint to run and how to place it."""

    model: str = "models/Llama-3.2-1B-Instruct"  # local dir or HF repo id
    tokenizer: str | None = None  # defaults to `model`
    # "auto" = bfloat16 on GPUs that support it, else float16; CPU tests use float32.
    dtype: str = "auto"
    device: str = "cuda"
    max_model_len: int = 4096  # prompt + output tokens per sequence
    seed: int = 0

    def __post_init__(self) -> None:
        _check(self.dtype in DTYPES, f"model.dtype must be one of {DTYPES}, got {self.dtype!r}")
        _check(self.max_model_len > 0, "model.max_model_len must be positive")


@dataclass
class CacheConfig:
    """Paged KV cache sizing (spec Section 9)."""

    block_size: int = 16  # tokens per KV block
    gpu_memory_utilization: float = 0.90  # fraction of total GPU memory TinyServe may use
    # Extra bytes held back after memory profiling, because activation peaks under real load
    # can exceed the single profiling forward pass (spec Section 16).
    memory_safety_margin_gib: float = 0.5
    num_gpu_blocks_override: int | None = None  # skip profiling and use exactly this many blocks
    enable_prefix_caching: bool = True

    def __post_init__(self) -> None:
        _check(self.block_size > 0, "cache.block_size must be positive")
        _check(
            0.0 < self.gpu_memory_utilization <= 1.0,
            "cache.gpu_memory_utilization must be in (0, 1]",
        )
        _check(self.memory_safety_margin_gib >= 0, "cache.memory_safety_margin_gib must be >= 0")
        _check(
            self.num_gpu_blocks_override is None or self.num_gpu_blocks_override > 0,
            "cache.num_gpu_blocks_override must be positive when set",
        )


@dataclass
class SchedulerConfig:
    """Continuous batching limits (spec Section 13, Phase 4)."""

    max_num_batched_tokens: int = 2048  # per-step token budget (decodes + prefill chunks)
    max_num_seqs: int = 64  # max running sequences per step
    enable_chunked_prefill: bool = True

    def __post_init__(self) -> None:
        _check(self.max_num_batched_tokens > 0, "scheduler.max_num_batched_tokens must be > 0")
        _check(self.max_num_seqs > 0, "scheduler.max_num_seqs must be positive")


@dataclass
class SpeculativeConfig:
    """Speculative decoding (spec Phase 8 and Section 14)."""

    enabled: bool = False
    draft_model: str = "models/Llama-3.2-1B-Instruct"
    num_speculative_tokens: int = 4  # k for the static policy
    policy: str = "static"  # off, static, batch_threshold, goodput, per_request
    batch_threshold: int = 16  # used by the batch_threshold policy

    def __post_init__(self) -> None:
        policies = ("off", "static", "batch_threshold", "goodput", "per_request")
        _check(self.policy in policies, f"speculative.policy must be one of {policies}")
        _check(self.num_speculative_tokens >= 0, "speculative.num_speculative_tokens must be >= 0")


@dataclass
class ServerConfig:
    """HTTP server and SLA-aware admission control (spec Phase 7)."""

    host: str = "127.0.0.1"
    port: int = 8000
    admission_policy: str = "fifo"  # fifo, reject (policy A), deadline (policy B)
    ttft_slo_ms: float = 2000.0
    tpot_slo_ms: float = 100.0

    def __post_init__(self) -> None:
        policies = ("fifo", "reject", "deadline")
        _check(self.admission_policy in policies, f"server.admission_policy must be in {policies}")
        _check(0 < self.port < 65536, "server.port must be in 1..65535")


@dataclass
class BenchmarkConfig:
    """Benchmark runs (spec Section 11)."""

    workload: str = "sharegpt"  # sharegpt, code, shared_prefix, synthetic
    num_requests: int = 200
    request_rate: float = float("inf")  # Poisson rate in req/s; inf = all at once (offline)
    num_warmup_requests: int = 10
    num_repeats: int = 3
    seed: int = 0
    ignore_eos: bool = True  # fixed output lengths so engines generate the same token counts
    output_dir: str = "docs/results"

    def __post_init__(self) -> None:
        workloads = ("sharegpt", "code", "shared_prefix", "synthetic")
        _check(self.workload in workloads, f"benchmark.workload must be one of {workloads}")
        _check(self.request_rate > 0, "benchmark.request_rate must be positive")


@dataclass
class TinyServeConfig:
    """Top-level config: one field per section. Section names are the override prefixes."""

    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    speculative: SpeculativeConfig = field(default_factory=SpeculativeConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Plain nested dict, for embedding in result files."""
        return dataclasses.asdict(self)


def _check(condition: bool, message: str) -> None:
    """Raise ValueError with a readable message; used by the __post_init__ validators."""
    if not condition:
        raise ValueError(message)


def _parse_value(text: str, annotation: Any) -> Any:
    """Convert a CLI string to the field's annotated type (int, float, bool, str, or X | None)."""
    if isinstance(annotation, types.UnionType):
        if text.lower() == "none":
            return None
        (inner,) = [a for a in typing.get_args(annotation) if a is not type(None)]
        return _parse_value(text, inner)
    if annotation is bool:
        lowered = text.lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"cannot parse {text!r} as bool")
    if annotation in (int, float, str):
        return annotation(text)
    raise TypeError(f"unsupported config field type {annotation!r}")


def iter_fields(config: TinyServeConfig) -> list[tuple[str, Any]]:
    """All (dotted name, field type) pairs, e.g. ("cache.block_size", int)."""
    names = []
    for section in dataclasses.fields(config):
        section_cls = type(getattr(config, section.name))
        hints = typing.get_type_hints(section_cls)
        for f in dataclasses.fields(section_cls):
            names.append((f"{section.name}.{f.name}", hints[f.name]))
    return names


def apply_overrides(config: TinyServeConfig, overrides: dict[str, Any]) -> TinyServeConfig:
    """Return a new config with dotted-name overrides applied and every section re-validated.

    String values are parsed to the field's type; non-string values are used as given.
    Unknown names raise KeyError, so a typo never silently falls back to a default.
    """
    types_by_name = dict(iter_fields(config))
    per_section: dict[str, dict[str, Any]] = {}
    for name, value in overrides.items():
        if name not in types_by_name:
            raise KeyError(f"unknown config field {name!r}")
        section, key = name.split(".", 1)
        if isinstance(value, str):
            value = _parse_value(value, types_by_name[name])
        per_section.setdefault(section, {})[key] = value

    # dataclasses.replace re-runs __post_init__, so overridden sections are validated again.
    new_sections = {
        section: dataclasses.replace(getattr(config, section), **changes)
        for section, changes in per_section.items()
    }
    return dataclasses.replace(config, **new_sections)


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add one optional `--<section>.<field>` flag per config field (default: not given)."""
    group = parser.add_argument_group("config overrides")
    for name, annotation in iter_fields(TinyServeConfig()):
        group.add_argument(f"--{name}", dest=name, default=None, metavar=_metavar(annotation))


def config_from_args(
    args: argparse.Namespace, base: TinyServeConfig | None = None
) -> TinyServeConfig:
    """Build a config from `base` (defaults if None) plus any flags the user actually passed."""
    base = base or TinyServeConfig()
    given = {
        name: getattr(args, name)
        for name, _ in iter_fields(base)
        if getattr(args, name, None) is not None
    }
    return apply_overrides(base, given)


def _metavar(annotation: Any) -> str:
    """Short type hint for --help output."""
    if isinstance(annotation, types.UnionType):
        inner = [a for a in typing.get_args(annotation) if a is not type(None)][0]
        return f"{inner.__name__.upper()}|none"
    return annotation.__name__.upper()
