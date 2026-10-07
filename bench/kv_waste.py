"""KV waste of paged blocks versus a contiguous max-length reservation.

The lengths are fixed. Waste does not depend on the weights: a token's slot
is the same on the tiny model and on Llama 3.2 1B. The script still runs the
engine, so the rows are the samples the step loop records. It refuses to
write if those samples disagree with the block-manager walk, or if the
contiguous reservation is not the larger waste on this schedule.

``max_len`` is ``TinyServeConfig.model.max_model_len`` (4096). The tiny
fixture's context is 512, and every request here is shorter than that. The
contiguous side still charges 4096, because that is the reservation a dense
cache makes at the serving limit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.offline import _json_ready, _percentiles
from scripts.env_info import collect_env_info
from tinyserve.engine.engine import Engine
from tinyserve.engine.sequence import SamplingParams
from tinyserve.kv.waste import concurrent_slot_sample, request_step_samples
from tinyserve.model.llama import LlamaForCausalLM, LlamaModelConfig

# Prompt length, then output length. Several land inside a block of 16.
LENGTHS = (
    (1, 1),
    (15, 1),
    (16, 8),
    (17, 4),
    (31, 3),
    (64, 16),
    (100, 7),
    (200, 32),
)
BLOCK_SIZE = 16
MAX_LEN = 4096
FIXTURE = Path("tests/fixtures/tiny_llama.json")


class _Ids:
    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return ""


def _run(engine: Engine) -> list[dict]:
    rows = []
    for index, (prompt_len, output_len) in enumerate(LENGTHS):
        params = SamplingParams(temperature=0.0, max_tokens=output_len)
        result = engine.generate_tokens([[1] * prompt_len], params)[0]
        expected = request_step_samples(
            prompt_len, output_len, block_size=BLOCK_SIZE, max_len=MAX_LEN
        )
        if result.slot_samples != tuple(expected):
            raise SystemExit(f"request {index} slot samples disagree with the block manager")
        for step, sample in enumerate(result.slot_samples):
            rows.append(
                {
                    "record": "step",
                    "request_index": index,
                    "step_index": step,
                    "prompt_len": prompt_len,
                    "output_len": output_len,
                    "used_slots": sample.used_slots,
                    "paged_allocated_slots": sample.paged_allocated_slots,
                    "contiguous_allocated_slots": sample.contiguous_allocated_slots,
                    "paged_waste": sample.paged_waste,
                    "contiguous_waste": sample.contiguous_waste,
                }
            )
    return rows


def _summary(rows: list[dict]) -> dict:
    lengths = [prompt + output for prompt, output in LENGTHS]
    live = concurrent_slot_sample(lengths, block_size=BLOCK_SIZE, max_len=MAX_LEN)
    paged = [row["paged_waste"] for row in rows]
    contiguous = [row["contiguous_waste"] for row in rows]
    if sum(contiguous) <= sum(paged):
        raise SystemExit("contiguous max-length waste was not larger than paged waste")
    if live.contiguous_waste <= live.paged_waste:
        raise SystemExit("concurrent contiguous waste was not larger than paged waste")
    return {
        "record": "summary",
        "num_steps": len(rows),
        "one_at_a_time": {
            "paged_waste": _percentiles(paged),
            "contiguous_waste": _percentiles(contiguous),
        },
        "concurrent_at_full_length": {
            "used_slots": live.used_slots,
            "paged_allocated_slots": live.paged_allocated_slots,
            "contiguous_allocated_slots": live.contiguous_allocated_slots,
            "paged_waste": live.paged_waste,
            "contiguous_waste": live.contiguous_waste,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="docs/results/phase3/2026-10-07_p3-7-kv-waste.jsonl",
    )
    args = parser.parse_args()
    model = LlamaForCausalLM(LlamaModelConfig.from_json(FIXTURE)).float().eval()
    engine = Engine(
        model,
        _Ids(),
        block_size=BLOCK_SIZE,
        enable_prefix_caching=False,
        record_kv_waste=True,
        contiguous_max_len=MAX_LEN,
    )
    rows = _run(engine)
    summary = _summary(rows)
    env = collect_env_info()
    meta = {
        "record": "meta",
        "config": {
            "model": "tiny_llama",
            "fixture": str(FIXTURE),
            "dtype": "float32",
            "workload": "variable_lengths",
            "lengths": [{"prompt_len": prompt, "output_len": output} for prompt, output in LENGTHS],
            "block_size": BLOCK_SIZE,
            "contiguous_max_len": MAX_LEN,
            "prefix_caching": False,
            "temperature": 0,
        },
        "env_info": env,
        "git_commit": env.get("git_commit"),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [meta, *rows, summary]
    path.write_text("".join(json.dumps(_json_ready(line)) + "\n" for line in lines))
    print(path)


if __name__ == "__main__":
    main()
