"""Flatten a step into the tensors the paged forward consumes.

The scheduler (Phase 4) decides which sequences run. This module only packs
the ones it is given: token ids, positions, slots, and the attention metadata
from spec Section 9. Block tables are padded with -1. Block 0 is a real block,
so a pad of 0 would be read as data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tinyserve.engine.sequence import Sequence
from tinyserve.kernels.kv_store import slot_mapping
from tinyserve.kv.cache import PagedKVCache
from tinyserve.model.llama import LlamaForCausalLM


@dataclass
class ModelInput:
    """One forward. Logits are computed only at ``logits_indices``."""

    input_ids: torch.Tensor  # [num_tokens] int64
    positions: torch.Tensor  # [num_tokens] int64
    slot_mapping: torch.Tensor  # [num_tokens] int64
    query_start_loc: torch.Tensor  # [num_seqs + 1] int32
    seq_lens: torch.Tensor  # [num_seqs] int32
    block_tables: torch.Tensor  # [num_seqs, max_blocks] int32, trailing -1
    logits_indices: torch.Tensor  # [num_seqs] int64


def prepare_model_input(
    seqs: list[Sequence],
    num_new_tokens: list[int],
    block_size: int,
    device: torch.device,
) -> ModelInput:
    """Pack the tokens that this step will run.

    Those tokens must already be on the sequence, starting at
    ``num_computed_tokens``. The block table must already cover them.
    ``num_computed_tokens`` is left unchanged: the caller advances it after
    the forward writes the KV.
    """
    if not seqs:
        raise ValueError("prepare_model_input needs at least one sequence")
    if len(num_new_tokens) != len(seqs):
        raise ValueError("num_new_tokens must have one entry per sequence")
    if any(count < 1 for count in num_new_tokens):
        raise ValueError("each sequence needs at least one new token")

    ids: list[int] = []
    positions: list[int] = []
    slots: list[torch.Tensor] = []
    lengths: list[int] = []
    context_lens: list[int] = []
    for seq, count in zip(seqs, num_new_tokens, strict=True):
        start = seq.num_computed_tokens
        token_ids = seq.prompt_token_ids + seq.output_token_ids
        if start < 0 or start + count > len(token_ids):
            raise ValueError("sequence is missing the new token ids")
        chunk = token_ids[start : start + count]
        pos = torch.arange(start, start + count, device=device)
        ids.extend(chunk)
        positions.extend(range(start, start + count))
        slots.append(slot_mapping(seq.block_table, pos, block_size))
        lengths.append(count)
        context_lens.append(start + count)

    query_start = [0]
    for count in lengths:
        query_start.append(query_start[-1] + count)
    max_blocks = max(len(seq.block_table) for seq in seqs)
    # -1, not 0: block 0 is a real page.
    tables = torch.full((len(seqs), max_blocks), -1, dtype=torch.int32, device=device)
    for index, seq in enumerate(seqs):
        if seq.block_table:
            row = torch.tensor(seq.block_table, dtype=torch.int32, device=device)
            tables[index, : row.shape[0]] = row
    logits_indices = [query_start[i + 1] - 1 for i in range(len(seqs))]
    return ModelInput(
        input_ids=torch.tensor(ids, dtype=torch.long, device=device),  # [num_tokens]
        positions=torch.tensor(positions, dtype=torch.long, device=device),
        slot_mapping=torch.cat(slots),  # [num_tokens]
        query_start_loc=torch.tensor(query_start, dtype=torch.int32, device=device),
        seq_lens=torch.tensor(context_lens, dtype=torch.int32, device=device),
        block_tables=tables,
        logits_indices=torch.tensor(logits_indices, dtype=torch.long, device=device),
    )


def run_paged(
    model: LlamaForCausalLM, cache: PagedKVCache, model_input: ModelInput
) -> torch.Tensor:
    """Run the flattened step. Returns logits ``[num_seqs, vocab]``."""
    return model.forward_paged(
        model_input.input_ids,
        model_input.positions,
        cache,
        model_input.slot_mapping,
        model_input.block_tables,
        model_input.seq_lens,
        model_input.query_start_loc,
        model_input.logits_indices,
    )
