"""Load a Hugging Face safetensors checkpoint into a TinyServe Llama.

The parameter names are the Hugging Face names, so the mapping is the identity.
Tied models (Llama 3.2 1B) do not store `lm_head.weight`; that matrix is the
token embedding. Weights are read onto the target device instead of staging
a second full copy on CPU.
"""

import json
from pathlib import Path

import torch

from tinyserve.model.llama import LlamaForCausalLM


def weight_files(model_dir: Path) -> list[Path]:
    """Safetensors shards for a checkpoint directory, in index order when there is an index."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        names = sorted(set(index["weight_map"].values()))
        return [model_dir / name for name in names]
    single = model_dir / "model.safetensors"
    if single.exists():
        return [single]
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors weights in {model_dir}")
    return shards


def load_hf_weights(
    model: LlamaForCausalLM,
    model_dir: str | Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> None:
    """Copy checkpoint tensors into `model` on `device` in `dtype`.

    `lm_head.weight` is allowed to be absent when embeddings are tied. Any other
    missing or unexpected name is a real mismatch and raises.
    """
    from safetensors.torch import load_file

    model.to(device=device, dtype=dtype)
    state: dict[str, torch.Tensor] = {}
    for path in weight_files(Path(model_dir)):
        # Load straight onto the target device. A CPU staging copy would not fit
        # next to the model on an 8 GB GPU.
        shard = load_file(str(path), device=str(device))
        for name, tensor in shard.items():
            state[name] = tensor if tensor.dtype == dtype else tensor.to(dtype)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if model.config.tie_word_embeddings:
        missing = [name for name in missing if name != "lm_head.weight"]
        model.lm_head.weight = model.model.embed_tokens.weight
    if missing or unexpected:
        raise RuntimeError(f"weight name mismatch, missing={missing}, unexpected={unexpected}")
