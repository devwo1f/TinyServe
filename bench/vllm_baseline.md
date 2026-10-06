# vLLM baseline

vLLM is not a dependency of TinyServe. It lives in its own virtualenv so its torch build cannot replace the project one.

The virtualenv is `/home/abhay/venvs/tinyserve-vllm` (outside the repo). Created and installed on 2026-10-06.

## Install

```bash
uv venv /home/abhay/venvs/tinyserve-vllm --python 3.11
uv pip install --python /home/abhay/venvs/tinyserve-vllm/bin/python \
  --index-strategy unsafe-best-match \
  https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl \
  --extra-index-url https://download.pytorch.org/whl/cu129
uv pip install --python /home/abhay/venvs/tinyserve-vllm/bin/python "torchcodec==0.14.0"
```

`unsafe-best-match` is required because the CUDA 12.9 PyTorch index publishes an old `packaging` and hides the PyPI release that flashinfer asks for.

`torchcodec==0.17.0` is what that solve picks, and its library links `libnvrtc.so.13`. The installed torch is `2.13.0+cu129`, which ships `libnvrtc.so.12`. Pinning torchcodec to 0.14.0 lets `import vllm` succeed.

## Version

| Package | Version |
|---|---|
| vLLM | 0.31.0+cu129 (`vllm.__version__` is `0.31.0`) |
| torch | 2.13.0+cu129 |
| torchcodec | 0.14.0 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, driver 595.97 |

## Run

From the repo root:

```bash
HF_HUB_OFFLINE=1 /home/abhay/venvs/tinyserve-vllm/bin/python -m bench.vllm_offline \
  --model models/Llama-3.2-1B-Instruct \
  --output-dir docs/results/phase2
```

Defaults match the P2.3 synthetic set: 8 requests, 64 prompt tokens, 32 new tokens, seed 0, warmup 2, 3 repeats, bf16, temperature 0, `ignore_eos`. One `LLM.generate` call holds the whole set.

The script also passes `max_model_len=512`, `gpu_memory_utilization=0.85`, `max_num_seqs=8`, `enable_prefix_caching=False`, and `disable_log_stats=False`. The offline `LLM` class turns stats off unless that last flag is set, and then the per-request clocks stay empty.

Before importing vLLM the script sets `VLLM_USE_FLASHINFER_SAMPLER=0` when it is unset. FlashInfer's sampler JIT-compiles with `nvcc`, and this machine has no CUDA toolkit. Greedy sampling does not use that kernel.

The result file is `docs/results/phase2/2026-10-06_p2-4-vllm-offline.jsonl`. Throughput is the summary line. Inter-token latencies are null: vLLM's request stats expose the first token and the last token, not each gap.
