# TinyServe References

Papers and projects consulted. Code is never copied from other projects; when an idea is borrowed conceptually, it is recorded under "Conceptual borrowing" with a link and a one-line description.

## Papers

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM, SOSP 2023). https://arxiv.org/abs/2309.06180
- Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models" (OSDI 2022). https://www.usenix.org/conference/osdi22/presentation/yu
- Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve" (OSDI 2024). https://arxiv.org/abs/2403.02310
- Dao et al., "FlashAttention" (NeurIPS 2022) https://arxiv.org/abs/2205.14135 and "FlashAttention-2" (2023) https://arxiv.org/abs/2307.08691
- Dao et al., "Flash-Decoding for long-context inference" (blog post, 2023). https://crfm.stanford.edu/2023/10/12/flashdecoding.html
- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (ICML 2023). https://arxiv.org/abs/2211.17192
- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (2023). https://arxiv.org/abs/2302.01318
- Liu et al., "Optimizing Speculative Decoding for Serving Large Language Models Using Goodput" (SmartSpec, 2024). https://arxiv.org/abs/2406.14066
- "AdaSpec: Adaptive Speculative Decoding for Fast, SLO-Aware Large Language Model Serving" (SoCC 2025).
- "Nightjar: Dynamic Adaptive Speculative Decoding for Large Language Models Serving" (2025).
- Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving" (OSDI 2024). https://arxiv.org/abs/2401.09670
- Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs" (RadixAttention). https://arxiv.org/abs/2312.07104

Links for AdaSpec and Nightjar are added when the papers are read.

## Projects (reference reading only)

- vLLM: https://github.com/vllm-project/vllm
- SGLang: https://github.com/sgl-project/sglang
- Nano-vLLM: https://github.com/GeeeekExplorer/nano-vllm
- PyTorch gpt-fast: https://github.com/pytorch-labs/gpt-fast
- FlashInfer: https://github.com/flashinfer-ai/flashinfer
- Triton tutorials (fused softmax, matmul, layer norm): https://triton-lang.org/main/getting-started/tutorials/index.html

## Conceptual borrowing

| Date | Task | Source | What was borrowed (conceptually) |
|---|---|---|---|
| 2026-09-29 | P1.2 | Llama 3 RoPE scaling, as documented in the checkpoint config and checked against `transformers` | Long wavelengths are divided by `factor`, short wavelengths are unchanged, and the band between `low_freq_factor` and `high_freq_factor` is blended. No code was copied. |
| 2026-10-07 | P3.6 | vLLM automatic prefix caching, described in the PagedAttention line of work and the vLLM docs (https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | A full block's identity is a hash of the parent block's hash and the token ids in the block. Unused cached blocks are evicted LRU. No code was copied. |
| 2026-10-07 | P3.7 | Kwon et al., PagedAttention (already listed above) | A contiguous cache reserves the maximum sequence length for every live request. Paging only holds the blocks those tokens need, so the unused part is the tail of the last block. No code was copied. |
