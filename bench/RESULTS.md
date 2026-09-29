# Benchmark results

Throughput from `bench/sweep.sh`. Speedup is relative to the `do-nothing`
policy on the same hardware and vLLM version, so the plugin is loaded in
both runs.

## Qwen3-30B-A3B-Instruct-2507, vLLM 0.20.1 vs 0.30.0

| vLLM | `do-nothing` tok/s | `quant_alpha3_beta2_optimized` tok/s | Speedup (run 1, run 2) | Median TPOT (ms) |
|---|---|---|---|---|
| 0.20.1 | 1627.9 | 1907.1 | 1.17x (1.165, 1.178) | 37.0 → 31.6 |
| 0.30.0 | 1659.5 | 1869.1 | 1.13x (1.115, 1.137) | 37.1 → 31.4 |

Throughput is the mean of two runs. The 0.30.0 row uses the ported
patches; the 0.20.1 row runs the same ported code, so it also checks that
the port kept 0.20.1 working. The two versions' ranges don't overlap, so
the ~4% gap between them looks real. Its cause is not yet known.

Setup:

- Hardware: 2× NVIDIA L40S (Stanford FarmShare, PCIe, no NVLink), TP=2, BF16
- Workload: `vllm bench serve`, random dataset, 512 input / 256 output
  tokens, `--temperature 0 --ignore-eos`, 500 prompts after a 64-prompt
  warm-up, max concurrency 64. Every run generates exactly 128,000 tokens.
- Flags: `NCCL_P2P_DISABLE=1`, `--disable-custom-all-reduce`,
  `--max-model-len 8192`, `VLLM_USE_FLASHINFER_SAMPLER=0`,
  `VLLM_ALLREDUCE_USE_FLASHINFER=0`
- Accuracy: not yet measured

Notes:

- These GPUs hang after CUDA graph capture unless GPU peer-to-peer is off,
  with or without Lynx.
- vLLM 0.30's FlashInfer sampler and all-reduce JIT-compile on first use,
  which needs `nvcc`. The two FlashInfer flags above avoid that.
- An earlier run reported 1.25x on vLLM 0.20.1. It used sampled decoding
  without `--ignore-eos`, so the two policies generated different amounts
  of work. The table above replaces it.

## Pending

- Find the cause of the 0.30.0 vs 0.20.1 gap.
- Qwen3.8-Flash-Next: full policy sweep with accuracy. Needs vLLM 0.30+
  and about 4 GPUs.
- GLM-5.3-Flash: not supported yet. It uses bias-corrected routing
  (`e_score_correction_bias`, `routed_scaling_factor`), which the Lynx
  kernels don't apply. See docs/MODELS.md.
