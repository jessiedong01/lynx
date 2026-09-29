# Benchmark results

Throughput from `bench/sweep.sh`. Speedup is relative to the `do-nothing`
policy on the same hardware, so the plugin is loaded in both runs.

## Qwen3-30B-A3B-Instruct-2507 (baseline reproduction)

| Policy | Output tok/s | Speedup |
|---|---|---|
| `do-nothing` | 1290.0 | 1.00x |
| `quant_alpha3_beta2_optimized` | 1606.2 | 1.25x |

- Hardware: 2× NVIDIA L40S (Stanford FarmShare), TP=2, BF16
- Software: vLLM 0.20.1, lynx-vllm main
- Workload: `vllm bench serve`, random dataset, 512 input / 256 output tokens,
  500 prompts, max concurrency 64
- Flags: `NCCL_P2P_DISABLE=1`, `--disable-custom-all-reduce`, `--max-model-len 8192`
- Accuracy: not yet measured

## Pending

- Qwen3.8-Flash-Next (needs vLLM >= 0.29)
- GLM-5.3-Flash (grouped-topk path: alpha-only sweep via the `quant` policy)
