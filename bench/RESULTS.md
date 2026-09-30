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
the ~4% gap between them looks real.

The gap does not come from the new model runner. With
`VLLM_USE_V2_MODEL_RUNNER=0`, vLLM 0.30.0 gives 1.14x (1654.4 → 1882.7
tok/s), the same as with the default runner. It comes from other changes
in vLLM 0.30.

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

## Policy sweep: speedup and accuracy, vLLM 0.30.0

Qwen3-30B-A3B-Instruct-2507, same setup as above. Accuracy is GSM8K
(5-shot, flexible-extract, all 1,319 questions) at concurrency 64, served
by the same Lynx-enabled server. Each half of the sweep ran as its own job
on its own node, and every row is compared against the `do-nothing` run on
that node.

| Policy | tok/s | Speedup | GSM8K | Change (pts) |
|---|---|---|---|---|
| `do-nothing` (node A) | 1664.6 | 1.00x | 88.9% | — |
| `quant_alpha0.7_beta1_optimized` | 1936.4 | 1.16x | 85.4% | −3.5 |
| `quant_alpha3_beta2_optimized` | 1859.8 | 1.12x | 86.4% | −2.6 |
| `quant_alpha1.5_beta2_optimized` | 1820.2 | 1.09x | 84.9% | −4.0 |
| `quant_alpha1_beta1_optimized` | 1814.0 | 1.09x | 86.5% | −2.4 |
| `quant_alpha1.125_beta2_optimized` | 1796.7 | 1.08x | 87.3% | −1.6 |
| `quant_alpha1_beta2_optimized` | 1793.2 | 1.08x | 85.0% | −3.9 |
| `quant_alpha2_beta2_optimized` | 1768.9 | 1.06x | 85.1% | −3.9 |
| `quant_alpha1.5_beta3_optimized` | 1758.5 | 1.06x | 86.4% | −2.5 |
| `quant_alpha1.4_beta2_optimized` | 1741.0 | 1.05x | 85.0% | −3.9 |
| `quant_alpha3_beta3_optimized` | 1739.5 | 1.05x | 86.9% | −2.0 |
| `quant_alpha2.25_beta4_optimized` | 1732.6 | 1.04x | 86.6% | −2.4 |
| `quant_alpha2_beta4_optimized` | 1718.2 | 1.03x | 85.5% | −3.4 |
| `quant_alpha2_beta3_optimized` | 1711.9 | 1.03x | 87.0% | −1.9 |
| `quant_alpha3_beta4_optimized` | 1710.0 | 1.03x | 85.3% | −3.6 |
| `do-nothing` (node B) | 1672.6 | 1.00x | 88.0% | — |
| `quant_alpha4_beta5_optimized` | 1722.6 | 1.03x | 88.0% | −0.1 |
| `quant_alpha4_beta7_optimized` | 1676.0 | 1.00x | 86.1% | −2.0 |
| `quant_alpha3_beta6_optimized` | 1667.3 | 1.00x | 84.6% | −3.4 |
| `quant_alpha6_beta8_optimized` | 1653.8 | 0.99x | 85.4% | −2.7 |
| `quant_alpha8_optimized` | 1634.3 | 0.98x | 85.1% | −3.0 |
| `quant_alpha3_optimized` | 1628.9 | 0.97x | 85.4% | −2.7 |
| `quant_alpha4_optimized` | 1622.5 | 0.97x | 84.8% | −3.2 |
| `quant_alpha0.7_optimized` | 1611.4 | 0.96x | 85.6% | −2.4 |
| `quant_alpha5_optimized` | 1608.4 | 0.96x | 86.9% | −1.1 |
| `quant_alpha1.125_optimized` | 1583.4 | 0.95x | 86.1% | −2.0 |
| `quant_alpha16_optimized` | 1579.6 | 0.94x | 85.6% | −2.4 |
| `quant_alpha2_optimized` | 1573.9 | 0.94x | 85.1% | −3.0 |

`quant_alpha1_optimized` did not run: its server crashed at startup.
Node A's `do-nothing` accuracy comes from a follow-up run on the same
setup, because the first run's accuracy step failed.

Findings:

- The fastest policy is `quant_alpha0.7_beta1_optimized` at 1.16x, ahead of
  the current Qwen3 default (`quant_alpha3_beta2_optimized`, 1.12x).
- Accuracy drops 2–4 points for most policies, more than the paper's
  "under 1 point". No policy combines a real speedup with a drop under
  1 point. `quant_alpha1.125_beta2_optimized` (1.08x, −1.6) and
  `quant_alpha4_beta5_optimized` (1.03x, −0.1) come closest.
- The noise floor is about 1 point: the two identical `do-nothing` runs
  scored 88.9% and 88.0%.
- The alpha-only policies are slower than `do-nothing` here.
- The accuracy drop is not caused by the port. On vLLM 0.20.1 (original
  patches path, same setup), `quant_alpha3_beta2_optimized` scores 86.1%
  against 88.1% for `do-nothing` (−2.0 points), at 1.13x
  (1627.9 → 1838.3 tok/s). That matches vLLM 0.30.0 within noise.

## Pending

- Find which vLLM 0.30 change narrows the speedup.
- Check how the accuracy drop depends on concurrency (batch size).
- Qwen3.8-Flash-Next: full policy sweep with accuracy. Needs vLLM 0.30+
  and about 4 GPUs.
- GLM-5.3-Flash: not supported yet. It uses bias-corrected routing
  (`e_score_correction_bias`, `routed_scaling_factor`), which the Lynx
  kernels don't apply. See docs/MODELS.md.
