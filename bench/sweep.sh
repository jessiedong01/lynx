#!/usr/bin/env bash
# Sweep Lynx policies for one model: throughput for each, optional GSM8K accuracy.
# Usage:   ./bench/sweep.sh <hf-model> [tensor-parallel-size]
# Example: ./bench/sweep.sh Qwen/Qwen3-30B-A3B-Instruct-2507 2
# Accuracy: RUN_ACC=1 ./bench/sweep.sh ...   (needs `pip install "lm_eval[api]"`;
#   set EVAL_BIN to its lm_eval if it lives in a separate environment)
# Grouped-topk models (GLM, DeepSeek) only have the generic "quant" policy,
# so sweep alpha instead:  ALPHAS="0.7 1 1.5 2 3 4" ./bench/sweep.sh ...
# Extra serve flags:       EXTRA_ARGS="--moe-backend triton" ./bench/sweep.sh ...
# Parallel sweeps on one node: give each its own PORT and RESULTS_DIR.
# PCIe-only nodes (e.g. L40S without NVLink) can hang after CUDA graph capture,
# with or without Lynx. Disable GPU peer-to-peer to avoid it:
#   NCCL_P2P_DISABLE=1 EXTRA_ARGS="--disable-custom-all-reduce" ./bench/sweep.sh ...
set -euo pipefail

MODEL=$1
TP=${2:-2}
PORT=${PORT:-8000}
OUT=${RESULTS_DIR:-results}/$(echo "$MODEL" | tr '/' '_')
mkdir -p "$OUT"

POLICIES=${POLICIES:-"do-nothing
quant_alpha0.7_beta1_optimized quant_alpha1_beta1_optimized quant_alpha1_beta2_optimized
quant_alpha1.125_beta2_optimized quant_alpha1.4_beta2_optimized quant_alpha1.5_beta2_optimized
quant_alpha1.5_beta3_optimized quant_alpha2_beta2_optimized quant_alpha2_beta3_optimized
quant_alpha2_beta4_optimized quant_alpha2.25_beta4_optimized quant_alpha3_beta2_optimized
quant_alpha3_beta3_optimized quant_alpha3_beta4_optimized quant_alpha3_beta6_optimized
quant_alpha4_beta5_optimized quant_alpha4_beta7_optimized quant_alpha6_beta8_optimized
quant_alpha0.7_optimized quant_alpha1_optimized quant_alpha1.125_optimized
quant_alpha2_optimized quant_alpha3_optimized quant_alpha4_optimized
quant_alpha5_optimized quant_alpha8_optimized quant_alpha16_optimized"}

if [ -n "${ALPHAS:-}" ]; then
  POLICIES="do-nothing $(for a in $ALPHAS; do printf 'quant@%s ' "$a"; done)"
fi

# Expert counts. Field names differ by family: Qwen uses num_experts,
# Mixtral uses num_local_experts, GLM/DeepSeek use n_routed_experts.
# Multimodal models nest them under text_config. Read the raw config.json
# so models newer than the installed transformers still work.
read -r TOPK NEXP < <(python - "$MODEL" <<'PY'
import json, sys
from huggingface_hub import hf_hub_download
c = json.load(open(hf_hub_download(sys.argv[1], "config.json")))
c = c.get("text_config", c)
e = next(c[k] for k in ("num_local_experts", "num_experts", "n_routed_experts") if c.get(k))
print(c["num_experts_per_tok"], e)
PY
)
echo "model=$MODEL top-k=$TOPK experts=$NEXP"

for P in $POLICIES; do
  NAME=${P%@*}; ALPHA=1
  [[ $P == *@* ]] && ALPHA=${P#*@}
  echo "=== $P ==="
  CFG="$OUT/$P.policy.json"
  cat > "$CFG" <<JSON
{
    "policy": "$NAME",
    "num_experts_per_tok": $TOPK,
    "num_local_experts": $NEXP,
    "alpha": $ALPHA,
    "beta": 0,
    "min_experts": 0,
    "threshold_percentile": 0,
    "count_of_topk": 0
}
JSON

  # setsid gives the server its own process group, so cleanup stops its
  # workers without touching another sweep running on the same node.
  VLLM_LYNX_ENABLED=1 VLLM_LYNX_CONFIG_FILE="$CFG" \
    setsid vllm serve "$MODEL" --tensor-parallel-size "$TP" --port "$PORT" ${EXTRA_ARGS:-} \
    > "$OUT/$P.server.log" 2>&1 &
  SERVER=$!

  # Wait up to 20 minutes. One failed policy should not end the sweep.
  up=0
  for _ in $(seq 240); do
    curl -sf "localhost:$PORT/health" > /dev/null && { up=1; break; }
    kill -0 $SERVER 2>/dev/null || break
    sleep 5
  done
  if [ $up = 0 ]; then
    echo "server failed for $P, see $OUT/$P.server.log"
    kill -- -$SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true
    sleep 10
    continue
  fi

  # Without this line the plugin is not active and the numbers mean nothing.
  grep -q "lynx: profiling complete" "$OUT/$P.server.log" \
    || echo "WARNING: lynx did not activate for $P"

  # Greedy decoding and a fixed output length give every policy identical
  # work. The warm-up run triggers Triton JIT compiles outside the timed run.
  BENCH="vllm bench serve --model $MODEL --port $PORT --dataset-name random
    --random-input-len 512 --random-output-len 256 --max-concurrency 64
    --temperature 0 --ignore-eos"
  $BENCH --num-prompts 64 > "$OUT/$P.warmup.log" 2>&1 || true
  $BENCH --num-prompts 500 \
    --save-result --result-dir "$OUT" --result-filename "$P.bench.json" \
    || echo "benchmark failed for $P"

  if [ "${RUN_ACC:-0}" = 1 ]; then
    # Full GSM8K at the benchmark's concurrency: Lynx remaps experts per
    # batch, so accuracy depends on how many requests share a batch.
    "${EVAL_BIN:-lm_eval}" --model local-completions \
      --model_args "model=$MODEL,base_url=http://localhost:$PORT/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False" \
      --tasks gsm8k --output_path "$OUT/$P.acc" > "$OUT/$P.acc.log" 2>&1 \
      || echo "accuracy failed for $P, see $OUT/$P.acc.log"
  fi

  kill -- -$SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true
  sleep 10
done

python - "$OUT" <<'PY'
import glob, json, os, sys
out = sys.argv[1]

def acc(p):
    files = glob.glob(f"{out}/{p}.acc/**/results_*.json", recursive=True)
    if not files:
        return None
    r = json.load(open(max(files)))["results"]["gsm8k"]
    return r.get("exact_match,flexible-extract")

rows = []
for f in sorted(glob.glob(f"{out}/*.bench.json")):
    p = os.path.basename(f).removesuffix(".bench.json")
    rows.append((p, json.load(open(f))["output_throughput"], acc(p)))
base = {p: (t, a) for p, t, a in rows}.get("do-nothing")
print(f"{'policy':36} {'tok/s':>8} {'speedup':>8} {'gsm8k':>7} {'change':>8}")
for p, t, a in sorted(rows, key=lambda r: -r[1]):
    s = f"{t / base[0]:7.2f}x" if base else "      -"
    g = f"{100 * a:6.1f}%" if a is not None else "      -"
    d = f"{100 * (a - base[1]):+7.1f}" if a is not None and base and base[1] is not None else "       -"
    print(f"{p:36} {t:8.1f} {s} {g} {d}")
PY
