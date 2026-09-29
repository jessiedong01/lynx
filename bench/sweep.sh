#!/usr/bin/env bash
# Sweep Lynx policies for one model: throughput for each, optional GSM8K accuracy.
# Usage:   ./bench/sweep.sh <hf-model> [tensor-parallel-size]
# Example: ./bench/sweep.sh Qwen/Qwen3-30B-A3B-Instruct-2507 2
# Accuracy: RUN_ACC=1 ./bench/sweep.sh ...   (needs `pip install lm_eval`)
# Grouped-topk models (GLM, DeepSeek) only have the generic "quant" policy,
# so sweep alpha instead:  ALPHAS="0.7 1 1.5 2 3 4" ./bench/sweep.sh ...
# Extra serve flags:       EXTRA_ARGS="--moe-backend triton" ./bench/sweep.sh ...
# PCIe-only nodes (e.g. L40S without NVLink) can hang after CUDA graph capture,
# with or without Lynx. Disable GPU peer-to-peer to avoid it:
#   NCCL_P2P_DISABLE=1 EXTRA_ARGS="--disable-custom-all-reduce" ./bench/sweep.sh ...
set -euo pipefail

MODEL=$1
TP=${2:-2}
PORT=8000
OUT=results/$(echo "$MODEL" | tr '/' '_')
mkdir -p "$OUT"

POLICIES=${POLICIES:-"do-nothing
quant_alpha0.7_beta1_optimized quant_alpha1_beta1_optimized quant_alpha1_beta2_optimized
quant_alpha1.125_beta2_optimized quant_alpha1.4_beta2_optimized quant_alpha1.5_beta2_optimized
quant_alpha1.5_beta3_optimized quant_alpha2_beta2_optimized quant_alpha2_beta3_optimized
quant_alpha2_beta4_optimized quant_alpha2.25_beta4_optimized quant_alpha3_beta2_optimized
quant_alpha3_beta3_optimized quant_alpha3_beta4_optimized quant_alpha3_beta6_optimized
quant_alpha4_beta5_optimized quant_alpha4_beta7_optimized quant_alpha6_beta8_optimized"}

if [ -n "${ALPHAS:-}" ]; then
  POLICIES="do-nothing $(for a in $ALPHAS; do printf 'quant@%s ' "$a"; done)"
fi

# Expert counts. Field names differ by family: Qwen uses num_experts,
# Mixtral uses num_local_experts, GLM/DeepSeek use n_routed_experts.
read -r TOPK NEXP < <(python - "$MODEL" <<'PY'
import sys
from transformers import AutoConfig
c = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
e = next(getattr(c, k) for k in ("num_local_experts", "num_experts", "n_routed_experts") if getattr(c, k, None))
print(c.num_experts_per_tok, e)
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

  VLLM_LYNX_ENABLED=1 VLLM_LYNX_CONFIG_FILE="$CFG" \
    vllm serve "$MODEL" --tensor-parallel-size "$TP" --port "$PORT" ${EXTRA_ARGS:-} \
    > "$OUT/$P.server.log" 2>&1 &
  SERVER=$!

  until curl -sf "localhost:$PORT/health" > /dev/null; do
    kill -0 $SERVER 2>/dev/null || { echo "server died, see $OUT/$P.server.log"; exit 1; }
    sleep 5
  done

  # Without this line the plugin is not active and the numbers mean nothing.
  grep -q "lynx: profiling complete" "$OUT/$P.server.log" \
    || echo "WARNING: lynx did not activate for $P"

  vllm bench serve --model "$MODEL" --port "$PORT" \
    --dataset-name random --random-input-len 512 --random-output-len 256 \
    --num-prompts 500 --max-concurrency 64 \
    --save-result --result-dir "$OUT" --result-filename "$P.bench.json"

  if [ "${RUN_ACC:-0}" = 1 ]; then
    lm_eval --model local-completions \
      --model_args "model=$MODEL,base_url=http://localhost:$PORT/v1/completions,num_concurrent=32" \
      --tasks gsm8k --limit 250 --output_path "$OUT/$P.acc"
  fi

  kill $SERVER; wait $SERVER 2>/dev/null || true
done

python - "$OUT" <<'PY'
import glob, json, os, sys
out = sys.argv[1]
rows = []
for f in sorted(glob.glob(f"{out}/*.bench.json")):
    p = os.path.basename(f).removesuffix(".bench.json")
    rows.append((p, json.load(open(f))["output_throughput"]))
base = dict(rows).get("do-nothing")
print(f"{'policy':40} {'tok/s':>10} {'speedup':>8}")
for p, t in sorted(rows, key=lambda r: -r[1]):
    print(f"{p:40} {t:10.1f} {t/base if base else 0:8.2f}x")
PY
