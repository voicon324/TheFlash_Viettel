#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1/completions}"
MODEL="${MODEL:-Qwen3.5-2B}"
OUT_DIR="${OUT_DIR:-track3/results/gpqa_diamond}"

mkdir -p "$OUT_DIR"

lm_eval \
  --model local-completions \
  --tasks gpqa_diamond_cot_zeroshot \
  --model_args "model=${MODEL},base_url=${BASE_URL},num_concurrent=8,max_retries=3,tokenized_requests=False" \
  --batch_size 1 \
  --output_path "$OUT_DIR"
