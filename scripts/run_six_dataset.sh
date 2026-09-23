#!/usr/bin/env bash
set -euo pipefail
: "${MODEL:?set MODEL to a local checkpoint directory}"
: "${DATA_ROOT:?set DATA_ROOT to the dataset download directory}"
DEVICE="${DEVICE:-cuda:0}"
PYTHONPATH="${PYTHONPATH:-$(pwd)/src}" python -m bridge.six_dataset_runner \
  --model "$MODEL" \
  --truthfulqa "$DATA_ROOT/truthfulqa_multiple_choice" \
  --mmlu "$DATA_ROOT/mmlu/test" \
  --arc-easy "$DATA_ROOT/arc_easy/test" \
  --arc-challenge "$DATA_ROOT/arc_challenge/test" \
  --bbq "$DATA_ROOT/bbq" \
  --sorrybench "$DATA_ROOT/sorrybench/question.jsonl" \
  --out "${OUT:-results/six_dataset_run.json}" \
  --device "$DEVICE"
