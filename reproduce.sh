#!/usr/bin/env bash
# Regenerate the distractor-robustness sweep for both models.
# Usage: ./reproduce.sh PORT_15B PORT_05B
set -euo pipefail
cd "$(dirname "$0")"
P15="${1:-8081}"; P05="${2:-8082}"
. .venv/bin/activate
[ -f data/gsm8k_test.jsonl ] || { mkdir -p data; curl -sL https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl -o data/gsm8k_test.jsonl; }
python tools/run_sweep.py --port "$P15" --model qwen-ds   --n 150 --out results/ds_15b.jsonl
python tools/run_sweep.py --port "$P05" --model qwen-ds05 --n 150 --out results/ds_05b.jsonl
python tools/analyze.py
python tools/verify.py
echo "regenerated results; see bench_results/frontier.md"
