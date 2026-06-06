#!/bin/bash
# ── Run this once at the top of your Colab notebook ──────────────────────────
# !bash colab_setup.sh

set -e

echo "=== Cloning repo ==="
git clone https://github.com/YOUR_USERNAME/tr-smart-home-llm-benchmark.git
cd tr-smart-home-llm-benchmark

echo "=== Installing dependencies ==="
pip install -q -r requirements.txt

echo "=== Ready. Example commands ==="
echo ""
echo "# Fine-tune LLaMA"
echo "python run.py --model llama --mode finetune --action train"
echo ""
echo "# Evaluate base LLaMA with zero-shot"
echo "python run.py --model llama --mode base --prompt zero_shot"
echo ""
echo "# Evaluate fine-tuned Gemma with few-shot"
echo "python run.py --model gemma --mode finetune --prompt few_shot"
echo ""
echo "# Evaluate general LLM with chain-of-thought"
echo "python run.py --model general --mode base --prompt cot"
echo ""
echo "# View results summary"
echo "cat results/summary.csv"
