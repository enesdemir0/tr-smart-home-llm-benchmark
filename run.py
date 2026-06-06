"""
Main entry point for the Turkish Smart-Home LLM Benchmark.

Usage examples
--------------
# Evaluate LLaMA base model with zero-shot prompting
python run.py --model llama --mode base --prompt zero_shot

# Fine-tune Gemma on training data
python run.py --model gemma --mode finetune --action train

# Evaluate fine-tuned Gemma with few-shot prompting
python run.py --model gemma --mode finetune --prompt few_shot

# Evaluate general LLM with chain-of-thought
python run.py --model general --mode base --prompt cot
"""

import argparse
import json
import csv
from pathlib import Path

from src.data_loader  import load_train, load_test, get_few_shot_examples
from src.evaluator    import evaluate, print_report


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="TR Smart-Home LLM Benchmark")

    parser.add_argument("--model",  required=True,
                        choices=["llama", "gemma", "general"],
                        help="Which model to use")

    parser.add_argument("--mode",   required=True,
                        choices=["base", "finetune"],
                        help="base = pretrained only | finetune = load LoRA adapter")

    parser.add_argument("--action", default="eval",
                        choices=["train", "eval"],
                        help="train = run SFT | eval = run inference + metrics (default: eval)")

    parser.add_argument("--prompt", default="zero_shot",
                        choices=["zero_shot", "few_shot", "cot"],
                        help="Prompting strategy (only used during eval)")

    parser.add_argument("--data_dir",    default="data",
                        help="Path to data directory (default: data/)")
    parser.add_argument("--results_dir", default="results",
                        help="Path to results directory (default: results/)")

    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--no_4bit",    action="store_true",
                        help="Disable 4-bit quantisation (use full precision)")

    return parser.parse_args()


# ── Train ────────────────────────────────────────────────────────────────────

def run_train(args):
    if args.model == "general":
        raise ValueError("The general LLM is not fine-tuned in this benchmark.")

    from src.fine_tuner import fine_tune

    print(f"\n{'='*60}")
    print(f" FINE-TUNING  model={args.model}")
    print(f"{'='*60}\n")

    train_data = load_train(args.data_dir)
    print(f"Training samples: {len(train_data)}")

    fine_tune(
        model_key=args.model,
        train_data=train_data,
        results_dir=args.results_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        load_in_4bit=not args.no_4bit,
    )


# ── Eval ─────────────────────────────────────────────────────────────────────

def run_eval(args):
    from src.model_loader import load_model_and_tokenizer
    from src.inference    import run_inference, save_results

    label = f"{args.model}__{args.mode}__{args.prompt}"

    print(f"\n{'='*60}")
    print(f" EVALUATION  {label}")
    print(f"{'='*60}\n")

    # Load data
    test_data = load_test(args.data_dir)
    few_shot_examples = (
        get_few_shot_examples(args.data_dir, n=5)
        if args.prompt == "few_shot" else None
    )
    print(f"Test samples: {len(test_data)}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        model_key=args.model,
        mode=args.mode,
        results_dir=args.results_dir,
        load_in_4bit=not args.no_4bit,
    )

    # Run inference
    results = run_inference(
        model=model,
        tokenizer=tokenizer,
        test_data=test_data,
        prompt_type=args.prompt,
        few_shot_examples=few_shot_examples,
    )

    # Save raw results
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    raw_path = results_dir / f"{label}__raw.json"
    save_results(results, str(raw_path))

    # Compute and display metrics
    metrics = evaluate(results)
    metrics["model"]   = args.model
    metrics["mode"]    = args.mode
    metrics["prompt"]  = args.prompt

    print_report(metrics, label=label)

    # Append to summary CSV
    csv_path = results_dir / "summary.csv"
    fieldnames = ["model", "mode", "prompt", "n_samples",
                  "valid_json_rate", "label_validity_rate",
                  "accuracy", "precision", "recall", "f1",
                  "avg_latency_s", "avg_input_tokens", "avg_output_tokens", "avg_total_tokens"]

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: metrics.get(k, "") for k in fieldnames})

    print(f"\nMetrics appended → {csv_path}")
    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.action == "train":
        run_train(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
