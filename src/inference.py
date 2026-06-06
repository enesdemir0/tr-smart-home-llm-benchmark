"""
Run a loaded model on the test set and collect raw outputs + timing.
"""

import json
import time
import torch
from src.prompter import build_prompt


def run_inference(
    model,
    tokenizer,
    test_data: list[dict],
    prompt_type: str,
    few_shot_examples: list[dict] | None = None,
    max_new_tokens: int = 128,
) -> list[dict]:
    """
    Args:
        model             : loaded HuggingFace model
        tokenizer         : matching tokenizer
        test_data         : list of test rows (instruction, context, response, intent_id)
        prompt_type       : "zero_shot" | "few_shot" | "cot"
        few_shot_examples : required when prompt_type == "few_shot"
        max_new_tokens    : generation budget

    Returns:
        list of result dicts (one per test row) with keys:
            instruction, gold_response, pred_response,
            latency_s, input_tokens, output_tokens
    """
    results = []
    device = next(model.parameters()).device

    for i, row in enumerate(test_data):
        messages = build_prompt(prompt_type, row["instruction"], few_shot_examples)

        # Apply the model's chat template
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - t0

        # Decode only the newly generated tokens
        new_ids = output_ids[0][input_len:]
        pred_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        results.append({
            "instruction"   : row["instruction"],
            "gold_response" : row["response"],
            "pred_response" : pred_text,
            "latency_s"     : round(latency, 4),
            "input_tokens"  : input_len,
            "output_tokens" : len(new_ids),
        })

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(test_data)}")

    return results


def save_results(results: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved → {path}")
