"""
Evaluation metrics for smart-home command inference.

Metrics computed per run:
  - valid_json_rate    : fraction of outputs that are parseable JSON
  - label_validity_rate: fraction of valid JSONs whose service is a known HA service
  - accuracy           : exact match of service + entity_id (+ extra params)
  - precision          : macro-averaged over service classes
  - recall             : macro-averaged over service classes
  - f1                 : macro-averaged over service classes
  - avg_latency_s      : mean seconds per inference
  - avg_input_tokens   : mean prompt token count
  - avg_output_tokens  : mean generated token count
  - avg_total_tokens   : mean total tokens (input + output)
"""

import json
from collections import defaultdict

# All valid Home Assistant service names in this benchmark
VALID_SERVICES = {
    "light.turn_on",
    "light.turn_off",
    "climate.set_hvac_mode",
    "climate.set_temperature",
    "fan.turn_on",
    "fan.turn_off",
    "fan.set_percentage",
    "media_player.turn_on",
    "media_player.turn_off",
    "media_player.media_play",
    "cover.open_cover",
    "cover.close_cover",
    "alarm_control_panel.alarm_arm_away",
    "alarm_control_panel.alarm_disarm",
    "input_boolean.turn_on",
    "input_boolean.turn_off",
    "scene.turn_on",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_response(text: str) -> dict | None:
    """Extract the first valid JSON object from model output."""
    text = text.strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def responses_match(pred: dict, gold: dict) -> bool:
    """True if pred contains all keys/values of gold."""
    return all(pred.get(k) == v for k, v in gold.items())


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate(results: list[dict]) -> dict:
    """
    Args:
        results: list of dicts, each with keys:
            instruction   – Turkish command
            gold_response – ground-truth JSON string
            pred_response – model output string
            latency_s     – inference time in seconds
            input_tokens  – token count of prompt
            output_tokens – token count of generation

    Returns:
        dict with all metrics
    """
    n = len(results)
    valid_count        = 0
    label_valid_count  = 0
    correct_count      = 0

    # per-service TP / FP / FN for precision / recall / F1
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    total_latency       = 0.0
    total_input_tokens  = 0
    total_output_tokens = 0

    for r in results:
        gold = json.loads(r["gold_response"])
        pred = parse_response(r["pred_response"])

        gold_service = gold.get("service", "")

        total_latency       += r.get("latency_s", 0.0)
        total_input_tokens  += r.get("input_tokens", 0)
        total_output_tokens += r.get("output_tokens", 0)

        if pred is None:
            fn[gold_service] += 1
            continue

        valid_count += 1
        pred_service = pred.get("service", "")

        # label validity: is the predicted service a real HA service?
        if pred_service in VALID_SERVICES:
            label_valid_count += 1

        # exact match
        if responses_match(pred, gold):
            correct_count += 1
            tp[gold_service] += 1
        else:
            fn[gold_service] += 1
            if pred_service != gold_service:
                fp[pred_service] += 1

    # ── Aggregate precision / recall / F1 (macro) ────────────────────────────
    all_services = set(list(tp.keys()) + list(fn.keys()) + list(fp.keys()))
    precisions, recalls, f1s = [], [], []

    for svc in all_services:
        p = tp[svc] / (tp[svc] + fp[svc]) if (tp[svc] + fp[svc]) > 0 else 0.0
        r = tp[svc] / (tp[svc] + fn[svc]) if (tp[svc] + fn[svc]) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    avg_input  = total_input_tokens  / n if n else 0
    avg_output = total_output_tokens / n if n else 0

    return {
        "n_samples"           : n,
        "valid_json_rate"     : round(valid_count / n, 4)             if n else 0,
        "label_validity_rate" : round(label_valid_count / n, 4)       if n else 0,
        "accuracy"            : round(correct_count / n, 4)           if n else 0,
        "precision"           : round(sum(precisions) / len(precisions), 4) if precisions else 0,
        "recall"              : round(sum(recalls)    / len(recalls),    4) if recalls    else 0,
        "f1"                  : round(sum(f1s)        / len(f1s),        4) if f1s        else 0,
        "avg_latency_s"       : round(total_latency / n, 4)           if n else 0,
        "avg_input_tokens"    : round(avg_input,  1),
        "avg_output_tokens"   : round(avg_output, 1),
        "avg_total_tokens"    : round(avg_input + avg_output, 1),
    }


def print_report(metrics: dict, label: str = "") -> None:
    header = f"  {'='*50}\n  {label}\n  {'='*50}" if label else ""
    if header:
        print(header)
    print(f"  Samples              : {metrics['n_samples']}")
    print(f"  Valid JSON rate      : {metrics['valid_json_rate']:.2%}")
    print(f"  Label validity rate  : {metrics['label_validity_rate']:.2%}")
    print(f"  Accuracy             : {metrics['accuracy']:.2%}")
    print(f"  Precision (macro)    : {metrics['precision']:.2%}")
    print(f"  Recall    (macro)    : {metrics['recall']:.2%}")
    print(f"  F1        (macro)    : {metrics['f1']:.2%}")
    print(f"  Avg latency          : {metrics['avg_latency_s']:.3f} s")
    print(f"  Avg input tokens     : {metrics['avg_input_tokens']:.0f}")
    print(f"  Avg output tokens    : {metrics['avg_output_tokens']:.0f}")
    print(f"  Avg total tokens     : {metrics['avg_total_tokens']:.0f}")
