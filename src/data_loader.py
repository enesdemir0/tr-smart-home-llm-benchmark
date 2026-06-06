import json
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_train(data_dir: str = "data") -> list[dict]:
    return load_jsonl(Path(data_dir) / "train.jsonl")


def load_test(data_dir: str = "data") -> list[dict]:
    return load_jsonl(Path(data_dir) / "test.jsonl")


def get_few_shot_examples(data_dir: str = "data", n: int = 5) -> list[dict]:
    """Return n representative examples from train set (one per domain)."""
    train = load_train(data_dir)
    seen_services = set()
    examples = []
    for row in train:
        service = json.loads(row["response"])["service"]
        if service not in seen_services:
            seen_services.add(service)
            examples.append(row)
        if len(examples) >= n:
            break
    return examples
