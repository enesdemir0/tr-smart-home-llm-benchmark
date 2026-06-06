"""
Load a HuggingFace model in base or fine-tuned (LoRA adapter) mode.

Supported model keys  (pass via --model CLI arg):
  llama    →  meta-llama/Llama-3.2-3B-Instruct
  gemma    →  google/gemma-2-2b-it
  general  →  mistralai/Mistral-7B-Instruct-v0.3

Fine-tuned checkpoints are expected at:
  results/<model_key>_finetuned/
"""

from pathlib import Path
import torch

MODEL_MAP = {
    "llama"  : "meta-llama/Llama-3.2-3B-Instruct",
    "gemma"  : "google/gemma-2-2b-it",
    "general": "mistralai/Mistral-7B-Instruct-v0.3",
}


def get_model_id(model_key: str) -> str:
    if model_key not in MODEL_MAP:
        raise ValueError(f"Unknown model key: {model_key!r}. Choose: {list(MODEL_MAP)}")
    return MODEL_MAP[model_key]


def load_model_and_tokenizer(
    model_key: str,
    mode: str = "base",
    results_dir: str = "results",
    load_in_4bit: bool = True,
):
    """
    Args:
        model_key    : "llama" | "gemma" | "general"
        mode         : "base" | "finetune"
        results_dir  : root folder where fine-tuned adapters are saved
        load_in_4bit : use 4-bit quantisation (recommended on Colab)

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    base_id = get_model_id(model_key)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if load_in_4bit else None

    print(f"Loading tokenizer: {base_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if mode == "base":
        print(f"Loading base model: {base_id}")
        model = AutoModelForCausalLM.from_pretrained(
            base_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    elif mode == "finetune":
        from peft import PeftModel
        adapter_path = Path(results_dir) / f"{model_key}_finetuned"
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Fine-tuned adapter not found at {adapter_path}. "
                "Run with --mode finetune first to train."
            )
        print(f"Loading base model: {base_id}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, str(adapter_path))

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose: base | finetune")

    model.eval()
    return model, tokenizer
