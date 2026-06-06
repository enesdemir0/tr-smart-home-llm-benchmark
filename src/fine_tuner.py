"""
Supervised fine-tuning with QLoRA (PEFT + TRL).

Saves the LoRA adapter to:  results/<model_key>_finetuned/
"""

import json
from pathlib import Path
from src.prompter import SYSTEM_PROMPT, apply_chat_template_compat


# ── Dataset formatting ────────────────────────────────────────────────────────

def format_row(row: dict, tokenizer) -> dict:
    """Convert a train row into the model's chat-template format."""
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": row["instruction"]},
        {"role": "assistant", "content": row["response"]},
    ]
    text = apply_chat_template_compat(
        tokenizer, messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


# ── Fine-tuning entry point ───────────────────────────────────────────────────

def fine_tune(
    model_key: str,
    train_data: list[dict],
    results_dir: str = "results",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
    load_in_4bit: bool = True,
):
    """
    Args:
        model_key    : "llama" | "gemma"  (general LLM is NOT fine-tuned)
        train_data   : list of training rows from data_loader
        results_dir  : adapter saved here as <model_key>_finetuned/
        num_epochs   : SFT epochs
        batch_size   : per-device batch size
        learning_rate: LoRA learning rate
        max_seq_length: truncation length
        load_in_4bit : 4-bit QLoRA (recommended on Colab)
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import Dataset

    from src.model_loader import get_model_id

    base_id = get_model_id(model_key)
    output_dir = Path(results_dir) / f"{model_key}_finetuned"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {base_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Model ─────────────────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if load_in_4bit else None

    print(f"Loading base model: {base_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA config ───────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    formatted = [format_row(row, tokenizer) for row in train_data]
    dataset = Dataset.from_list(formatted)

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=20,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        warmup_steps=50,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )

    print("Starting fine-tuning...")
    trainer.train()

    # Save only the LoRA adapter
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Adapter saved → {output_dir}")
