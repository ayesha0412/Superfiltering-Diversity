import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

# ====== CONFIG ======
BASE_MODEL = "meta-llama/Llama-2-7b-hf"
DATA_PATH = "alpaca_data_kmeans_hybrid_10per.json"
OUTPUT_DIR = "llama2_hybrid_kmeans_10per"
MAX_SEQ_LEN = 1024

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides further context.\n"
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "{input_block}"
    "### Response:\n{response}\n"
)

def main():
    print("➡️ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    print("➡️ Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # IMPORTANT for gradient checkpointing (HF requirement)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    print("➡️ Loading dataset:", DATA_PATH)
    dataset = load_dataset("json", data_files=DATA_PATH)
    train_dataset = dataset["train"]

    print("➡️ Formatting and tokenizing...")

    def format_and_tokenize(example):
        instruction = example.get("instruction", "")
        inp = example.get("input", "")
        output = example.get("output", "")

        input_block = f"### Input:\n{inp}\n\n" if inp else ""
        prompt = PROMPT_TEMPLATE.format(
            instruction=instruction,
            input_block=input_block,
            response=output,
        )

        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=False,
        )

        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    train_dataset = train_dataset.map(
        format_and_tokenize,
        remove_columns=train_dataset.column_names,
    )

    print("➡️ Preparing training args...")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        warmup_ratio=0.03,
        fp16=False,   # KEEP DISABLED TO FIX GRADSCALER ERROR
        save_strategy="epoch",
        logging_steps=20,
        optim="adafactor",
        remove_unused_columns=False,
        report_to="none",
    )

    print("➡️ Creating trainer...")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("🚀 Training...")
    trainer.train()

    print("➡️ Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("✅ DONE. Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
