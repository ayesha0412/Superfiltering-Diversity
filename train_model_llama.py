import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# ====== CONFIG ======
BASE_MODEL = "meta-llama/Llama-2-7b-hf"   # <-- CHANGE if you use some other LLaMA
DATA_PATH = "alpaca_data_gpt2_data_10per.json"  # output of step3
OUTPUT_DIR = "llama2_superfiltering_10per"
MAX_SEQ_LEN = 1024

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "{input_block}"
    "### Response:\n{response}\n"
)


def main():
    # ----- 1. Load tokenizer -----
    print("➡️  Loading tokenizer from:", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("   • pad_token was None, set to eos_token")

    print("✅ Tokenizer loaded:", tokenizer.__class__.__name__)

    # ----- 2. Load model -----
    print("➡️  Loading model (this may take a while)...")
    
    # Load in FP16 to save memory, we'll handle training carefully
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,  # Keep model in FP16 to save memory
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False  # Disable cache for training
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    print(f"✅ Model loaded: {model.__class__.__name__}")

    # ----- 3. Load dataset -----
    print("➡️  Loading dataset from:", DATA_PATH)
    dataset = load_dataset("json", data_files=DATA_PATH)
    train_dataset = dataset["train"]
    print("✅ Dataset loaded. Number of examples:", len(train_dataset))

    # ----- 4. Format dataset into prompts AND tokenize -----
    print("➡️  Formatting and tokenizing examples...")

    def format_and_tokenize(example):
        instruction = example.get("instruction", "")
        inp = example.get("input", "")
        output = example.get("output", "")

        if inp and len(inp.strip()) > 0:
            input_block = f"### Input:\n{inp}\n\n"
        else:
            input_block = ""

        prompt = PROMPT_TEMPLATE.format(
            instruction=instruction,
            input_block=input_block,
            response=output,
        )
        
        # Tokenize the full text
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=False,
            return_tensors=None,
        )
        
        # Set labels same as input_ids for causal LM training
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized

    # Tokenize all examples
    train_dataset = train_dataset.map(
        format_and_tokenize,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing dataset",
    )

    print("✅ Tokenization done.")
    print(f"   • Dataset has {len(train_dataset)} examples")
    print(f"   • First example has {len(train_dataset[0]['input_ids'])} tokens\n")

    # ----- 5. Training arguments -----
    print("➡️  Setting up training arguments...")
    
    # Tesla P40 memory-optimized configuration
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        fp16=False,  # Keep FP16 disabled for training loop
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="no",
        report_to="none",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        remove_unused_columns=False,
        gradient_checkpointing=False,  # Already enabled on model
        optim="adafactor",  # Memory-efficient optimizer
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
    )
    print(f"✅ TrainingArguments created (using Adafactor optimizer for memory efficiency).")

    # ----- 6. Data collator -----
    print("➡️  Creating data collator...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    print("✅ Data collator ready.")

    # ----- 7. Trainer -----
    print("➡️  Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    print("✅ Trainer initialized.")

    # ----- 8. Train -----
    print("🚀 Starting training...")
    trainer.train()
    print("✅ Training finished.")

    # ----- 9. Save model -----
    print("➡️  Saving model and tokenizer to:", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ All done. Model saved.")


if __name__ == "__main__":
    main()