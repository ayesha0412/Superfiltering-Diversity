import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# === CONFIG ===
model_name = "gpt2"
train_file = "alpaca_data_gpt2_data_10per.json"  # Path to your filtered data
output_dir = "./gpt2-finetuned"
num_train_epochs = 3
per_device_train_batch_size = 4
max_length = 512

# === LOAD DATA ===
with open(train_file, "r") as f:
    data = json.load(f)

# Prepare text for LM training
def build_prompt(example):
    prompt = example["instruction"]
    if "input" in example and example["input"]:
        prompt += "\n" + example["input"]
    prompt += "\n" + example["output"]
    return {"text": prompt}

dataset = Dataset.from_list(data)
dataset = dataset.map(build_prompt)

# === TOKENIZER & MODEL ===
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # For GPT-2

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# === TRAINING ===
model = AutoModelForCausalLM.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    prediction_loss_only=True,
    fp16=True,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)