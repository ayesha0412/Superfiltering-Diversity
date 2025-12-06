import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your fine-tuned model
model_path = "./llama-finetuned"  # Your trained model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load test data (adjust path to your test file)
with open("wizardlm_test.json", "r") as f:  # Replace with your actual test file
    test_data = json.load(f)

results = []
max_length = 1024

for example in test_data[:100]:  # Adjust number as needed
    # Build prompt
    prompt = f"### Instruction:\n{example['instruction']}\n\n"
    if "input" in example and example["input"]:
        prompt += f"### Input:\n{example['input']}\n\n"
    prompt += "### Response:\n"
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    results.append({
        "instruction": example["instruction"],
        "input": example.get("input", ""),
        "output": generated.split("### Response:\n")[-1].strip()
    })
    
    print(f"Processed {len(results)}/{len(test_data[:100])}")

# Save results
import os
os.makedirs("cherry_alpaca_5_per/test_inference", exist_ok=True)

with open("cherry_alpaca_5_per/test_inference/wizardlm_1024.json", "w") as f:
    json.dump(results, f, indent=2)

print("✓ Predictions saved!")