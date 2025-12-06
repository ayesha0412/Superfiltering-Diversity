from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "./gpt2-finetuned"  # Your trained GPT-2
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

def generate(instruction, input_text=""):
    prompt = instruction
    if input_text:
        prompt += "\n" + input_text
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test it
print(generate("Write a poem about mountains"))