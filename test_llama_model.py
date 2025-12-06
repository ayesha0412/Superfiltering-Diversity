import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "llama2_superfiltering_10per"   # folder that was saved after training

def main():
    # 1. Load tokenizer & model
    print("➡️ Loading tokenizer and model from:", MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    print("✅ Loaded model for inference.")

    # 2. Build a prompt (same format as training)
    instruction = "Explain what the Superfiltering method does in simple terms."
    inp = ""  # no extra input for this example

    if inp.strip():
        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            "### Response:\n"
        )
    else:
        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            "### Response:\n"
        )

    print("\n📝 Prompt going into the model:\n", prompt)

    # 3. Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 4. Generate
    print("\n🚀 Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n📤 Raw model output:\n")
    print(full_text)

    # ---- Extract only the first answer ----
    if "### Response:" in full_text:
        response_section = full_text.split("### Response:")[-1]
        # stop if the model started another instruction
        if "### Instruction:" in response_section:
            response_section = response_section.split("### Instruction:")[0]

        print("\n✨ Extracted answer:\n")
        print(response_section.strip())
    else:
        print("\n(no '### Response:' marker found)")



if __name__ == "__main__":
    main()
