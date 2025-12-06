import json
import os
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Alpaca-style prompt template (similar to your training script)
PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "{input_block}"
    "### Response:\n"
)

# Map each test set to its file path and key for the instruction text
DATASETS = {
    "vicuna":    ("evaluation/test_data/vicuna_test_set.jsonl",   "text"),
    "koala":     ("evaluation/test_data/koala_test_set.jsonl",    "prompt"),
    "wizardlm":  ("evaluation/test_data/wizardlm_test_set.jsonl", "Instruction"),
    "sinstruct": ("evaluation/test_data/sinstruct_test_set.jsonl","instruction"),
    "lima":      ("evaluation/test_data/lima_test_set.jsonl",     "conversations"),
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str, required=True,
                   choices=list(DATASETS.keys()))
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def build_prompt(dataset_name, sample, prompt_key):
    # For most sets, just grab the text field as "instruction"
    if dataset_name == "lima":
        conv = sample["conversations"]
        user_turns = [c["value"] for c in conv if c.get("from") == "human"]
        instruction = user_turns[0] if user_turns else conv[0]["value"]
        input_block = ""
    else:
        instruction = sample[prompt_key]
        input_block = ""

    prompt = PROMPT_TEMPLATE.format(
        instruction=instruction,
        input_block=input_block,
    )
    return prompt

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading tokenizer and model from:", args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    data_path, prompt_key = DATASETS[args.dataset_name]

    print("Reading test data from:", data_path)
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples. Generating...")

    results = []
    for sample in tqdm(samples):
        prompt = build_prompt(args.dataset_name, sample, prompt_key)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_text = tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # everything after "### Response:" is the answer
        if "### Response:" in full_text:
            answer = full_text.split("### Response:")[-1].strip()
        else:
            answer = full_text

        sample["prompt"] = prompt
        sample["raw_output"] = full_text
        sample["response"] = answer
        results.append(sample)

    out_dir = os.path.join(args.model_name_or_path, "test_inference")
    os.makedirs(out_dir, exist_ok=True)

    save_name = f"{args.dataset_name}_llama2_superfilter_10per.json"
    out_path = os.path.join(out_dir, save_name)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Saved results to:", out_path)


if __name__ == "__main__":
    main()