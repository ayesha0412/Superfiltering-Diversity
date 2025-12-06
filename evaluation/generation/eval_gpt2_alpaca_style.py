import json
import os
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task.\n"
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

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
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--max_input_tokens", type=int, default=800)  # 🔹 NEW
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def build_prompt(dataset_name, sample, prompt_key):
    if dataset_name == "lima":
        conv = sample.get("conversations", sample.get(prompt_key))

        # If conv is a string instead of list → just use it directly
        if not isinstance(conv, list):
            instruction = str(conv)
        else:
            # conv is a list of {"from": ..., "value": ...}
            user_turns = [
                c["value"] for c in conv
                if isinstance(c, dict) and c.get("from") == "human"
            ]
            if user_turns:
                instruction = user_turns[0]
            else:
                instruction = conv[0]["value"]
    else:
        instruction = sample[prompt_key]

    return PROMPT_TEMPLATE.format(instruction=instruction)

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("➡️ Loading GPT-2 tokenizer and model from:", args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.to(device)
    model.eval()

    data_path, prompt_key = DATASETS[args.dataset_name]

    print("➡️ Reading test data from:", data_path)
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            samples.append(json.loads(line))

    print(f"✅ Loaded {len(samples)} samples. Generating...")

    results = []
    for sample in tqdm(samples):
        prompt = build_prompt(args.dataset_name, sample, prompt_key)

        # Tokenize FIRST (on CPU), then truncate if needed
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]

        # 🔹 Truncate very long prompts to last max_input_tokens
        max_inp = args.max_input_tokens
        if input_ids.shape[1] > max_inp:
            input_ids = input_ids[:, -max_inp:]
            attn_mask = attn_mask[:, -max_inp:]

        inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attn_mask.to(device),
        }

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_text = tokenizer.decode(
            gen_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

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

    save_name = f"{args.dataset_name}_gpt2_eval.json"
    out_path = os.path.join(out_dir, save_name)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("✅ Saved results to:", out_path)


if __name__ == "__main__":
    main()
