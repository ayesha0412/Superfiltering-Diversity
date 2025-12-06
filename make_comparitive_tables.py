import os
import json
from glob import glob

import numpy as np
import pandas as pd


# ===== Where eval jsons live =====
GPT2_DIR = os.path.join("gpt2-finetuned", "test_inference")
GPT2_SUFFIX = "_gpt2_eval.json"

LLAMA_DIR = os.path.join("llama2_superfiltering_10per", "test_inference")
LLAMA_SUFFIX = "_llama2_superfilter_10per.json"


def safe_get_response(sample):
    """Try common keys to get the model's final answer text."""
    for key in ["response", "output", "answer", "model_output"]:
        if key in sample and isinstance(sample[key], str):
            return sample[key]
    return ""


def safe_get_prompt(sample):
    """Prefer full prompt; fallback to instruction+input."""
    if "prompt" in sample and isinstance(sample["prompt"], str):
        return sample["prompt"]

    instr = sample.get("instruction", "")
    inp = sample.get("input", "")
    if inp:
        return instr + "\n" + inp
    return instr


def word_stats(texts):
    words = []
    for t in texts:
        if not isinstance(t, str):
            continue
        words.extend(t.split())

    total_words = len(words)
    if total_words == 0:
        return 0.0, 0.0

    avg_len = total_words / max(1, len(texts))
    vocab_size = len(set(words))
    lex_div = vocab_size / total_words if total_words > 0 else 0.0
    return avg_len, lex_div


def analyze_file(path, model_label):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = [safe_get_prompt(s) for s in data]
    responses = [safe_get_response(s) for s in data]

    num_samples = len(data)

    avg_prompt_words, prompt_lex_div = word_stats(prompts)
    avg_resp_words, resp_lex_div = word_stats(responses)

    empty_responses = sum(1 for r in responses if len(r.strip()) == 0)
    empty_pct = 100.0 * empty_responses / max(1, num_samples)

    filename = os.path.basename(path)
    dataset_name = filename.split("_")[0]  # e.g. vicuna_gpt2_eval.json -> "vicuna"

    return {
        "model": model_label,
        "dataset": dataset_name,
        "num_samples": num_samples,
        "avg_prompt_words": round(avg_prompt_words, 2),
        "prompt_lex_div": round(prompt_lex_div, 4),
        "avg_response_words": round(avg_resp_words, 2),
        "response_lex_div": round(resp_lex_div, 4),
        "empty_response_%": round(empty_pct, 2),
    }


def build_table(eval_dir, suffix, model_label, csv_name):
    if not os.path.isdir(eval_dir):
        print(f"[WARN] Directory not found: {eval_dir}")
        return

    pattern = os.path.join(eval_dir, f"*{suffix}")
    files = sorted(glob(pattern))

    if not files:
        print(f"[WARN] No files matching {pattern}")
        return

    rows = []

    print(f"\n=== Processing {model_label} in {eval_dir} ===")
    for path in files:
        print(f"  -> {os.path.basename(path)}")
        row = analyze_file(path, model_label)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("dataset")

    print(f"\n=== {model_label} Evaluation Summary (Markdown) ===\n")
    try:
        print(df.to_markdown(index=False))
    except Exception:
        print(df)

    df.to_csv(csv_name, index=False, encoding="utf-8")
    print(f"\n📁 Saved CSV: {csv_name}\n")


def main():
    # GPT-2 table
    build_table(
        eval_dir=GPT2_DIR,
        suffix=GPT2_SUFFIX,
        model_label="GPT-2 Superfiltered 10%",
        csv_name="gpt2_eval_summary.csv",
    )

    # LLaMA-2 table
    build_table(
        eval_dir=LLAMA_DIR,
        suffix=LLAMA_SUFFIX,
        model_label="LLaMA2 Superfiltered 10%",
        csv_name="llama_eval_summary.csv",
    )


if __name__ == "__main__":
    main()
