import os
import json
from statistics import mean
import csv

# ==== CONFIG ====
MODELS = {
    "gpt2_superfilter_10per": "gpt2-finetuned",
    "gpt2_kmeans_hybrid_10per": "gpt2-kmeans-hybrid",
}

DATASETS = ["vicuna", "koala", "wizardlm", "sinstruct", "lima"]
FILE_PATTERN = "{dataset}_gpt2_eval.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def wc(text):
    if not isinstance(text, str):
        return 0
    return len(text.strip().split())


def extract_metrics(model_name, model_dir, dataset):
    """
    Load JSON from:
    model_dir/test_inference/{dataset}_gpt2_eval.json
    and compute summary metrics.
    """
    path = os.path.join(model_dir, "test_inference", FILE_PATTERN.format(dataset=dataset))
    if not os.path.exists(path):
        print(f"[WARN] File missing: {path}")
        return None

    data = load_json(path)

    prompts = []
    responses = []

    for ex in data:
        p = ex.get("prompt", "")
        r = ex.get("response", "")

        prompts.append(p if isinstance(p, str) else "")
        responses.append(r if isinstance(r, str) else "")

    N = len(data)
    avg_prompt = round(mean(wc(p) for p in prompts), 2)
    avg_response = round(mean(wc(r) for r in responses), 2)
    empty = sum(1 for r in responses if len(r.strip()) == 0)
    empty_pct = round((empty / N) * 100, 2)

    return {
        "dataset": dataset,
        "model": model_name,
        "num_samples": N,
        "avg_prompt_words": avg_prompt,
        "avg_response_words": avg_response,
        "empty_responses": empty,
        "empty_response_pct": empty_pct,
    }


def main():
    OUTPUT = "gpt2_comparison.csv"

    rows = []

    for model_name, folder in MODELS.items():
        for ds in DATASETS:
            m = extract_metrics(model_name, folder, ds)
            if m:
                rows.append(m)

    # Sort by dataset -> model
    rows.sort(key=lambda x: (x["dataset"], x["model"]))

    # Write CSV
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset",
            "Model",
            "Num Samples",
            "Avg Prompt Words",
            "Avg Response Words",
            "Empty Responses",
            "Empty Response (%)"
        ])
        for r in rows:
            writer.writerow([
                r["dataset"],
                r["model"],
                r["num_samples"],
                r["avg_prompt_words"],
                r["avg_response_words"],
                r["empty_responses"],
                r["empty_response_pct"]
            ])

    print(f"\n📁 CSV saved as: {OUTPUT}")
    print("Done.")


if __name__ == "__main__":
    main()
