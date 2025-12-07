import os
import json
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# 🔹 Load variables from .env (must contain OPENAI_API_KEY=...)
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found. Please set it in .env or environment variables.")

# 🔹 Initialize OpenAI client with API key from .env
client = OpenAI(api_key=api_key)


JUDGE_INSTRUCTIONS = """
You are a strict, fair evaluator for two model responses.

Given:
- An instruction (or question)
- Response A
- Response B

Judge which response is better overall, considering:
1. Following the instruction
2. Helpfulness and level of detail
3. Coherence and fluency
4. Factual correctness (when applicable)
5. Safety (avoid harmful or toxic content)

Return a JSON object ONLY, with this exactly:

{
  "winner": "A" or "B" or "tie",
  "explanation": "short reason"
}
"""

def extract_instruction(rec):
    # Try common patterns from your JSON files
    if "prompt" in rec:
        return rec["prompt"]
    if "instruction" in rec:
        instr = rec["instruction"]
        inp = rec.get("input", "")
        if inp:
            return f"Instruction: {instr}\nInput: {inp}"
        return instr
    if "question" in rec:
        return rec["question"]
    # Fallback
    return rec.get("instruction", "")

def extract_response(rec):
    # Try keys in order of likelihood
    for key in ["response", "output", "model_output", "answer", "generated"]:
        val = rec.get(key, None)
        if isinstance(val, str) and val.strip():
            return val
    # Last resort
    return ""

def call_judge_model(instruction, resp_a, resp_b, model_name="gpt-4o-mini"):
    content = f"""
Instruction:
{instruction}

Response A:
{resp_a}

Response B:
{resp_b}

{JUDGE_INSTRUCTIONS}
"""
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
    )
    txt = completion.choices[0].message.content

    # Be a bit defensive in case judge adds extra text
    try:
        start = txt.find("{")
        end = txt.rfind("}") + 1
        json_str = txt[start:end]
        return json.loads(json_str)
    except Exception:
        # Fallback: treat as tie
        return {"winner": "tie", "explanation": "Parsing error, counted as tie."}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fileA", type=str, required=True,
                        help="Path to JSON results for model A")
    parser.add_argument("--fileB", type=str, required=True,
                        help="Path to JSON results for model B")
    parser.add_argument("--labelA", type=str, required=True,
                        help="Name/label for model A (e.g., gpt2-superfilter)")
    parser.add_argument("--labelB", type=str, required=True,
                        help="Name/label for model B (e.g., gpt2-hybrid)")
    parser.add_argument("--out", type=str, default=None,
                        help="Path to save summary JSON (default: auto name)")
    parser.add_argument("--max_samples", type=int, default=80,
                        help="Max number of samples to evaluate (to control cost)")
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini",
                        help="OpenAI judge model name")
    args = parser.parse_args()

    with open(args.fileA, "r", encoding="utf-8") as f:
        dataA = json.load(f)
    with open(args.fileB, "r", encoding="utf-8") as f:
        dataB = json.load(f)

    n = min(len(dataA), len(dataB), args.max_samples)
    dataA = dataA[:n]
    dataB = dataB[:n]

    print(f"Evaluating {n} paired samples")
    print(f"Model A: {args.labelA}")
    print(f"Model B: {args.labelB}")
    print(f"Judge model: {args.judge_model}")

    wins = losses = ties = 0
    per_sample = []

    for i, (ra, rb) in enumerate(tqdm(zip(dataA, dataB), total=n)):
        instr = extract_instruction(ra)
        resp_a = extract_response(ra)
        resp_b = extract_response(rb)

        result = call_judge_model(instr, resp_a, resp_b, args.judge_model)
        winner = result.get("winner", "tie")

        if winner == "A":
            wins += 1
        elif winner == "B":
            losses += 1
        else:
            ties += 1

        per_sample.append({
            "index": i,
            "instruction": instr,
            "response_A": resp_a,
            "response_B": resp_b,
            "winner": winner,
            "explanation": result.get("explanation", "")
        })

    total = max(1, wins + losses + ties)
    score = (wins - losses) / total + 1.0

    summary = {
        "model_A": args.labelA,
        "model_B": args.labelB,
        "judge_model": args.judge_model,
        "num_samples": n,
        "wins_for_B": wins,      # B better than A
        "losses_for_B": losses,  # A better than B
        "ties": ties,
        "pairwise_score_for_B": score,
    }

    out_path = args.out
    if out_path is None:
        baseA = os.path.basename(args.fileA).replace(".json", "")
        baseB = os.path.basename(args.fileB).replace(".json", "")
        out_path = f"pairwise_{args.labelB}_vs_{args.labelA}_{baseA}_{baseB}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": per_sample}, f, indent=2)

    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"Saved details to: {out_path}")


if __name__ == "__main__":
    main()
