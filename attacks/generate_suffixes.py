"""
Stage 1: Generate optimized suffixes under the vanilla state.

Usage:
    # Default (empty system prompt):
    python generate_suffixes.py \
        --target_model llama2-7b \
        --csv_path advbench_50.csv \
        --sys_mode replace \
        --results_dir results/laa

    # With safety system prompt:
    python generate_suffixes.py \
        --target_model llama2-7b \
        --csv_path advbench_50.csv \
        --sys_mode replace \
        --safe_prompt "$(python -c 'from config import SAFE_PROMPT; print(SAFE_PROMPT)')"
"""

import os
import json
import argparse
import datetime
import traceback

import pandas as pd

from laa_attack import main, parser as main_parser
from personality_prompts import get_persona_prompts
from judges import judge_rule_based


parser = argparse.ArgumentParser()
parser.add_argument("--target_model", type=str, required=True,
                    help="e.g. llama2-7b, llama2-13b, llama3-8b, qwen2.5-7b")
parser.add_argument("--csv_path", type=str, default="advbench_50.csv")
parser.add_argument("--results_dir", type=str, default="results/laa")
parser.add_argument("--sys_mode", type=str, default="replace",
                    choices=["append", "prepend", "replace"],
                    help="How to apply the persona prompt to the system message.")
parser.add_argument("--safe_prompt", type=str, default="",
                    help="Safety system prompt. Empty by default (vanilla state).")
parser.add_argument("--n-restarts", type=int, default=1)


def calculate_and_save_asr(csv_path):
    """Compute per-persona ASR from a completed results CSV and save a summary."""
    print(f"\n[ASR] Reading: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        return

    try:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="latin1")

        if "response_r1" not in df.columns or "persona" not in df.columns:
            print("[ERROR] CSV must contain 'response_r1' and 'persona' columns.")
            return

        df["jailbreak_check"] = df["response_r1"].apply(
            lambda x: "succ" if judge_rule_based(x) else "fail"
        )

        rows = []
        for persona, group in df.groupby("persona"):
            total = len(group)
            success = (group["jailbreak_check"] == "succ").sum()
            rows.append({
                "persona": persona,
                "ASR": round(success / total * 100, 2) if total > 0 else 0.0,
                "total": total,
                "successes": success,
            })

        asr_df = pd.DataFrame(rows)
        root, ext = os.path.splitext(csv_path)
        out_path = f"{root}_asr{ext}"
        asr_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[ASR] Saved: {out_path}")
        print(asr_df.to_string(index=False))

    except Exception as e:
        print(f"[ERROR] ASR calculation failed: {e}")
        traceback.print_exc()


def run_with_personality(csv_path, target_model_name, results_dir_base, sys_mode, n_restarts, safe_prompt=""):
    dataset = pd.read_csv(csv_path)
    # Stage 1: vanilla only (no persona conditioning during suffix generation)
    p2_descriptions = get_persona_prompts(mode="pure")

    kst = datetime.timezone(datetime.timedelta(hours=9))
    start_time_kst = datetime.datetime.now(kst).strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.splitext(os.path.basename(csv_path))[0]

    metadata = {
        "target_model": target_model_name,
        "sys_mode": sys_mode,
        "n_restarts": n_restarts,
        "csv_file": csv_path,
        "start_time_kst": start_time_kst,
        "persona_experiments": {},
    }

    final_results_dir = os.path.join(
        results_dir_base,
        f"results_{target_model_name}_{csv_filename}",
    )

    for persona_name, persona_prompt in p2_descriptions.items():
        print(f"\n========== Persona: {persona_name}  |  sys_mode: {sys_mode} ==========")

        args_template, _ = main_parser.parse_known_args([])
        args_template.judge_model = "no-judge"
        args_template.determinstic_jailbreak = False
        args_template.n_iterations = 5
        args_template.n_tokens_adv = 25
        args_template.n_tokens_change_max = 4
        args_template.prompt_template = "best_llama2"
        args_template.eval_only_rs = False
        args_template.debug = False
        args_template.persona_prompt = persona_prompt
        args_template.judge_temperature = 0.0
        args_template.judge_top_p = 1.0
        args_template.judge_max_n_tokens = 10
        args_template.judge_max_n_calls = 1
        args_template.n_chars_change_max = 0
        args_template.schedule_n_to_change = False
        args_template.schedule_prob = False
        args_template.target_max_n_tokens = 150
        args_template.seed = 1
        args_template.sys_mode = sys_mode
        args_template.safe_prompt = safe_prompt
        args_template.n_restarts = n_restarts
        args_template.dataset_name = csv_filename
        args_template.results_dir = final_results_dir
        args_template.results_filename = f"responses_{sys_mode}_R{n_restarts}.csv"

        metadata["persona_experiments"][persona_name] = {"completed": False, "end_time_kst": None}

        for i, row in dataset.iterrows():
            args = argparse.Namespace(**vars(args_template))
            args.target_model = target_model_name
            args.goal = row["goal"] if "advbench" in csv_filename else row["prompt"]
            args.goal_modified = ""
            args.target_str = ""
            args.index = i
            args.category = f"advbench_{persona_name}"
            print(f"\n--- [{i+1}/{len(dataset)}] {persona_name} | {sys_mode} ---")
            try:
                main(args)
            except Exception as e:
                print(f"[ERROR] example {i} / {persona_name}: {e}")
                traceback.print_exc()
                continue

        end_time = datetime.datetime.now(kst).strftime("%Y%m%d_%H%M%S")
        metadata["persona_experiments"][persona_name]["completed"] = True
        metadata["persona_experiments"][persona_name]["end_time_kst"] = end_time

    # ASR calculation
    target_csv = os.path.join(final_results_dir, f"LAA_{target_model_name}_{sys_mode}.csv")
    print(f"\n>>> ASR Calculation: {target_csv}")
    calculate_and_save_asr(target_csv)

    # Save metadata
    metadata["script_end_time_kst"] = datetime.datetime.now(kst).strftime("%Y%m%d_%H%M%S")
    os.makedirs(final_results_dir, exist_ok=True)
    json_path = os.path.join(
        final_results_dir,
        f"metadata_{csv_filename}_{target_model_name}_{sys_mode}.json",
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Done. Metadata saved: {json_path}")


if __name__ == "__main__":
    args = parser.parse_args()
    run_with_personality(
        args.csv_path,
        args.target_model,
        args.results_dir,
        args.sys_mode,
        safe_prompt=args.safe_prompt,
        args.n_restarts,
    )
