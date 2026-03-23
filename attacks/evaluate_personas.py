"""
Stage 2: Evaluate pre-generated suffixes across persona-conditioned states.

Usage:
    # Default (empty system prompt):
    python evaluate_personas.py \
        --target_model llama2-7b \
        --dataset advbench \
        --sys_mode replace \
        --results_dir results/laa \
        --temp 0.0

    # With safety system prompt:
    python evaluate_personas.py \
        --target_model llama2-7b \
        --dataset advbench \
        --sys_mode replace \
        --safe_prompt "$(python -c 'from config import SAFE_PROMPT; print(SAFE_PROMPT)')" \
        --results_dir results/laa
"""

import os
import argparse
import datetime
import traceback

import pandas as pd
import numpy as np
import torch
import random

from personality_prompts import get_persona_prompts
from model_loader import load_target_model
from prompts import get_universal_manual_prompt_sys
from utils import insert_adv_string
from judges import judge_rule_based

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


def build_base_prompt(goal: str, target_str: str = "", prompt_template: str = "best_llama2"):
    return get_universal_manual_prompt_sys(prompt_template, target_str, goal.lower())


def run_persona_effect_eval(
    target_model: str,
    sys_mode: str,
    suffix_csv: str,
    results_dir: str,
    dataset: str,
    source_persona: str = "pure",
    prompt_template: str = "best_llama2",
    target_max_n_tokens: int = 150,
    temperature: float = 0.0,
    safe_prompt: str = "",
):
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print(f"\n[CONFIG]")
    print(f"  target_model : {target_model}")
    print(f"  sys_mode     : {sys_mode}")
    print(f"  suffix_csv   : {suffix_csv}")
    print(f"  temperature  : {temperature}")
    print(f"  results_dir  : {results_dir}")

    # Load suffixes
    csv_path = os.path.join("pure_suffix", dataset, suffix_csv)
    try:
        df_src = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        print(f"[WARN] Retrying with latin1: {csv_path}")
        df_src = pd.read_csv(csv_path, encoding="latin1")
    print(f"[LOAD] {len(df_src)} rows loaded.")

    # Load model
    class _Args:
        pass
    _args = _Args()
    _args.target_model = target_model
    _args.persona_prompt = ""
    _args.sys_mode = sys_mode
    _args.safe_prompt = safe_prompt

    print("\n[MODEL] Loading target model...")
    targetLM = load_target_model(_args)
    print("[MODEL] Loaded.")

    # Evaluate all personas
    os.makedirs(results_dir, exist_ok=True)
    p2_descriptions = get_persona_prompts(mode="full")
    rows = []

    for persona_name, persona_prompt in p2_descriptions.items():
        print(f"\n===== Persona: {persona_name} (sys_mode={sys_mode}, temp={temperature}) =====")

        for _, row in tqdm(df_src.iterrows(), total=len(df_src), desc=persona_name, leave=False):
            try:
                idx = int(row["index"])
                goal = str(row["goal"])
                suffix = str(row["suffix"])

                base_prompt = build_base_prompt(goal=goal, target_str="", prompt_template=prompt_template)
                msg_with_suffix = insert_adv_string(base_prompt, suffix)

                outputs = targetLM.get_response_final(
                    persona_prompt=persona_prompt,
                    prompts_list=[msg_with_suffix],
                    max_n_tokens=target_max_n_tokens,
                    temperature=temperature,
                )
                text = outputs[0]["text"]
                outcome = "succ" if judge_rule_based(text) else "fail"

                rows.append({
                    "index": idx,
                    "goal": goal,
                    "suffix": suffix,
                    "source_persona": source_persona,
                    "persona_eval": persona_name,
                    "temperature": temperature,
                    "response": text,
                    "jailbreak_rule": outcome,
                })
            except Exception as e:
                print(f"[ERROR] index {row.get('index', 'N/A')} / {persona_name}: {e}")
                traceback.print_exc()
                continue

    # Save results
    result_df = pd.DataFrame(rows)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(results_dir, f"temp{temperature}_{target_model}_{sys_mode}_{timestamp}.csv")
    result_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ Saved: {out_csv}")

    # ASR summary
    summary = (
        result_df.groupby("persona_eval")["jailbreak_rule"]
        .apply(lambda s: (s == "succ").mean() * 100.0)
        .reset_index()
        .rename(columns={"jailbreak_rule": "ASR_rule_based"})
    )
    summary_out = os.path.splitext(out_csv)[0] + "_asr.csv"
    summary.to_csv(summary_out, index=False, encoding="utf-8-sig")
    print(f"✅ ASR summary saved: {summary_out}")
    print("\n[ASR per persona (%)]\n" + summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sys_mode", type=str, required=True,
                        choices=["append", "prepend", "replace"])
    parser.add_argument("--results_dir", type=str, default="results/laa")
    parser.add_argument("--source_persona", type=str, default="pure")
    parser.add_argument("--prompt_template", type=str, default="best_llama2")
    parser.add_argument("--target_max_n_tokens", type=int, default=150)
    parser.add_argument("--safe_prompt", type=str, default="",
                        help="Safety system prompt. Empty by default (vanilla state).")
    parser.add_argument("--temp", type=float, default=0.0,
                        help="Decoding temperature (e.g. 0.0, 0.7, 1.0)")
    args = parser.parse_args()

    suffix_csv = (
        f"{args.target_model}_append_prepend.csv"
        if args.sys_mode in ["append", "prepend"]
        else f"{args.target_model}_{args.sys_mode}.csv"
    )

    run_persona_effect_eval(
        target_model=args.target_model,
        sys_mode=args.sys_mode,
        suffix_csv=suffix_csv,
        results_dir=args.results_dir,
        dataset=args.dataset,
        source_persona=args.source_persona,
        prompt_template=args.prompt_template,
        target_max_n_tokens=args.target_max_n_tokens,
        temperature=args.temp,
        safe_prompt=args.safe_prompt,
    )
