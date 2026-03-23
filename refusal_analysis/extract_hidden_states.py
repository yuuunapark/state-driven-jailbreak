"""
Step 1: Extract hidden states from Llama-2-13B under persona-conditioned prompts.

Covers:
  - Original prompts: vanilla + 5 Big Five personas (300 samples)
  - Semantic paraphrases: 5 traits x 10 paraphrases x 50 queries (2500 samples)

Usage:
    python extract_hidden_states.py \
        --original_csv  results/laa/llama2-13b_replace_asr.csv \
        --semantic_dir  results_semantic/Llama2-13b \
        --dataset       advbench \
        --sys_mode      replace \
        --output_dir    probe_data
"""

import os
import argparse
import glob
import traceback

import numpy as np
import pandas as pd
import torch
import random

from personality_prompts import get_persona_prompts
import personality_prompts_semantic as _sem
from model_loader import load_target_model
from prompts import get_universal_manual_prompt_sys
from utils import insert_adv_string

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

TARGET_LAYERS = {"early": 4, "middle": 16, "late": 32}
TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

semantic_prompts = {
    "Openness":          _sem.Openness,
    "Conscientiousness": _sem.Conscientiousness,
    "Extraversion":      _sem.Extraversion,
    "Agreeableness":     _sem.Agreeableness,
    "Neuroticism":       _sem.Neuroticism,
}


def _build_full_prompts(targetLM, prompts_list, persona_prompt):
    import common
    batchsize = len(prompts_list)
    tokenizer = targetLM.model.tokenizer
    convs_list = [common.conv_template(targetLM.template) for _ in range(batchsize)]
    has_persona = bool(persona_prompt and persona_prompt.strip())
    model_name = targetLM.model_name
    sys_mode = targetLM.sys_mode
    full_prompts = []

    for conv, prompt in zip(convs_list, prompts_list):
        if "llama2" in model_name:
            from model_loader import get_system_message
            cur = get_system_message(conv)
            S, E = "<s>[INST] <<SYS>>\n", "\n<</SYS>>\n\n"
            default = cur.replace(S, "").replace(E, "").strip()
            if not has_persona:
                new_sys = f"{S}{default}{E}" if sys_mode in ("append", "prepend") else f"{S}{persona_prompt}{E}"
            elif sys_mode == "append":
                new_sys = f"{S}{default}\n\n{persona_prompt}{E}"
            elif sys_mode == "prepend":
                new_sys = f"{S}{persona_prompt}\n\n{default}{E}"
            else:
                new_sys = f"{S}{persona_prompt}{E}"
            conv.system = new_sys
            prompt = prompt + " "

        conv.append_message(conv.roles[0], prompt)

        if "llama2" in model_name:
            conv.append_message(conv.roles[1], None)
            full_prompts.append("<s>" + conv.get_prompt())
        else:
            raise ValueError(f"Model {model_name} not supported for hidden state extraction.")

    return full_prompts


@torch.no_grad()
def extract_hidden_states(targetLM, full_prompt, target_layers):
    tokenizer = targetLM.model.tokenizer
    model = targetLM.model.model
    inputs = tokenizer(full_prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs, output_hidden_states=True)
    hs = outputs.hidden_states
    last_pos = inputs["input_ids"].shape[1] - 1
    return {
        name: hs[min(idx, len(hs) - 1)][0, last_pos, :].float().cpu().numpy()
        for name, idx in target_layers.items()
    }


def load_original_labels(original_csv):
    try:
        df = pd.read_csv(original_csv, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(original_csv, encoding="latin1")
    df["persona_norm"] = df["persona_eval"].replace({"pure": "vanilla"})
    print(f"[original] {len(df)} rows loaded.")
    print(df.groupby("persona_norm")["jailbreak_rule"]
          .apply(lambda s: (s == "succ").sum()).rename("n_succ"))
    return df


def load_semantic_labels(semantic_dir):
    records = []
    for trait in TRAITS:
        trait_dir = os.path.join(semantic_dir, trait)
        csv_files = [f for f in glob.glob(os.path.join(trait_dir, "temp*.csv"))
                     if "_asr" not in f]
        if not csv_files:
            print(f"[WARN] No CSV found in {trait_dir}")
            continue

        csv_path = sorted(csv_files)[-1]
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="latin1")

        for _, row in df.iterrows():
            records.append({
                "trait":      trait,
                "prompt_num": row["persona_eval"],
                "query_idx":  int(row["index"]),
                "label":      1 if row["jailbreak_rule"] == "succ" else 0,
                "goal":       row["goal"],
                "suffix":     row["suffix"],
            })

        n_succ = (df["jailbreak_rule"] == "succ").sum()
        print(f"[semantic] {trait}: {len(df)} rows, succ={n_succ}")

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_csv",    type=str, required=True,
                        help="Path to ASR results CSV for original prompts.")
    parser.add_argument("--semantic_dir",    type=str, required=True,
                        help="Directory containing paraphrase results (e.g. results_semantic/Llama2-13b).")
    parser.add_argument("--dataset",         type=str, required=True)
    parser.add_argument("--sys_mode",        type=str, default="replace")
    parser.add_argument("--output_dir",      type=str, default="probe_data")
    parser.add_argument("--prompt_template", type=str, default="best_llama2")
    parser.add_argument("--skip_original",   action="store_true",
                        help="Skip hidden state extraction for original prompts.")
    parser.add_argument("--skip_semantic",   action="store_true",
                        help="Skip hidden state extraction for paraphrase prompts.")
    args = parser.parse_args()

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)

    os.makedirs(args.output_dir, exist_ok=True)

    suffix_csv = f"pure_suffix/{args.dataset}/llama2-13b_replace.csv"
    try:
        df_src = pd.read_csv(suffix_csv, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df_src = pd.read_csv(suffix_csv, encoding="latin1")
    queries = df_src["goal"].tolist()
    suffixes = df_src["suffix"].tolist()
    print(f"[suffix] {len(queries)} queries loaded.")

    df_original = load_original_labels(args.original_csv)
    df_semantic = load_semantic_labels(args.semantic_dir)

    class _Args:
        target_model = "llama2-13b"
        persona_prompt = ""
        sys_mode = args.sys_mode
        safe_prompt = ""

    print("\n[model] Loading llama2-13b...")
    targetLM = load_target_model(_Args())
    print("[model] Loaded.")

    X = {layer: [] for layer in TARGET_LAYERS}
    y_list, persona_list, prompt_num_list, idx_list = [], [], [], []
    original_prompts = get_persona_prompts(mode="full")

    if not args.skip_original:
        print("\n" + "=" * 50)
        print("PART 1: Original prompts")
        print("=" * 50)

        for persona_name, persona_prompt in original_prompts.items():
            persona_norm = "vanilla" if persona_name == "vanilla" else persona_name
            print(f"\n-- {persona_name}")
            rows = df_original[df_original["persona_norm"] == persona_norm]

            for q_idx in tqdm(range(len(queries)), desc=persona_name, leave=False):
                try:
                    goal = queries[q_idx]
                    suffix = suffixes[q_idx]
                    row = rows[rows["index"] == q_idx]
                    if len(row) == 0:
                        continue
                    label = 1 if row.iloc[0]["jailbreak_rule"] == "succ" else 0

                    base_prompt = get_universal_manual_prompt_sys(
                        args.prompt_template, "", goal.lower()
                    )
                    prompt_with_sfx = insert_adv_string(base_prompt, suffix)
                    full_prompt = _build_full_prompts(targetLM, [prompt_with_sfx], persona_prompt)[0]
                    h = extract_hidden_states(targetLM, full_prompt, TARGET_LAYERS)

                    for layer in TARGET_LAYERS:
                        X[layer].append(h[layer])
                    y_list.append(label)
                    persona_list.append(persona_norm)
                    prompt_num_list.append("original")
                    idx_list.append(q_idx)

                except Exception as e:
                    print(f"[ERROR] {persona_name} q={q_idx}: {e}")
                    traceback.print_exc()

        print(f"\n[PART 1] Done: {len(y_list)} samples")

    if not args.skip_semantic:
        print("\n" + "=" * 50)
        print("PART 2: Semantic paraphrases")
        print("=" * 50)

        for trait in TRAITS:
            trait_prompts = semantic_prompts[trait]

            for prompt_num, persona_prompt in trait_prompts.items():
                print(f"\n-- {trait} / {prompt_num}")
                rows = df_semantic[
                    (df_semantic["trait"] == trait) &
                    (df_semantic["prompt_num"] == prompt_num)
                ]

                for q_idx in tqdm(range(len(queries)), desc=f"{trait[:3]}/{prompt_num}", leave=False):
                    try:
                        goal = queries[q_idx]
                        suffix = suffixes[q_idx]
                        row = rows[rows["query_idx"] == q_idx]
                        if len(row) == 0:
                            continue
                        label = int(row.iloc[0]["label"])

                        base_prompt = get_universal_manual_prompt_sys(
                            args.prompt_template, "", goal.lower()
                        )
                        prompt_with_sfx = insert_adv_string(base_prompt, suffix)
                        full_prompt = _build_full_prompts(targetLM, [prompt_with_sfx], persona_prompt)[0]
                        h = extract_hidden_states(targetLM, full_prompt, TARGET_LAYERS)

                        for layer in TARGET_LAYERS:
                            X[layer].append(h[layer])
                        y_list.append(label)
                        persona_list.append(trait)
                        prompt_num_list.append(prompt_num)
                        idx_list.append(q_idx)

                    except Exception as e:
                        print(f"[ERROR] {trait}/{prompt_num} q={q_idx}: {e}")
                        traceback.print_exc()

        print(f"\n[PART 2] Done: {len(y_list)} samples")

    out_path = os.path.join(args.output_dir, "hidden_vectors.npz")
    np.savez(
        out_path,
        X_early     = np.array(X["early"]),
        X_middle    = np.array(X["middle"]),
        X_late      = np.array(X["late"]),
        y           = np.array(y_list),
        personas    = np.array(persona_list),
        prompt_nums = np.array(prompt_num_list),
        query_idx   = np.array(idx_list),
    )
    print(f"\nSaved: {out_path}  ({len(y_list)} samples)")


if __name__ == "__main__":
    main()
