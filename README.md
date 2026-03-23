# state-driven-jailbreak

Code for evaluating jailbreak robustness as a function of the model's operational state. We propose a state-conditioned framework that analyzes how variation in model state, induced by ordinary system prompts, affects jailbreak susceptibility under fixed attacks.

## Overview

Most jailbreak evaluations assume a *vanilla* model state — a default configuration with no prior context. In practice, LLMs are deployed with system prompts that shape their operational state. This work shows that the same jailbreak artifact can have substantially different effects depending on the model's state.

We propose a **state-conditioned evaluation framework** that:
1. Generates jailbreak artifacts under the vanilla state (Stage 1)
2. Applies those artifacts to persona-conditioned states and measures the resulting ASR shift (Stage 2)

Using **Big Five personality traits** (OCEAN) as controlled state approximations, we demonstrate that ordinary system prompts alone can shift Attack Success Rate (ASR) by up to 56 percentage points — without any modification to the attack itself.

## Repository Structure

```
state-driven-jailbreak/
├── attacks/
│   ├── generate_suffixes.py            # Stage 1: suffix generation under vanilla state
│   ├── evaluate_personas.py            # Stage 2: persona-conditioned evaluation
│   ├── laa_attack.py                   # LAA core logic
│   ├── model_loader.py                 # Model loading and prompt formatting
│   ├── language_models.py              # GPT / HuggingFace wrappers
│   ├── prompts.py                      # Attack prompt templates
│   ├── personality_prompts.py          # Big Five persona prompts
│   ├── config.py                       # Model paths, decoding settings, safe prompt
|   ├── judges.py                       # Rule-based judge
│   └── utils.py                        # Utility functions
├── refusal_analysis/
│   ├── extract_hidden_states.py        # Step 1: extract hidden states (Llama-2-13B)
│   ├── train_probe.py                  # Step 2: train probe + compute projections
│   ├── analyze_paraphrase_projection.py # Step 3: paraphrase projection analysis
│   └── personality_prompts_semantic.py # Paraphrase prompts (10 per trait)
└── .gitignore
```

## Setup

```bash
pip install torch transformers fastchat openai tiktoken anthropic pandas tqdm bert-score
```

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export HF_TOKEN="your_huggingface_token"   # required for gated models (Llama-2, etc.)
```

## System Prompt Configuration

This codebase follows a **vanilla-by-default** design: the model's system prompt is **empty** unless explicitly specified.

| Flag | System prompt |
|---|---|
| *(none)* | `""` — vanilla state |
| `--safe_prompt "..."` | The provided safety prompt |

When a persona is also applied (Stage 2), the final system message depends on `--sys_mode`:

| `sys_mode` | Result |
|---|---|
| `replace` | persona only |
| `append` | `safe_prompt` + `"\n\n"` + persona |
| `prepend` | persona + `"\n\n"` + `safe_prompt` |

The safety prompt used in our experiments (Appendix B.3) is the standard Llama-2 safety prompt:

```
You are a helpful, respectful and honest assistant. Always answer as helpfully
as possible, while being safe. ...
```

## Usage

### Stage 1: Generate Suffixes (vanilla state)

Optimize adversarial suffixes under the vanilla state using LAA.

```bash
# Vanilla (no system prompt)
python attacks/generate_suffixes.py \
    --target_model llama2-7b \
    --csv_path advbench_50.csv \
    --sys_mode replace \
    --results_dir results/laa

# With safety prompt
python attacks/generate_suffixes.py \
    --target_model llama2-7b \
    --csv_path advbench_50.csv \
    --sys_mode replace \
    --safe_prompt "You are a helpful, respectful and honest assistant..." \
    --results_dir results/laa
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--target_model` | — | Target model key (see Supported Models below) |
| `--csv_path` | `advbench_50.csv` | Dataset CSV with a `goal` column |
| `--sys_mode` | `replace` | System prompt combination mode: `replace`, `append`, `prepend` |
| `--safe_prompt` | `""` | Safety system prompt. Empty = vanilla state |
| `--results_dir` | `results/laa` | Output directory |
| `--n-restarts` | `1` | Number of random restarts |

### Stage 2: Evaluate Across Persona-Conditioned States

Apply pre-generated suffixes to all Big Five persona-conditioned states.

```bash
# Vanilla baseline
python attacks/evaluate_personas.py \
    --target_model llama2-7b \
    --dataset advbench \
    --sys_mode replace \
    --results_dir results/laa \
    --temp 0.0

# Persona + safety prompt (Appendix B.3 setting)
python attacks/evaluate_personas.py \
    --target_model llama2-7b \
    --dataset advbench \
    --sys_mode append \
    --safe_prompt "You are a helpful, respectful and honest assistant..." \
    --results_dir results/laa \
    --temp 0.0
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--target_model` | — | Target model key |
| `--dataset` | — | Dataset subdirectory name under `pure_suffix/` |
| `--sys_mode` | `replace` | System prompt combination mode |
| `--safe_prompt` | `""` | Safety system prompt. Empty = vanilla state |
| `--results_dir` | `results/laa` | Output directory |
| `--temp` | `0.0` | Decoding temperature |

### GPT-4 Judge (optional)

```bash
python evaluation/run_gpt_judge.py \
    --input_csv results/laa/temp0.0_llama2-7b_replace_<timestamp>.csv \
    --judge_model gpt-4-0613
```

## Refusal Analysis (Section 6)

The `refusal_analysis/` folder contains code for the representation-level analysis in Section 6 and Appendix D.

```
refusal_analysis/
├── extract_hidden_states.py        # Step 1: extract hidden states from Llama-2-13B
├── train_probe.py                  # Step 2: train logistic regression probe + compute projections
└── analyze_paraphrase_projection.py # Step 3: paraphrase-level projection analysis (Appendix D.3)
```

### Step 1: Extract Hidden States

```bash
python refusal_analysis/extract_hidden_states.py \
    --original_csv  results/laa/llama2-13b_replace_asr.csv \
    --semantic_dir  results_semantic/Llama2-13b \
    --dataset       advbench \
    --sys_mode      replace \
    --output_dir    probe_data
```

Outputs `probe_data/hidden_vectors.npz` with hidden states at layers 4 (early), 16 (middle), 32 (late).

### Step 2: Train Probe

```bash
# Original prompts only (Figure 6)
python refusal_analysis/train_probe.py \
    --npz_path    probe_data/hidden_vectors.npz \
    --output_dir  probe_results \
    --model       13b \
    --original_only
```

Outputs `probe_results/probe_analysis.pdf`, `probe_correlation.csv`, and `probe_directions.npz`.

### Step 3: Analyze Paraphrase Projections

```bash
python refusal_analysis/analyze_paraphrase_projection.py \
    --npz_path        probe_data/hidden_vectors.npz \
    --directions_path probe_results/probe_directions.npz \
    --output_dir      probe_results
```

Outputs `paraphrase_projection.pdf` and `paraphrase_projection.csv` (Figure 9, Appendix D.3).

**Note:** `extract_hidden_states.py` requires `personality_prompts_semantic.py`, which contains the 10 paraphrases per Big Five trait used in Section 5.1.

## Supported Models

| Model | Key |
|---|---|
| Llama-2-7B-chat | `llama2-7b` |
| Llama-2-13B-chat | `llama2-13b` |
| Llama-3-8B-Instruct | `llama3-8b` |
| Llama-3.1-8B-Instruct | `llama3.1-8b` |
| Qwen2.5-7B-Instruct | `qwen2.5-7b` |
| Mistral-7B-Instruct-v0.2 | `mistral-7b` |
| Vicuna-7B-v1.5 | `vicuna` |

Model paths are configured in `attacks/config.py`.

## Persona Prompts

The Big Five persona prompts are adapted from:

> Jiang et al. (2023). *Evaluating and Inducing Personality in Pre-Trained Language Models.* NeurIPS 2023.
> [[paper]](https://arxiv.org/abs/2206.07550) [[code]](https://github.com/jianggy/MPI)

All five prompts (O/C/E/A/N) are in `attacks/personality_prompts.py`.

## Acknowledgements

This codebase builds on the following open-source repositories:

| Method | Repository |
|---|---|
| LAA (Andriushchenko et al., 2025) | https://github.com/tml-epfl/llm-adaptive-attacks |
| PAIR (Chao et al., 2025) | https://github.com/patrickrchao/JailbreakingLLMs |
| AutoDAN (Liu et al., 2024) | https://github.com/SheltonLiu-N/AutoDAN |
