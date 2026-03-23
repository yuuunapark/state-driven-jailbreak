"""
Step 2: Train a logistic regression probe on hidden states and compute
projections onto the refusal direction.

Usage:
    python train_probe.py \
        --npz_path    probe_data/hidden_vectors.npz \
        --output_dir  probe_results \
        --model       13b

    # Original prompts only (no paraphrases):
    python train_probe.py \
        --npz_path    probe_data/hidden_vectors.npz \
        --output_dir  probe_results \
        --model       13b \
        --original_only
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr


KNOWN_ASR = {
    "7b": {
        "vanilla":            2,
        "Openness":          14,
        "Conscientiousness": 58,
        "Extraversion":      16,
        "Agreeableness":     10,
        "Neuroticism":       44,
    },
    "13b": {
        "vanilla":            0,
        "Openness":          46,
        "Conscientiousness": 14,
        "Extraversion":      38,
        "Agreeableness":      8,
        "Neuroticism":       52,
    },
}

PERSONA_ORDER = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
LAYER_NAMES   = ["early", "middle", "late"]
LAYER_KEYS    = {"early": "X_early", "middle": "X_middle", "late": "X_late"}


def train_probe(X, y, layer_name, cv=5):
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    probe = LogisticRegression(C=1.0, max_iter=1000, random_state=1)
    cv_scores = cross_val_score(
        probe, X_sc, y,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=1),
        scoring="roc_auc",
    )
    probe.fit(X_sc, y)
    print(f"  [{layer_name}] AUROC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}"
          f"  (n={len(y)}, compliance={y.sum()}, refusal={len(y) - y.sum()})")
    return probe, scaler, cv_scores


def compute_projections(X_by_persona, probe, scaler):
    direction = probe.coef_[0]
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    proj_mean, proj_std = {}, {}
    for persona, X in X_by_persona.items():
        projs = scaler.transform(X) @ direction
        proj_mean[persona] = projs.mean()
        proj_std[persona] = projs.std()
    return proj_mean, proj_std, direction


def compute_correlation_table(probe_results, known_asr):
    print("\n-- Projection vs ASR Correlation --")
    rows = []
    for layer in LAYER_NAMES:
        proj_mean = probe_results[layer]["proj_mean"]
        xs = [proj_mean.get(p, 0) for p in PERSONA_ORDER]
        ys = [known_asr.get(p, 0) for p in PERSONA_ORDER]
        r_p, p_p = pearsonr(xs, ys)
        r_s, p_s = spearmanr(xs, ys)
        sig = "**" if p_p < 0.05 else ("*" if p_p < 0.1 else "")
        print(f"  [{layer:6s}]  Pearson r={r_p:+.3f} (p={p_p:.3f}){sig} | "
              f"Spearman r={r_s:+.3f} (p={p_s:.3f})")
        rows.append({
            "layer":       layer,
            "pearson_r":   round(r_p, 3),
            "pearson_p":   round(p_p, 3),
            "spearman_r":  round(r_s, 3),
            "spearman_p":  round(p_s, 3),
        })
    return pd.DataFrame(rows)


def plot_results(probe_results, known_asr, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    layer_colors = {"early": "#4C72B0", "middle": "#DD8452", "late": "#55A868"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for i, layer in enumerate(LAYER_NAMES):
        mean = probe_results[layer]["cv_mean"]
        std  = probe_results[layer]["cv_std"]
        ax.bar(i, mean, 0.5, color=layer_colors[layer], alpha=0.85, yerr=std, capsize=5)
        ax.text(i, mean + std + 0.01, f"{mean:.3f}", ha="center", fontsize=10)
    ax.axhline(0.5, color="gray", linestyle="--", lw=1, label="Random (0.5)")
    ax.set_xticks(range(len(LAYER_NAMES)))
    ax.set_xticklabels(LAYER_NAMES, fontsize=12)
    ax.set_ylabel("AUROC (5-fold CV)", fontsize=12)
    ax.set_ylim(0.4, 1.05)
    ax.set_title("(a) Probe Classifier Performance by Layer", fontsize=13)
    ax.legend(fontsize=9)

    ax = axes[1]
    for layer in LAYER_NAMES:
        proj_mean = probe_results[layer]["proj_mean"]
        proj_std  = probe_results[layer]["proj_std"]
        xs = [proj_mean.get(p, 0) for p in PERSONA_ORDER]
        ys = [known_asr.get(p, 0) for p in PERSONA_ORDER]
        es = [proj_std.get(p, 0) for p in PERSONA_ORDER]
        ax.errorbar(xs, ys, xerr=es, fmt="o", color=layer_colors[layer],
                    ms=8, label=layer, capsize=3, zorder=5,
                    elinewidth=1, markeredgecolor="white")
        for p, xi, yi in zip(PERSONA_ORDER, xs, ys):
            ax.annotate(p[:3], (xi, yi), textcoords="offset points",
                        xytext=(5, 3), fontsize=9)
    ax.axhline(known_asr["vanilla"], color="gray", linestyle="--", lw=0.8,
               label=f"vanilla ASR={known_asr['vanilla']}%")
    ax.set_xlabel("Projection onto Refusal Direction", fontsize=12)
    ax.set_ylabel("ASR (%)", fontsize=12)
    ax.set_title("(b) Refusal Direction Projection vs ASR", fontsize=13)
    ax.legend(title="Layer", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "probe_analysis.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    print(f"\n[plot] Saved: {path}")
    plt.close()


def save_directions(probe_results, output_dir):
    out = {}
    for layer in LAYER_NAMES:
        r = probe_results[layer]
        out[f"direction_{layer}"]    = r["direction"]
        out[f"scaler_mean_{layer}"]  = r["scaler"].mean_
        out[f"scaler_scale_{layer}"] = r["scaler"].scale_
    path = os.path.join(output_dir, "probe_directions.npz")
    np.savez(path, **out)
    print(f"[save] Directions saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path",      type=str, required=True,
                        help="Path to hidden_vectors.npz from extract_hidden_states.py.")
    parser.add_argument("--output_dir",    type=str, default="probe_results")
    parser.add_argument("--model",         type=str, default="13b", choices=["7b", "13b"])
    parser.add_argument("--original_only", action="store_true",
                        help="Use only original prompt samples (no paraphrases).")
    args = parser.parse_args()

    known_asr = KNOWN_ASR[args.model]
    os.makedirs(args.output_dir, exist_ok=True)

    data      = np.load(args.npz_path, allow_pickle=True)
    personas  = data["personas"]
    y         = data["y"].astype(int)
    query_idx = data["query_idx"]

    if args.original_only and "prompt_nums" in data:
        mask      = data["prompt_nums"] == "original"
        personas  = personas[mask]
        y         = y[mask]
        query_idx = query_idx[mask]
        print(f"[filter] original_only: {mask.sum()} / {len(mask)} samples")
    else:
        mask = None

    print(f"[data] {len(y)} samples  "
          f"(compliance={y.sum()}, refusal={len(y) - y.sum()})")
    print(f"  personas: {np.unique(personas)}")

    probe_results = {}

    for layer in LAYER_NAMES:
        X_all = data[LAYER_KEYS[layer]]
        if mask is not None:
            X_all = X_all[mask]

        print(f"\n{'=' * 50}\nLayer: {layer}  |  shape: {X_all.shape}")

        probe, scaler, cv_scores = train_probe(X_all, y, layer)

        X_by_persona = {p: X_all[personas == p] for p in np.unique(personas)}
        proj_mean, proj_std, direction = compute_projections(X_by_persona, probe, scaler)

        print(f"\n  Projection onto refusal direction:")
        for p in ["vanilla"] + PERSONA_ORDER:
            m   = proj_mean.get(p, 0)
            s   = proj_std.get(p, 0)
            asr = known_asr.get(p, "?")
            print(f"    {p:20s}: {m:+.4f} +/- {s:.4f}  (ASR={asr}%)")

        probe_results[layer] = {
            "probe":    probe,
            "scaler":   scaler,
            "cv_mean":  cv_scores.mean(),
            "cv_std":   cv_scores.std(),
            "proj_mean": proj_mean,
            "proj_std":  proj_std,
            "direction": direction,
        }

    df_corr = compute_correlation_table(probe_results, known_asr)
    corr_path = os.path.join(args.output_dir, "probe_correlation.csv")
    df_corr.to_csv(corr_path, index=False, encoding="utf-8-sig")
    print(f"\n[save] {corr_path}")

    plot_results(probe_results, known_asr, args.output_dir)
    save_directions(probe_results, args.output_dir)


if __name__ == "__main__":
    main()
