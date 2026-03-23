"""
Step 3: Analyze paraphrase-level projection onto the refusal direction.

Examines whether semantically equivalent paraphrases of the same persona
cluster in similar positions along the refusal direction.

Usage:
    python analyze_paraphrase_projection.py \
        --npz_path        probe_data/hidden_vectors.npz \
        --directions_path probe_results/probe_directions.npz \
        --output_dir      probe_results
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, f_oneway


KNOWN_ASR_13B = {
    "vanilla":            0,
    "Openness":          46,
    "Conscientiousness": 14,
    "Extraversion":      38,
    "Agreeableness":      8,
    "Neuroticism":       52,
}

TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
LAYER_KEYS = {"early": "X_early", "middle": "X_middle", "late": "X_late"}
TRAIT_COLORS = {
    "Openness":          "#4C72B0",
    "Conscientiousness": "#DD8452",
    "Extraversion":      "#55A868",
    "Agreeableness":     "#C44E52",
    "Neuroticism":       "#8172B2",
}


def project(X, direction, scaler_mean, scaler_scale):
    X_sc = (X - scaler_mean) / scaler_scale
    return X_sc @ direction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path",        type=str, required=True,
                        help="Path to hidden_vectors.npz from extract_hidden_states.py.")
    parser.add_argument("--directions_path", type=str, required=True,
                        help="Path to probe_directions.npz from train_probe.py.")
    parser.add_argument("--output_dir",      type=str, default="probe_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = np.load(args.npz_path, allow_pickle=True)
    dirs = np.load(args.directions_path, allow_pickle=True)

    personas    = data["personas"]
    prompt_nums = data["prompt_nums"]
    y           = data["y"].astype(int)

    print(f"[data] {len(y)} samples")
    print(f"  prompt_nums: {np.unique(prompt_nums)}")

    sem_mask = prompt_nums != "original"
    print(f"  paraphrase samples: {sem_mask.sum()}")

    layer        = "late"
    X_all        = data[LAYER_KEYS[layer]]
    direction    = dirs[f"direction_{layer}"]
    scaler_mean  = dirs[f"scaler_mean_{layer}"]
    scaler_scale = dirs[f"scaler_scale_{layer}"]

    records = []
    for trait in TRAITS:
        for pnum in [f"prompt{i}" for i in range(1, 11)]:
            mask = (personas == trait) & (prompt_nums == pnum)
            if mask.sum() == 0:
                continue
            projs = project(X_all[mask], direction, scaler_mean, scaler_scale)
            n_succ = y[mask].sum()
            asr = n_succ / mask.sum() * 100
            records.append({
                "trait":     trait,
                "prompt_num": pnum,
                "proj_mean": projs.mean(),
                "proj_std":  projs.std(),
                "asr":       asr,
                "n":         mask.sum(),
            })

    df = pd.DataFrame(records)
    csv_path = os.path.join(args.output_dir, "paraphrase_projection.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[save] {csv_path}")

    print("\n-- Per-trait paraphrase projection stats (late layer) --")
    for trait in TRAITS:
        sub = df[df["trait"] == trait]
        print(f"  {trait:20s}: mean={sub['proj_mean'].mean():+.3f}, "
              f"std={sub['proj_mean'].std():.3f}, "
              f"range=[{sub['proj_mean'].min():+.2f}, {sub['proj_mean'].max():+.2f}]  "
              f"ASR mean={sub['asr'].mean():.1f}%")

    groups = [df[df["trait"] == t]["proj_mean"].values for t in TRAITS]
    f_stat, p_anova = f_oneway(*groups)
    print(f"\n  ANOVA (across traits): F={f_stat:.3f}, p={p_anova:.3f}")

    trait_means = df.groupby("trait")["proj_mean"].mean()
    asr_vals    = [KNOWN_ASR_13B[t] for t in trait_means.index]
    r_p, p_p    = pearsonr(trait_means.values, asr_vals)
    r_s, p_s    = spearmanr(trait_means.values, asr_vals)
    print(f"\n  Trait mean projection vs ASR:")
    print(f"    Pearson  r={r_p:+.3f} (p={p_p:.3f})")
    print(f"    Spearman r={r_s:+.3f} (p={p_s:.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    positions = np.arange(len(TRAITS))
    for i, trait in enumerate(TRAITS):
        sub_vals = df[df["trait"] == trait]["proj_mean"].values
        ax.boxplot(sub_vals, positions=[i], widths=0.5,
                   patch_artist=True,
                   boxprops=dict(facecolor=TRAIT_COLORS[trait], alpha=0.7),
                   medianprops=dict(color="black", lw=2))
        orig_mask = (personas == trait) & (prompt_nums == "original")
        if orig_mask.sum() > 0:
            orig_proj = project(X_all[orig_mask], direction,
                                scaler_mean, scaler_scale).mean()
            ax.scatter(i, orig_proj, marker="D", color=TRAIT_COLORS[trait],
                       s=100, zorder=5, edgecolors="black", lw=1.5,
                       label="Original" if i == 0 else "")

    ax.set_xticks(positions)
    ax.set_xticklabels([t[:3] for t in TRAITS], fontsize=11)
    ax.set_ylabel("Projection onto Refusal Direction", fontsize=12)
    ax.set_title("(a) Paraphrase Projection Distribution by Trait", fontsize=13)
    ax.axhline(0, color="gray", linestyle="--", lw=0.8)
    ax.legend(fontsize=9)

    ax2 = ax.twinx()
    ax2.plot(positions, [KNOWN_ASR_13B[t] for t in TRAITS],
             "D--", color="crimson", lw=1.5, ms=7, label="ASR (%)")
    ax2.set_ylabel("ASR (%)", color="crimson", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="crimson")

    ax = axes[1]
    for trait in TRAITS:
        sub = df[df["trait"] == trait]
        ax.scatter(sub["proj_mean"], sub["asr"],
                   color=TRAIT_COLORS[trait], s=70, alpha=0.8,
                   label=trait[:3], zorder=5)
    for trait in TRAITS:
        sub = df[df["trait"] == trait]
        ax.scatter(sub["proj_mean"].mean(), KNOWN_ASR_13B[trait],
                   color=TRAIT_COLORS[trait], s=150, marker="D",
                   edgecolors="black", lw=1.5, zorder=6)

    ax.set_xlabel("Projection onto Refusal Direction (paraphrase mean)", fontsize=12)
    ax.set_ylabel("ASR (%)", fontsize=12)
    ax.set_title("(b) Paraphrase Projection vs ASR", fontsize=13)
    ax.axhline(0, color="gray", linestyle="--", lw=0.5)
    ax.legend(title="Trait", fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, "paraphrase_projection.pdf")
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    print(f"[plot] Saved: {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
