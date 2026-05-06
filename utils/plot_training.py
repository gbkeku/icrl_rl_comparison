# ─────────────────────────────────────────────
# Training dynamics plots — mirrors Figure 3
# ─────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

ALGO_COLORS = {
    "grpo":      "#2563EB",
    "ppo":       "#16A34A",
    "reinforce": "#DC2626",
    "rloo":      "#D97706",
}
ALGO_LABELS = {
    "grpo":      "GRPO (baseline)",
    "ppo":       "PPO",
    "reinforce": "REINFORCE",
    "rloo":      "RLOO",
}
STAGE_COLORS = ["#DBEAFE", "#DCFCE7", "#FEF9C3"]
STAGE_LABELS = ["3-shot", "2-shot", "0-shot"]


def smooth(values, window=7):
    if len(values) < window:
        return values
    result = []
    for i in range(len(values)):
        s = max(0, i - window // 2)
        e = min(len(values), i + window // 2 + 1)
        result.append(np.mean(values[s:e]))
    return result


def plot_training_dynamics(logs, steps_per_stage=50,
                            save_path="figures/training_dynamics.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(
        "Training Dynamics — ICRL Curriculum (3-shot → 2-shot → 0-shot)",
        fontsize=13, fontweight="bold"
    )
    metrics = [
        ("reward",          "Reward (mean)",            axes[0]),
        ("response_length", "Response Length (chars)",   axes[1]),
        ("valid_searches",  "Valid Search Calls (mean)", axes[2]),
    ]
    for metric_key, ylabel, ax in metrics:
        for algo, log in logs.items():
            values = log.get(metric_key, [])
            if not values:
                continue
            x        = list(range(len(values)))
            smoothed = smooth(values, window=7)
            ax.plot(x, values, color=ALGO_COLORS.get(algo, "#888"),
                    alpha=0.15, linewidth=0.8)
            ax.plot(x, smoothed, color=ALGO_COLORS.get(algo, "#888"),
                    linewidth=2.2, label=ALGO_LABELS.get(algo, algo))
        for i in range(3):
            ax.axvspan(i * steps_per_stage, (i + 1) * steps_per_stage,
                       alpha=0.08, color=STAGE_COLORS[i], zorder=0)
            if i < 2:
                ax.axvline(x=(i + 1) * steps_per_stage,
                           color="#94A3B8", linestyle="--",
                           linewidth=0.8, zorder=1)
        ymin, ymax = ax.get_ylim()
        for i in range(3):
            ax.text(
                i * steps_per_stage + steps_per_stage / 2, ymax,
                STAGE_LABELS[i], ha="center", va="bottom",
                fontsize=8, color="#64748B", style="italic"
            )
        ax.set_xlabel("Training Step", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=8, frameon=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Training dynamics saved → {save_path}")
    plt.show()
