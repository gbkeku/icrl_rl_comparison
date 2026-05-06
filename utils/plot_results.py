# ─────────────────────────────────────────────
# Results table + bar chart visualization
# ─────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import os

ALGO_COLORS = {
    "grpo":      "#2563EB",
    "ppo":       "#16A34A",
    "reinforce": "#DC2626",
    "rloo":      "#D97706",
}
BENCHMARKS = ["trivia_qa","hotpot_qa","2wiki","musique","bamboogle"]
BENCH_LABELS = {
    "trivia_qa": "TriviaQA", "hotpot_qa": "HotpotQA",
    "2wiki": "2Wiki", "musique": "Musique", "bamboogle": "Bamboogle",
}


def plot_results_table(results, save_path="figures/results_table.png"):
    algos      = list(results.keys())
    cols       = BENCHMARKS + ["average"]
    col_labels = [BENCH_LABELS.get(c, c.title()) for c in BENCHMARKS] + ["Average"]
    cell_data  = []
    for algo in algos:
        row = [f"{results[algo].get(c, 0.0):.1f}" for c in cols]
        cell_data.append(row)

    fig, ax = plt.subplots(figsize=(13, len(algos) * 0.75 + 1.8))
    ax.axis("off")
    ax.set_title(
        "Exact Match Accuracy (%) — All Algorithms on 5 QA Benchmarks\n"
        "Backbone: Qwen2.5-1.5B-Instruct  |  Curriculum: 3→2→0",
        fontsize=11, fontweight="bold", pad=16
    )
    table = ax.table(cellText=cell_data, rowLabels=[a.upper() for a in algos],
                     colLabels=col_labels, cellLoc="center",
                     rowLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2.0)

    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#1E3A5F")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    for j, col in enumerate(cols):
        col_vals = [results[a].get(col, 0.0) for a in algos]
        best_idx = int(np.argmax(col_vals))
        for i, algo in enumerate(algos):
            cell      = table[(i + 1, j)]
            row_label = table[(i + 1, -1)]
            row_label.set_facecolor("#F1F5F9")
            row_label.set_text_props(
                color=ALGO_COLORS.get(algo, "#000"), fontweight="bold"
            )
            if i == best_idx:
                cell.set_facecolor("#DBEAFE")
                cell.set_text_props(fontweight="bold", color="#1E3A5F")
            else:
                cell.set_facecolor("#FFFFFF" if i % 2 == 0 else "#F8FAFC")

    for i in range(len(algos)):
        table[(i + 1, len(cols) - 1)].set_facecolor("#FEF9C3")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Results table saved → {save_path}")
    plt.show()


def plot_results_bar(results, save_path="figures/results_bar.png"):
    algos = list(results.keys())
    x     = np.arange(len(BENCHMARKS))
    width = 0.8 / len(algos)

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, algo in enumerate(algos):
        values = [results[algo].get(b, 0.0) for b in BENCHMARKS]
        offset = (i - len(algos) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, values, width,
                        label=algo.upper(),
                        color=ALGO_COLORS.get(algo, "#888"),
                        alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                        f"{h:.1f}", ha="center", va="bottom",
                        fontsize=7.5, color="#374151")

    ax.set_xticks(x)
    ax.set_xticklabels([BENCH_LABELS[b] for b in BENCHMARKS], fontsize=11)
    ax.set_ylabel("Exact Match Accuracy (%)", fontsize=11)
    ax.set_title(
        "RL Algorithm Comparison — Exact Match Accuracy per Benchmark\n"
        "Backbone: Qwen2.5-1.5B-Instruct  |  ICRL Curriculum: 3→2→0",
        fontsize=12, fontweight="bold"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    all_vals = [results[a].get(b, 0) for a in algos for b in BENCHMARKS]
    ax.set_ylim(0, max(all_vals) * 1.18 if all_vals else 100)
    ax.legend(fontsize=10, frameon=False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Bar chart saved → {save_path}")
    plt.show()


def plot_average_comparison(results, save_path="figures/average_em.png"):
    algos  = list(results.keys())
    avgs   = [results[a].get("average", 0.0) for a in algos]
    colors = [ALGO_COLORS.get(a, "#888") for a in algos]
    paired = sorted(zip(avgs, algos, colors), reverse=True)
    avgs, algos, colors = zip(*paired)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    bars = ax.barh([a.upper() for a in algos], avgs,
                   color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.5, height=0.45)
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", fontsize=11,
                fontweight="bold", color="#1E293B")

    ax.set_xlabel("Average Exact Match (%)", fontsize=11)
    ax.set_title("Average EM Accuracy Across All 5 Benchmarks",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.set_xlim(0, max(avgs) * 1.25 if avgs else 100)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Average EM chart saved → {save_path}")
    plt.show()
