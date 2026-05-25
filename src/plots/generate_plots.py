"""
Generate Plots for Surgical Quantization Project
==================================================

Reads output JSON files and creates publication-ready figures.

Required files (place in same directory or adjust paths):
- layer_sensitivity_heatmap.json
- bidirectional_flip_report.json
- cross_task_interference_report.json
- optimal_mixed_precision_report.json (in results/ or ~/Downloads/)

Output: saves PNG files to current directory.
"""

import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# =============================================================================
# Configuration
# =============================================================================

# Try multiple paths for input files
SENSITIVITY_FILE = None
FLIP_FILE = None

for path in ["layer_sensitivity_heatmap.json", os.path.expanduser("~/Downloads/layer_sensitivity_heatmap.json")]:
    if os.path.exists(path):
        SENSITIVITY_FILE = path
        break

for path in ["bidirectional_flip_report.json", os.path.expanduser("~/Downloads/bidirectional_flip_report.json")]:
    if os.path.exists(path):
        FLIP_FILE = path
        break

INTERFERENCE_FILE = None
for path in [
    "cross_task_interference_report.json",
    os.path.expanduser("~/Downloads/cross_task_interference_report.json"),
]:
    if os.path.exists(path):
        INTERFERENCE_FILE = path
        break

CAPSTONE_FILE = None
for path in [
    "results/optimal_mixed_precision_report.json",
    "optimal_mixed_precision_report.json",
    os.path.expanduser("~/Downloads/optimal_mixed_precision_report.json"),
]:
    if os.path.exists(path):
        CAPSTONE_FILE = path
        break

ABLATION_FILE = None
for path in [
    "results/patch_count_ablation_report.json",
    "patch_count_ablation_report.json",
    os.path.expanduser("~/Downloads/patch_count_ablation_report.json"),
]:
    if os.path.exists(path):
        ABLATION_FILE = path
        break

SIGNIFICANCE_FILE = None
for path in [
    "results/multi_seed_significance_report.json",
    "multi_seed_significance_report.json",
    os.path.expanduser("~/Downloads/multi_seed_significance_report.json"),
]:
    if os.path.exists(path):
        SIGNIFICANCE_FILE = path
        break

MATH_STRAT_FILE = None
for path in [
    "results/math_difficulty_stratification_report.json",
    "math_difficulty_stratification_report.json",
    os.path.expanduser("~/Downloads/math_difficulty_stratification_report.json"),
]:
    if os.path.exists(path):
        MATH_STRAT_FILE = path
        break

MISTRAL_FILE = None
for path in [
    "results/mistral_replication_report.json",
    "mistral_replication_report.json",
    os.path.expanduser("~/Downloads/mistral_replication_report.json"),
]:
    if os.path.exists(path):
        MISTRAL_FILE = path
        break

PHASE21_FILE = None
for path in [
    "results/sensitivity_ranking_comparison.json",
    "sensitivity_ranking_comparison.json",
    os.path.expanduser("~/Downloads/sensitivity_ranking_comparison.json"),
]:
    if os.path.exists(path):
        PHASE21_FILE = path
        break

PHASE22_FILE = None
for path in [
    "results/multi_quant_probe_ckpt.json",
    "multi_quant_probe_ckpt.json",
    os.path.expanduser("~/Downloads/multi_quant_probe_ckpt.json"),
]:
    if os.path.exists(path):
        PHASE22_FILE = path
        break

PHASE23_FILE = None
for path in [
    "results/multi_model_probe.json",
    "multi_model_probe.json",
    os.path.expanduser("~/Downloads/multi_model_probe-2.json"),
    os.path.expanduser("~/Downloads/multi_model_probe.json"),
]:
    if os.path.exists(path):
        PHASE23_FILE = path
        break

PHASE24_FILE = None
for path in [
    "results/hessian_vs_restoration.json",
    "paper_a_mechanistic/results/hessian_vs_restoration.json",
    os.path.expanduser("~/Downloads/hessian_vs_restoration.json"),
]:
    if os.path.exists(path):
        PHASE24_FILE = path
        break

# Style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

COLORS = {
    "gsm8k": "#2196F3",   # Blue
    "arc": "#F44336",      # Red
    "math": "#4CAF50",     # Green
    "fp16": "#1976D2",     # Dark blue
    "4bit": "#D32F2F",     # Dark red
    "forward": "#E53935",  # Red
    "reverse": "#43A047",  # Green
    "both_correct": "#1976D2",   # Blue
    "both_incorrect": "#9E9E9E", # Grey
    "attention": "#FF9800",  # Orange
    "mlp": "#9C27B0",       # Purple
}


# =============================================================================
# Plot 1: Layer Sensitivity Heatmap
# =============================================================================

def plot_layer_sensitivity_heatmap(data):
    """
    Create a 2x3 heatmap grid showing recovery rate by layer.
    Rows: Attention, MLP. Columns: GSM8K, ARC, MATH.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 6), sharey=True)

    components = ["attention", "mlp"]
    datasets = ["gsm8k", "arc", "math"]
    dataset_labels = ["GSM8K (Math)", "ARC-Challenge (Science)", "MATH (Competition)"]
    component_labels = ["Attention", "MLP"]

    # Custom colormap: white → blue → dark blue
    cmap = LinearSegmentedColormap.from_list(
        "recovery", ["#FFFFFF", "#BBDEFB", "#2196F3", "#0D47A1"], N=256
    )

    num_layers = 32

    for row, component in enumerate(components):
        for col, (ds, ds_label) in enumerate(zip(datasets, dataset_labels)):
            ax = axes[row][col]

            layer_data = data["layer_sensitivity"][component][ds]
            values = [layer_data.get(str(i), 0.0) for i in range(num_layers)]

            # Reshape into 4x8 grid for visual clarity
            grid = np.array(values).reshape(4, 8)

            im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=max(0.55, max(values) + 0.05),
                          aspect="auto")

            # Label each cell
            for i in range(4):
                for j in range(8):
                    layer_num = i * 8 + j
                    val = grid[i, j]
                    color = "white" if val > 0.35 else "black"
                    ax.text(j, i, f"L{layer_num}\n{val:.0%}",
                           ha="center", va="center", fontsize=6.5,
                           color=color, fontweight="bold" if val > 0.3 else "normal")

            if row == 0:
                ax.set_title(ds_label, fontweight="bold")
            if col == 0:
                ax.set_ylabel(component_labels[row], fontweight="bold", fontsize=12)

            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Layer Sensitivity Heatmap: Recovery Rate by Layer\n"
                 "(FP16 transplant into single 4-bit layer, tested on flipped examples)",
                 fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig("plot1_layer_sensitivity_heatmap.png")
    plt.close()
    print("  Saved: plot1_layer_sensitivity_heatmap.png")


# =============================================================================
# Plot 2: Attention vs MLP — Top Layers Bar Chart
# =============================================================================

def plot_attention_vs_mlp_top_layers(data):
    """
    Side-by-side bar chart showing top-5 attention vs top-5 MLP layers per dataset.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    datasets = ["gsm8k", "arc", "math"]
    dataset_labels = ["GSM8K (Math)", "ARC-Challenge (Science)", "MATH (Competition)"]

    for idx, (ds, ds_label) in enumerate(zip(datasets, dataset_labels)):
        ax = axes[idx]

        # Get top 5 for each component
        attn_ranking = data["rankings"]["attention"][ds][:5]
        mlp_ranking = data["rankings"]["mlp"][ds][:5]

        # Create grouped bar positions
        x = np.arange(5)
        width = 0.35

        attn_vals = [r["recovery_rate"] for r in attn_ranking]
        attn_labels = [f"L{r['layer']}" for r in attn_ranking]
        mlp_vals = [r["recovery_rate"] for r in mlp_ranking]
        mlp_labels = [f"L{r['layer']}" for r in mlp_ranking]

        bars1 = ax.bar(x - width/2, attn_vals, width, label="Attention",
                       color=COLORS["attention"], alpha=0.85, edgecolor="white")
        bars2 = ax.bar(x + width/2, mlp_vals, width, label="MLP",
                       color=COLORS["mlp"], alpha=0.85, edgecolor="white")

        # Add value labels
        for bar, label in zip(bars1, attn_labels):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   label, ha="center", va="bottom", fontsize=7, fontweight="bold")
        for bar, label in zip(bars2, mlp_labels):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   label, ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_title(ds_label, fontweight="bold")
        ax.set_ylabel("Recovery Rate" if idx == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Rank {i+1}" for i in range(5)])
        ax.set_ylim(0, max(max(attn_vals), max(mlp_vals)) * 1.25)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend(loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Top-5 Most Critical Layers: Attention vs MLP per Dataset",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("plot2_attention_vs_mlp_top_layers.png")
    plt.close()
    print("  Saved: plot2_attention_vs_mlp_top_layers.png")


# =============================================================================
# Plot 3: Bidirectional Flip Summary
# =============================================================================

def plot_bidirectional_flips(data):
    """
    Stacked bar chart showing the 4-quadrant breakdown per dataset,
    plus net flip annotation.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    gridspec_kw={"width_ratios": [2, 1]})

    datasets = ["gsm8k", "arc", "math"]
    dataset_labels = ["GSM8K\n(Math)", "ARC\n(Science)", "MATH\n(Competition)"]

    # --- Left panel: stacked bar showing all 4 quadrants ---
    x = np.arange(len(datasets))
    width = 0.5

    both_correct = []
    forward_flips = []
    reverse_flips = []
    both_incorrect = []

    for ds in datasets:
        counts = data["analyses"][ds]["counts"]
        both_correct.append(counts["both_correct"])
        forward_flips.append(counts["forward_flips"])
        reverse_flips.append(counts["reverse_flips"])
        both_incorrect.append(counts["both_incorrect"])

    bc = np.array(both_correct)
    ff = np.array(forward_flips)
    rf = np.array(reverse_flips)
    bi = np.array(both_incorrect)

    ax1.bar(x, bc, width, label="Both Correct", color=COLORS["both_correct"], alpha=0.8)
    ax1.bar(x, ff, width, bottom=bc, label="Forward Flip (quant hurts)",
            color=COLORS["forward"], alpha=0.8)
    ax1.bar(x, rf, width, bottom=bc+ff, label="Reverse Flip (quant helps)",
            color=COLORS["reverse"], alpha=0.8)
    ax1.bar(x, bi, width, bottom=bc+ff+rf, label="Both Incorrect",
            color=COLORS["both_incorrect"], alpha=0.6)

    ax1.set_xticks(x)
    ax1.set_xticklabels(dataset_labels)
    ax1.set_ylabel("Number of Examples")
    ax1.set_title("Example Classification by Quantization Effect", fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # --- Right panel: forward vs reverse comparison ---
    x2 = np.arange(len(datasets))
    width2 = 0.3

    ax2.bar(x2 - width2/2, forward_flips, width2, label="Forward Flips",
            color=COLORS["forward"], alpha=0.85, edgecolor="white")
    ax2.bar(x2 + width2/2, reverse_flips, width2, label="Reverse Flips",
            color=COLORS["reverse"], alpha=0.85, edgecolor="white")

    # Add net annotations
    for i, ds in enumerate(datasets):
        net = forward_flips[i] - reverse_flips[i]
        y_pos = max(forward_flips[i], reverse_flips[i]) + 2
        ax2.annotate(f"Net: {net:+d}", xy=(i, y_pos),
                    ha="center", fontweight="bold", fontsize=10,
                    color=COLORS["forward"] if net > 0 else COLORS["reverse"])

    ax2.set_xticks(x2)
    ax2.set_xticklabels(dataset_labels)
    ax2.set_ylabel("Number of Flipped Examples")
    ax2.set_title("Forward vs Reverse Flips", fontweight="bold")
    ax2.legend()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Bidirectional Flip Analysis: Quantization Damage Characterization",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("plot3_bidirectional_flips.png")
    plt.close()
    print("  Saved: plot3_bidirectional_flips.png")


# =============================================================================
# Plot 4: ARC Confidence Distribution
# =============================================================================

def plot_arc_confidence(data):
    """
    Histogram + scatter comparing FP16 confidence for:
    - Stable correct examples
    - Forward flipped examples
    - Reverse flipped examples
    """
    arc = data["analyses"]["arc"]
    if "confidence_analysis" not in arc:
        print("  Skipping ARC confidence plot (no confidence data)")
        return

    ca = arc["confidence_analysis"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: FP16 confidence histograms ---
    bins = np.linspace(0, 1, 21)

    fwd_probs = ca["forward_flips"]["fp16_correct_probs"]
    rev_probs = ca["reverse_flips"]["fp16_correct_probs"]

    # We don't have individual stable_correct probs in the JSON, so show summary stats
    ax1.hist(fwd_probs, bins=bins, alpha=0.7, label=f"Forward Flips (n={len(fwd_probs)})",
             color=COLORS["forward"], edgecolor="white")
    ax1.hist(rev_probs, bins=bins, alpha=0.7, label=f"Reverse Flips (n={len(rev_probs)})",
             color=COLORS["reverse"], edgecolor="white")

    # Add mean lines
    fwd_mean = ca["forward_flips"]["mean_fp16_correct_prob"]
    rev_mean = ca["reverse_flips"]["mean_fp16_correct_prob"]
    stable_mean = ca["stable_correct"]["mean_fp16_correct_prob"]

    ax1.axvline(fwd_mean, color=COLORS["forward"], linestyle="--", linewidth=2,
               label=f"Forward mean: {fwd_mean:.3f}")
    ax1.axvline(rev_mean, color=COLORS["reverse"], linestyle="--", linewidth=2,
               label=f"Reverse mean: {rev_mean:.3f}")
    ax1.axvline(stable_mean, color=COLORS["both_correct"], linestyle="--", linewidth=2,
               label=f"Stable correct mean: {stable_mean:.3f}")

    ax1.set_xlabel("FP16 Probability for Correct Answer")
    ax1.set_ylabel("Count")
    ax1.set_title("FP16 Confidence on Correct Answer\n(ARC-Challenge)", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # --- Right: FP16 vs 4-bit confidence scatter for flipped examples ---
    fwd_fp16 = ca["forward_flips"]["fp16_correct_probs"]
    fwd_int4 = ca["forward_flips"]["int4_correct_probs"]
    rev_fp16 = ca["reverse_flips"]["fp16_correct_probs"]
    rev_int4 = ca["reverse_flips"]["int4_correct_probs"]

    ax2.scatter(fwd_fp16, fwd_int4, c=COLORS["forward"], alpha=0.7, s=50,
               label=f"Forward Flips (n={len(fwd_fp16)})", edgecolors="white", linewidth=0.5)
    ax2.scatter(rev_fp16, rev_int4, c=COLORS["reverse"], alpha=0.7, s=50,
               label=f"Reverse Flips (n={len(rev_fp16)})", edgecolors="white", linewidth=0.5,
               marker="^")

    # Diagonal line (equal confidence)
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Equal confidence")

    ax2.set_xlabel("FP16 P(correct answer)")
    ax2.set_ylabel("4-bit P(correct answer)")
    ax2.set_title("FP16 vs 4-bit Confidence\n(ARC-Challenge Flipped Examples)", fontweight="bold")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_aspect("equal")

    fig.suptitle("Confidence Analysis: Is Quantization Damage Boundary Perturbation?",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("plot4_arc_confidence.png")
    plt.close()
    print("  Saved: plot4_arc_confidence.png")


# =============================================================================
# Plot 5: Layer Sensitivity Profile (Line Chart)
# =============================================================================

def plot_layer_sensitivity_lines(data):
    """
    Line chart showing recovery rate across all 32 layers for each dataset,
    with attention and MLP as separate panels.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    layers = list(range(32))

    for ds, color, label in [
        ("gsm8k", COLORS["gsm8k"], "GSM8K"),
        ("arc", COLORS["arc"], "ARC-Challenge"),
        ("math", COLORS["math"], "MATH"),
    ]:
        # Attention
        attn_vals = [data["layer_sensitivity"]["attention"][ds].get(str(i), 0) for i in layers]
        ax1.plot(layers, attn_vals, color=color, marker="o", markersize=4,
                linewidth=1.5, label=label, alpha=0.85)

        # MLP
        mlp_vals = [data["layer_sensitivity"]["mlp"][ds].get(str(i), 0) for i in layers]
        ax2.plot(layers, mlp_vals, color=color, marker="s", markersize=4,
                linewidth=1.5, label=label, alpha=0.85)

    # Annotate key layers on attention plot
    ax1.annotate("L14: 51.6%", xy=(14, 0.516), xytext=(18, 0.50),
                arrowprops=dict(arrowstyle="->", color=COLORS["gsm8k"]),
                fontsize=8, color=COLORS["gsm8k"], fontweight="bold")
    ax1.annotate("L13: 48.4%", xy=(13, 0.484), xytext=(8, 0.48),
                arrowprops=dict(arrowstyle="->", color=COLORS["gsm8k"]),
                fontsize=8, color=COLORS["gsm8k"], fontweight="bold")

    # Annotate dead zone on ARC
    ax1.axvspan(15, 31, alpha=0.05, color="red")
    ax1.text(23, 0.45, "ARC Dead Zone\n(layers 15-31)", fontsize=8,
            color=COLORS["arc"], alpha=0.7, ha="center", style="italic")

    for ax, title in [(ax1, "Attention Transplant"), (ax2, "MLP Transplant")]:
        ax.set_ylabel("Recovery Rate")
        ax.set_title(title, fontweight="bold", loc="left")
        ax.legend(loc="upper right")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

    ax2.set_xlabel("Layer Index")
    ax2.set_xticks(range(0, 32, 2))

    fig.suptitle("Per-Layer Recovery Rate: Which Layers Are Precision-Sensitive?",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("plot5_layer_sensitivity_lines.png")
    plt.close()
    print("  Saved: plot5_layer_sensitivity_lines.png")


# =============================================================================
# Plot 6: Summary Dashboard
# =============================================================================

def plot_summary_dashboard(sensitivity_data, flip_data):
    """
    Single-figure dashboard summarizing key findings.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # --- Panel A: Accuracy comparison ---
    ax_a = fig.add_subplot(gs[0, 0])

    datasets = ["GSM8K", "ARC", "MATH"]
    fp16_accs = [
        flip_data["analyses"]["gsm8k"]["fp16_accuracy"],
        flip_data["analyses"]["arc"]["fp16_accuracy"],
        flip_data["analyses"]["math"]["fp16_accuracy"],
    ]
    int4_accs = [
        flip_data["analyses"]["gsm8k"]["int4_accuracy"],
        flip_data["analyses"]["arc"]["int4_accuracy"],
        flip_data["analyses"]["math"]["int4_accuracy"],
    ]

    x = np.arange(3)
    w = 0.3
    ax_a.bar(x - w/2, fp16_accs, w, label="FP16", color=COLORS["fp16"], alpha=0.85)
    ax_a.bar(x + w/2, int4_accs, w, label="4-bit NF4", color=COLORS["4bit"], alpha=0.85)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(datasets)
    ax_a.set_ylabel("Accuracy")
    ax_a.set_title("(A) FP16 vs 4-bit Accuracy", fontweight="bold")
    ax_a.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax_a.legend()
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # Add drop annotations
    for i in range(3):
        drop = fp16_accs[i] - int4_accs[i]
        ax_a.annotate(f"{drop:+.1%}", xy=(i, min(fp16_accs[i], int4_accs[i]) - 0.02),
                     ha="center", fontsize=9, color=COLORS["forward"], fontweight="bold")

    # --- Panel B: Forward vs Reverse flips ---
    ax_b = fig.add_subplot(gs[0, 1])

    ff = [flip_data["analyses"][ds]["counts"]["forward_flips"] for ds in ["gsm8k", "arc", "math"]]
    rf = [flip_data["analyses"][ds]["counts"]["reverse_flips"] for ds in ["gsm8k", "arc", "math"]]

    ax_b.bar(x - w/2, ff, w, label="Forward (quant hurts)", color=COLORS["forward"], alpha=0.85)
    ax_b.bar(x + w/2, rf, w, label="Reverse (quant helps)", color=COLORS["reverse"], alpha=0.85)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(datasets)
    ax_b.set_ylabel("Count")
    ax_b.set_title("(B) Bidirectional Flips", fontweight="bold")
    ax_b.legend(fontsize=8)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    for i in range(3):
        net = ff[i] - rf[i]
        ax_b.text(i, max(ff[i], rf[i]) + 1.5, f"Net: {net:+d}",
                 ha="center", fontweight="bold", fontsize=9)

    # --- Panel C: Best single-layer recovery ---
    ax_c = fig.add_subplot(gs[0, 2])

    best_attn = []
    best_mlp = []
    for ds in ["gsm8k", "arc", "math"]:
        best_attn.append(sensitivity_data["rankings"]["attention"][ds][0]["recovery_rate"])
        best_mlp.append(sensitivity_data["rankings"]["mlp"][ds][0]["recovery_rate"])

    ax_c.bar(x - w/2, best_attn, w, label="Best Attention Layer",
            color=COLORS["attention"], alpha=0.85)
    ax_c.bar(x + w/2, best_mlp, w, label="Best MLP Layer",
            color=COLORS["mlp"], alpha=0.85)

    # Add layer labels
    for i, ds in enumerate(["gsm8k", "arc", "math"]):
        attn_layer = sensitivity_data["rankings"]["attention"][ds][0]["layer"]
        mlp_layer = sensitivity_data["rankings"]["mlp"][ds][0]["layer"]
        ax_c.text(i - w/2, best_attn[i] + 0.01, f"L{attn_layer}",
                 ha="center", fontsize=8, fontweight="bold")
        ax_c.text(i + w/2, best_mlp[i] + 0.01, f"L{mlp_layer}",
                 ha="center", fontsize=8, fontweight="bold")

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(datasets)
    ax_c.set_ylabel("Recovery Rate")
    ax_c.set_title("(C) Attention vs MLP Dominance", fontweight="bold")
    ax_c.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax_c.legend(fontsize=8)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # --- Panel D & E: Layer sensitivity lines (bottom spanning 2 cols) ---
    ax_d = fig.add_subplot(gs[1, 0:2])

    layers = list(range(32))
    for ds, color, label in [
        ("gsm8k", COLORS["gsm8k"], "GSM8K"),
        ("arc", COLORS["arc"], "ARC"),
        ("math", COLORS["math"], "MATH"),
    ]:
        attn_vals = [sensitivity_data["layer_sensitivity"]["attention"][ds].get(str(i), 0) for i in layers]
        ax_d.plot(layers, attn_vals, color=color, marker="o", markersize=3,
                 linewidth=1.5, label=f"{label} Attn", alpha=0.85)

        mlp_vals = [sensitivity_data["layer_sensitivity"]["mlp"][ds].get(str(i), 0) for i in layers]
        ax_d.plot(layers, mlp_vals, color=color, marker="s", markersize=3,
                 linewidth=1.5, label=f"{label} MLP", alpha=0.4, linestyle="--")

    ax_d.set_xlabel("Layer Index")
    ax_d.set_ylabel("Recovery Rate")
    ax_d.set_title("(D) Full Layer Sensitivity Profile (Attention=solid, MLP=dashed)", fontweight="bold")
    ax_d.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax_d.legend(ncol=3, fontsize=7, loc="upper right")
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    ax_d.set_xticks(range(0, 32, 2))
    ax_d.grid(axis="y", alpha=0.15)

    # --- Panel F: ARC confidence ---
    ax_f = fig.add_subplot(gs[1, 2])

    if "confidence_analysis" in flip_data["analyses"]["arc"]:
        ca = flip_data["analyses"]["arc"]["confidence_analysis"]

        categories = ["Stable\nCorrect", "Forward\nFlips", "Reverse\nFlips"]
        means = [
            ca["stable_correct"]["mean_fp16_correct_prob"],
            ca["forward_flips"]["mean_fp16_correct_prob"],
            ca["reverse_flips"]["mean_fp16_correct_prob"],
        ]
        medians = [
            ca["stable_correct"]["median_fp16_correct_prob"],
            ca["forward_flips"]["median_fp16_correct_prob"],
            ca["reverse_flips"]["median_fp16_correct_prob"],
        ]

        bar_colors = [COLORS["both_correct"], COLORS["forward"], COLORS["reverse"]]

        bars = ax_f.bar(range(3), means, 0.5, color=bar_colors, alpha=0.85,
                       edgecolor="white", linewidth=1.5)

        # Add median markers
        for i, med in enumerate(medians):
            ax_f.plot([i - 0.2, i + 0.2], [med, med], "k-", linewidth=2)
            ax_f.text(i + 0.22, med, f"med: {med:.2f}", fontsize=7, va="center")

        ax_f.set_xticks(range(3))
        ax_f.set_xticklabels(categories)
        ax_f.set_ylabel("FP16 P(correct)")
        ax_f.set_title("(E) ARC Confidence Analysis\n(bars=mean, lines=median)", fontweight="bold")
        ax_f.spines["top"].set_visible(False)
        ax_f.spines["right"].set_visible(False)

    fig.suptitle("Surgical Quantization: Mapping Functional Specialization in Llama-3.1-8B",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    plt.savefig("plot6_summary_dashboard.png")
    plt.close()
    print("  Saved: plot6_summary_dashboard.png")


# =============================================================================
# Plot 7: Cross-Task Interference Matrix
# =============================================================================

def plot_cross_task_interference(data):
    """
    Heatmap showing recovery rate when applying each task-specific patch
    to all three datasets, plus grouped bar chart for full accuracy.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                                    gridspec_kw={"width_ratios": [1.2, 1]})

    matrix = data["interference_matrix"]

    # --- Left: Recovery rate heatmap ---
    conditions = ["4bit_baseline", "gsm8k_optimal", "arc_optimal", "math_optimal", "all_combined"]
    condition_labels = ["4-bit Baseline", "GSM8K-opt\n(Attn L13,14)", "ARC-opt\n(MLP L6,7)",
                        "MATH-opt\n(MLP L1,31)", "All Combined"]
    datasets = ["gsm8k", "arc", "math"]
    dataset_labels = ["GSM8K", "ARC", "MATH"]

    recovery_grid = np.array([
        [matrix[cond][ds]["recovery_rate"] for ds in datasets]
        for cond in conditions
    ])

    cmap = LinearSegmentedColormap.from_list(
        "interference", ["#FFFFFF", "#C8E6C9", "#4CAF50", "#1B5E20"], N=256
    )

    im = ax1.imshow(recovery_grid, cmap=cmap, vmin=0, vmax=0.75, aspect="auto")

    for i in range(len(conditions)):
        for j in range(len(datasets)):
            val = recovery_grid[i, j]
            color = "white" if val > 0.5 else "black"
            ax1.text(j, i, f"{val:.1%}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    ax1.set_xticks(range(len(datasets)))
    ax1.set_xticklabels(dataset_labels, fontweight="bold")
    ax1.set_yticks(range(len(conditions)))
    ax1.set_yticklabels(condition_labels, fontsize=9)
    ax1.set_title("Recovery Rate on Flipped Examples", fontweight="bold")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label("Recovery Rate")

    # Highlight diagonal (on-target patches) - rows 1,2,3 cols 0,1,2
    for idx, (row, col) in enumerate([(1, 0), (2, 1), (3, 2)]):
        rect = plt.Rectangle((col - 0.5, row - 0.5), 1, 1,
                             linewidth=2.5, edgecolor="#FF6F00", facecolor="none")
        ax1.add_patch(rect)

    # --- Right: Full accuracy grouped bars ---
    # Skip baseline, show the 4 patch conditions
    patch_conditions = ["gsm8k_optimal", "arc_optimal", "math_optimal", "all_combined"]
    patch_labels = ["GSM8K-opt", "ARC-opt", "MATH-opt", "All Combined"]

    x = np.arange(len(datasets))
    n_conds = len(patch_conditions)
    total_width = 0.7
    bar_width = total_width / n_conds

    bar_colors = [COLORS["gsm8k"], COLORS["arc"], COLORS["math"], "#FF9800"]

    # Draw 4-bit baseline as horizontal lines
    for j, ds in enumerate(datasets):
        baseline_acc = matrix["4bit_baseline"][ds]["full_accuracy"]
        ax2.hlines(baseline_acc, j - 0.4, j + 0.4, colors="gray",
                  linestyles="dashed", linewidth=1.5,
                  label="4-bit baseline" if j == 0 else None)

    for i, (cond, label) in enumerate(zip(patch_conditions, patch_labels)):
        offsets = x - total_width/2 + bar_width * (i + 0.5)
        accs = [matrix[cond][ds]["full_accuracy"] for ds in datasets]
        ax2.bar(offsets, accs, bar_width, label=label, color=bar_colors[i],
                alpha=0.85, edgecolor="white")

    ax2.set_xticks(x)
    ax2.set_xticklabels(dataset_labels, fontweight="bold")
    ax2.set_ylabel("Full Accuracy (500 examples)")
    ax2.set_title("Full Accuracy by Patch Condition", fontweight="bold")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.legend(fontsize=7, loc="lower right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_ylim(0.5, 0.85)

    fig.suptitle("Cross-Task Interference: Do Task-Specific Patches Help Other Tasks?\n"
                 "(Orange borders = on-target patch; off-diagonal = cross-task spillover)",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig("plot7_cross_task_interference.png")
    plt.close()
    print("  Saved: plot7_cross_task_interference.png")


# =============================================================================
# Plot 8: Optimal Mixed-Precision Capstone
# =============================================================================

def plot_optimal_mixed_precision(data):
    """
    Multi-panel figure showing the capstone results:
    - Accuracy comparison (3 models x 3 datasets)
    - Gap recovery percentages
    - Memory footprint comparison
    """
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3,
                           width_ratios=[1.2, 1, 0.8])

    results = data["results"]

    datasets = ["gsm8k", "arc", "math"]
    dataset_labels = ["GSM8K", "ARC", "MATH"]

    fp16_accs = [results["fp16"][ds]["accuracy"] for ds in datasets]
    int4_accs = [results["int4"][ds]["accuracy"] for ds in datasets]
    surg_accs = [results["surgical"][ds]["accuracy"] for ds in datasets]

    # --- Panel A: Accuracy comparison ---
    ax_a = fig.add_subplot(gs[0, 0])
    x = np.arange(len(datasets))
    w = 0.22

    bars_fp16 = ax_a.bar(x - w, fp16_accs, w, label="Full FP16",
                          color=COLORS["fp16"], alpha=0.85, edgecolor="white")
    bars_int4 = ax_a.bar(x, int4_accs, w, label="Full 4-bit (NF4)",
                          color=COLORS["4bit"], alpha=0.85, edgecolor="white")
    bars_surg = ax_a.bar(x + w, surg_accs, w, label="Surgical Mix (6/32 layers)",
                          color="#FF9800", alpha=0.85, edgecolor="white")

    # Value labels on top
    for bars in [bars_fp16, bars_int4, bars_surg]:
        for bar in bars:
            h = bar.get_height()
            ax_a.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                     f"{h:.1%}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(dataset_labels, fontweight="bold")
    ax_a.set_ylabel("Accuracy")
    ax_a.set_title("(A) Accuracy: FP16 vs 4-bit vs Surgical", fontweight="bold")
    ax_a.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax_a.legend(fontsize=8, loc="lower right")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # Set y-axis range to show differences clearly
    min_val = min(min(fp16_accs), min(int4_accs), min(surg_accs))
    ax_a.set_ylim(max(0, min_val - 0.08), 1.0)

    # --- Panel B: Gap recovery ---
    ax_b = fig.add_subplot(gs[0, 1])

    gap_recoveries = []
    for i, ds in enumerate(datasets):
        gap = fp16_accs[i] - int4_accs[i]
        if gap > 0:
            recovery = (surg_accs[i] - int4_accs[i]) / gap * 100
        else:
            recovery = 100.0
        gap_recoveries.append(recovery)

    avg_recovery = np.mean(gap_recoveries)

    bar_colors_recovery = [COLORS["gsm8k"], COLORS["arc"], COLORS["math"]]
    bars = ax_b.bar(range(len(datasets)), gap_recoveries, 0.5,
                    color=bar_colors_recovery, alpha=0.85, edgecolor="white")

    # Average line
    ax_b.axhline(avg_recovery, color="black", linestyle="--", linewidth=1.5,
                 label=f"Average: {avg_recovery:.1f}%")

    for i, (bar, val) in enumerate(zip(bars, gap_recoveries)):
        ax_b.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax_b.set_xticks(range(len(datasets)))
    ax_b.set_xticklabels(dataset_labels, fontweight="bold")
    ax_b.set_ylabel("% of FP16-4bit Gap Recovered")
    ax_b.set_title("(B) Gap Recovery by Dataset", fontweight="bold")
    ax_b.set_ylim(0, 100)
    ax_b.legend(fontsize=9)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # --- Panel C: Memory footprint ---
    ax_c = fig.add_subplot(gs[0, 2])

    # Use theoretical/reported values
    int4_mem = results["int4"]["memory_mb"]["peak_mb"]
    surg_overhead = results["surgical"]["patch_info"]["overhead_mb"]
    surg_mem = int4_mem + surg_overhead  # Theoretical persistent size
    fp16_mem = 16000  # Approximate

    models = ["Full FP16", "Surgical\nMix", "Full\n4-bit"]
    mems = [fp16_mem, surg_mem, int4_mem]
    mem_colors = [COLORS["fp16"], "#FF9800", COLORS["4bit"]]

    bars = ax_c.barh(range(len(models)), mems, 0.5, color=mem_colors,
                     alpha=0.85, edgecolor="white")

    for i, (bar, val) in enumerate(zip(bars, mems)):
        ax_c.text(val + 100, bar.get_y() + bar.get_height()/2,
                 f"{val:,.0f} MB", va="center", fontsize=10, fontweight="bold")

    ax_c.set_yticks(range(len(models)))
    ax_c.set_yticklabels(models, fontweight="bold")
    ax_c.set_xlabel("GPU Memory (MB)")
    ax_c.set_title("(C) Memory Footprint", fontweight="bold")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.invert_yaxis()

    # Add annotation about efficiency
    ax_c.text(0.5, -0.15,
             f"Surgical model: {surg_mem/fp16_mem*100:.0f}% of FP16 memory,\n"
             f"{avg_recovery:.0f}% of accuracy gap recovered",
             transform=ax_c.transAxes, ha="center", fontsize=9,
             style="italic", color="#555555")

    fig.suptitle("Optimal Mixed-Precision Model: 6/32 Layers Patched (18.75% of Network)\n"
                 "Llama-3.1-8B-Instruct | 500 examples per dataset",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig("plot8_optimal_mixed_precision.png")
    plt.close()
    print("  Saved: plot8_optimal_mixed_precision.png")


# =============================================================================
# Plot 9: Patch Count Ablation — Diminishing Returns Curve
# =============================================================================

def plot_patch_count_ablation(data):
    """
    Multi-panel figure showing diminishing returns as layers are added:
    - Left: per-dataset accuracy curves with layer annotations
    - Right: marginal gain per layer + memory overhead
    """
    curve = data["diminishing_returns_curve"]
    layer_order = data["config"]["layer_order"]
    layer_labels = [
        "MLP L6", "MLP L7", "Attn L13", "Attn L14", "MLP L1", "MLP L31"
    ]

    num_layers = [c["num_layers"] for c in curve]
    gsm8k_accs = [c["gsm8k"] for c in curve]
    arc_accs = [c["arc"] for c in curve]
    math_accs = [c["math"] for c in curve]
    avg_accs = [c["average"] for c in curve]
    overheads = [c["overhead_mb"] for c in curve]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7),
                                    gridspec_kw={"width_ratios": [1.3, 1]})

    # --- Left: Accuracy curves ---
    ax1.plot(num_layers, gsm8k_accs, color=COLORS["gsm8k"], marker="o",
             markersize=8, linewidth=2.5, label="GSM8K", zorder=5)
    ax1.plot(num_layers, arc_accs, color=COLORS["arc"], marker="s",
             markersize=8, linewidth=2.5, label="ARC", zorder=5)
    ax1.plot(num_layers, math_accs, color=COLORS["math"], marker="^",
             markersize=8, linewidth=2.5, label="MATH", zorder=5)
    ax1.plot(num_layers, avg_accs, color="black", marker="D",
             markersize=7, linewidth=2, label="Average", linestyle="--",
             alpha=0.7, zorder=5)

    # Shade the interference zone (layer 5)
    ax1.axvspan(4.5, 5.5, alpha=0.1, color="red", zorder=0)
    ax1.annotate("Interference!\nMLP L1 hurts",
                xy=(5, min(gsm8k_accs[5], avg_accs[5]) - 0.005),
                xytext=(5, min(math_accs) - 0.025),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                fontsize=9, color="red", fontweight="bold", ha="center")

    # Mark sweet spot at 4 layers
    ax1.axvline(4, color="#FF9800", linestyle=":", linewidth=2, alpha=0.7)
    ax1.annotate("Sweet spot\n(4 layers, 624 MB)",
                xy=(4, avg_accs[4] + 0.003),
                xytext=(2.2, avg_accs[4] + 0.018),
                arrowprops=dict(arrowstyle="->", color="#FF9800", lw=1.5),
                fontsize=9, color="#FF9800", fontweight="bold")

    # X-axis layer labels
    x_labels = ["0\n(4-bit)"]
    for i, label in enumerate(layer_labels):
        x_labels.append(f"{i+1}\n+{label}")
    ax1.set_xticks(num_layers)
    ax1.set_xticklabels(x_labels, fontsize=8)
    ax1.set_xlabel("Number of FP16 Layers Patched")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy vs Patch Count", fontweight="bold")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax1.legend(loc="lower right", fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.15)

    # --- Right: Marginal gains + memory overhead ---
    marginal_avg = [0]  # no marginal gain for baseline
    for i in range(1, len(curve)):
        marginal_avg.append(avg_accs[i] - avg_accs[i - 1])

    # Bar chart for marginal gains (skip index 0)
    bar_colors = []
    for m in marginal_avg[1:]:
        bar_colors.append("#4CAF50" if m >= 0 else "#F44336")

    bars = ax2.bar(range(1, len(curve)), [m * 100 for m in marginal_avg[1:]],
                   0.5, color=bar_colors, alpha=0.85, edgecolor="white")

    # Value labels on bars
    for i, (bar, val) in enumerate(zip(bars, marginal_avg[1:])):
        y_pos = bar.get_height() + 0.05 if val >= 0 else bar.get_height() - 0.15
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                f"{val*100:+.1f}pp", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=9, fontweight="bold")

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(range(1, len(curve)))
    ax2.set_xticklabels(layer_labels, fontsize=8, rotation=15)
    ax2.set_ylabel("Marginal Gain in Average Accuracy (pp)")
    ax2.set_title("Marginal Gain per Layer Added", fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Secondary y-axis for memory overhead
    ax2b = ax2.twinx()
    ax2b.plot(range(1, len(curve)), overheads[1:], color="#9E9E9E",
             marker="o", markersize=6, linewidth=1.5, linestyle="--",
             label="Memory overhead", alpha=0.7)
    ax2b.set_ylabel("Memory Overhead (MB)", color="#9E9E9E")
    ax2b.tick_params(axis="y", labelcolor="#9E9E9E")
    ax2b.spines["top"].set_visible(False)

    # Combined legend
    ax2.legend(["Marginal gain"], loc="upper left", fontsize=8)
    ax2b.legend(loc="upper right", fontsize=8)

    fig.suptitle("Patch Count Ablation: How Many Layers Do You Actually Need?\n"
                 "Llama-3.1-8B-Instruct | 500 examples per dataset",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig("plot9_patch_count_ablation.png")
    plt.close()
    print("  Saved: plot9_patch_count_ablation.png")


# =============================================================================
# Plot 10: Multi-Seed Statistical Significance
# =============================================================================

def plot_math_difficulty_stratification(data):
    """2-panel: (1) grouped accuracy bars per tier, (2) gap + surgical gain with p-value annotation."""
    summary = data["tier_summary"]

    tier_labels = [r["tier_label"] for r in summary]
    fp16_acc  = [r["fp16_accuracy"] * 100 for r in summary]
    int4_acc  = [r["int4_accuracy"] * 100 for r in summary]
    surg_acc  = [r["surgical_accuracy"] * 100 for r in summary]
    gaps_pp   = [r["gap_pp"] for r in summary]
    surg_gain = [r["surgical_gain_pp"] for r in summary]
    p_vals    = [r["fp16_vs_int4_p"] for r in summary]
    sig_flags = [r["fp16_vs_int4_sig"] for r in summary]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # ── Panel 1: Grouped accuracy bars ───────────────────────────────────────
    x = np.arange(len(tier_labels))
    width = 0.25
    cond_colors = [COLORS["fp16"], COLORS["4bit"], "#43A047"]
    cond_labels = ["FP16", "4-bit NF4", "Surgical Mix"]
    accs = [fp16_acc, int4_acc, surg_acc]

    for i, (acc_vals, color, label) in enumerate(zip(accs, cond_colors, cond_labels)):
        bars = ax1.bar(x + (i - 1) * width, acc_vals, width,
                       label=label, color=color, alpha=0.85, zorder=3)
        for bar, val in zip(bars, acc_vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    # Annotate p-values above FP16 bars
    for i, (p, is_sig) in enumerate(zip(p_vals, sig_flags)):
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        y_top = max(fp16_acc[i], int4_acc[i], surg_acc[i]) + 2.5
        ax1.text(i, y_top, f"FP16 vs 4-bit\np={p:.3f} ({stars})",
                 ha="center", fontsize=8, color="#555555",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.8))

    ax1.set_xticks(x)
    ax1.set_xticklabels(tier_labels, fontsize=11)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, max(fp16_acc) * 1.25)
    ax1.set_title("Accuracy by Difficulty Tier\n(Easy: L1-3 vs Hard: L4-5)", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    # ── Panel 2: Gap + surgical gain bars ────────────────────────────────────
    x2 = np.arange(len(tier_labels))
    width2 = 0.3

    bars_gap  = ax2.bar(x2 - width2 / 2, gaps_pp, width2,
                        label="FP16–4-bit gap (damage)", color=COLORS["4bit"], alpha=0.8, zorder=3)
    bars_gain = ax2.bar(x2 + width2 / 2, surg_gain, width2,
                        label="Surgical gain over 4-bit", color="#43A047", alpha=0.8, zorder=3)

    for bar, val in zip(bars_gap, gaps_pp):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.05 if val >= 0 else bar.get_height() - 0.3,
                 f"{val:+.1f}pp", ha="center", va="bottom", fontsize=10,
                 fontweight="bold", color=COLORS["4bit"])

    for bar, val in zip(bars_gain, surg_gain):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.05 if val >= 0 else bar.get_height() - 0.3,
                 f"{val:+.1f}pp", ha="center", va="bottom", fontsize=10,
                 fontweight="bold", color="#2E7D32")

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(tier_labels, fontsize=11)
    ax2.set_ylabel("Accuracy Difference (pp)")
    ax2.set_title("Quantization Damage & Surgical Recovery\n(All comparisons non-significant)",
                  fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle(
        "MATH Difficulty Stratification: Quantization Damage is Capability-Limited, Not Precision-Limited\n"
        "Llama-3.1-8B-Instruct | MATH Easy (L1-3) vs Hard (L4-5) | n=500 per tier | McNemar's test",
        fontsize=12, fontweight="bold", y=1.03
    )
    plt.tight_layout()
    plt.savefig("plot11_math_difficulty_stratification.png")
    plt.close()
    print("  Saved: plot11_math_difficulty_stratification.png")


def plot_multi_seed_significance(data):
    """3-panel: (1) mean±std accuracy bars, (2) McNemar p-values, (3) efficiency comparison."""
    summary = data["summary"]
    stats = data["statistical_tests"]["mcnemar"]
    efficiency = data["efficiency"]

    tasks = ["gsm8k", "arc", "math"]
    task_labels = ["GSM8K", "ARC-Challenge", "MATH"]
    conditions = ["fp16", "int4", "surgical"]
    cond_labels = ["FP16", "4-bit NF4", "Surgical Mix"]
    cond_colors = [COLORS["fp16"], COLORS["4bit"], "#43A047"]

    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    # ── Panel 1: Mean ± std accuracy bars ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    x = np.arange(len(tasks))
    width = 0.25
    offsets = [-width, 0, width]

    for i, (cond, label, color) in enumerate(zip(conditions, cond_labels, cond_colors)):
        means = [summary[cond][t]["mean"] * 100 for t in tasks]
        stds  = [summary[cond][t]["std"] * 100 for t in tasks]
        bars = ax1.bar(x + offsets[i], means, width, label=label,
                       color=color, alpha=0.85, zorder=3,
                       yerr=stds, capsize=4, error_kw={"elinewidth": 1.5})
        for bar, m in zip(bars, means):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                     f"{m:.1f}", ha="center", va="bottom", fontsize=7.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(task_labels)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(25, 92)
    ax1.set_title("Mean Accuracy ± Std\n(5 seeds × 500 examples)", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    # ── Panel 2: McNemar significance heatmap ────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    comparisons = [
        ("fp16_vs_int4",     "FP16 vs 4-bit"),
        ("fp16_vs_surgical", "FP16 vs Surgical"),
        ("int4_vs_surgical", "4-bit vs Surgical"),
    ]
    p_matrix = np.zeros((len(comparisons), len(tasks)))
    sig_matrix = np.zeros((len(comparisons), len(tasks)), dtype=bool)
    for row_i, (key, _) in enumerate(comparisons):
        for col_j, task in enumerate(tasks):
            p_val = stats[key][task]["p_value"]
            p_matrix[row_i, col_j] = p_val
            sig_matrix[row_i, col_j] = stats[key][task]["significant_at_0.05"]

    # Color by significance: green=significant, red=not
    cell_colors = np.where(sig_matrix, 0.0, 1.0)  # 0=sig (green end), 1=not sig (red end)
    cmap_sig = LinearSegmentedColormap.from_list("sig", ["#43A047", "#E53935"])
    im = ax2.imshow(cell_colors, cmap=cmap_sig, vmin=0, vmax=1, aspect="auto")

    ax2.set_xticks(range(len(tasks)))
    ax2.set_xticklabels(task_labels, fontsize=9)
    ax2.set_yticks(range(len(comparisons)))
    ax2.set_yticklabels([c[1] for c in comparisons], fontsize=9)

    for row_i in range(len(comparisons)):
        for col_j, task in enumerate(tasks):
            p_val = p_matrix[row_i, col_j]
            sig = sig_matrix[row_i, col_j]
            stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
            label = f"p={p_val:.3f}\n{stars}"
            color = "white" if sig else "black"
            ax2.text(col_j, row_i, label, ha="center", va="center",
                     fontsize=8.5, fontweight="bold", color=color)

    ax2.set_title("McNemar's Test\n(paired per-example, α=0.05)", fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)

    # ── Panel 3: Efficiency comparison ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    eff_labels = ["FP16", "4-bit NF4", "Surgical Mix"]
    throughputs = [efficiency[c]["throughput_tok_per_s"]["mean"] for c in conditions]
    mem_gb = [efficiency[c]["memory_mb"] / 1024 for c in conditions]
    acc_per_gb = [efficiency[c]["accuracy_per_gb"] for c in conditions]

    bar_x = np.arange(len(eff_labels))
    bars3 = ax3.bar(bar_x, throughputs, color=cond_colors, alpha=0.85, width=0.5, zorder=3)
    for bar, thr in zip(bars3, throughputs):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{thr:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax3.set_xticks(bar_x)
    ax3.set_xticklabels(eff_labels, fontsize=9)
    ax3.set_ylabel("Throughput (tok/s)")
    ax3.set_title("Efficiency Comparison\n(throughput, memory, acc/GB)", fontweight="bold")
    ax3.set_ylim(0, 260)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.grid(axis="y", alpha=0.3, zorder=0)

    # Annotate memory and acc/GB below bars
    for i, (m, a) in enumerate(zip(mem_gb, acc_per_gb)):
        ax3.text(i, 10, f"{m:.1f} GB\n{a:.3f} acc/GB",
                 ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")

    fig.suptitle(
        "Multi-Seed Statistical Significance: Surgical Mix ≡ FP16, Surgical Mix > 4-bit\n"
        "Llama-3.1-8B-Instruct | 5 seeds × 500 examples | McNemar's test",
        fontsize=13, fontweight="bold", y=1.03
    )
    plt.savefig("plot10_multi_seed_significance.png")
    plt.close()
    print("  Saved: plot10_multi_seed_significance.png")


def plot_cross_model_comparison(mistral_data, sensitivity_data):
    """Plot 12: Cross-model comparison (Llama vs Mistral)."""
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

    datasets = ["gsm8k", "arc", "math"]
    ds_labels = {"gsm8k": "GSM8K", "arc": "ARC", "math": "MATH"}
    llama_color = "#1976D2"
    mistral_color = "#E65100"

    # --- Panel 1: Capstone accuracy comparison (top-left) ---
    ax1 = fig.add_subplot(gs[0, 0])
    # Llama baselines from sensitivity_data (if available)
    llama_baselines = sensitivity_data.get("baselines", {}) if sensitivity_data else {}
    mistral_baselines = mistral_data["baselines"]
    mistral_capstone = mistral_data["capstone"]

    x = np.arange(len(datasets))
    width = 0.15
    conditions = ["fp16", "int4", "surgical"]
    labels_cond = ["FP16", "4-bit", "Surgical"]

    for i, (cond, lbl) in enumerate(zip(conditions, labels_cond)):
        # Mistral capstone
        vals_m = [mistral_capstone[cond][ds]["accuracy"] * 100 for ds in datasets]
        ax1.bar(x + i * width + width / 2, vals_m, width, label=f"Mistral {lbl}",
                color=mistral_color, alpha=0.4 + 0.2 * i, edgecolor=mistral_color)

    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels([ds_labels[ds] for ds in datasets])
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Mistral-7B: Capstone Accuracy")
    ax1.legend(fontsize=8, ncol=3)
    ax1.set_ylim(0, 85)

    # --- Panel 2: Layer sensitivity profile comparison (top-right) ---
    ax2 = fig.add_subplot(gs[0, 1])
    m_rankings = mistral_data["layer_rankings"]
    layers = list(range(32))

    # GSM8K attention scores for both models
    m_attn = [m_rankings["gsm8k"]["attention_scores"].get(str(l), 0) * 100 for l in layers]
    ax2.plot(layers, m_attn, '-o', color=mistral_color, markersize=3, linewidth=1.5,
             label="Mistral Attention (GSM8K)")

    if sensitivity_data and "layer_results" in sensitivity_data:
        l_attn = []
        for l in layers:
            key = f"layer_{l}_attention"
            val = sensitivity_data["layer_results"].get(key, {}).get("gsm8k", {}).get("recovery_rate", 0)
            l_attn.append(val * 100)
        ax2.plot(layers, l_attn, '-s', color=llama_color, markersize=3, linewidth=1.5,
                 label="Llama Attention (GSM8K)")

    # Highlight L14
    ax2.axvline(x=14, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax2.annotate("L14", xy=(14, max(m_attn) * 0.95), fontsize=8, color='red', ha='center')
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Recovery Rate (%)")
    ax2.set_title("Attention Sensitivity: GSM8K (Both Models)")
    ax2.legend(fontsize=8)

    # --- Panel 3: ARC MLP comparison (middle-left) ---
    ax3 = fig.add_subplot(gs[1, 0])
    m_mlp_arc = [m_rankings["arc"]["mlp_scores"].get(str(l), 0) * 100 for l in layers]
    ax3.bar(layers, m_mlp_arc, color=mistral_color, alpha=0.7, label="Mistral MLP (ARC)")

    if sensitivity_data and "layer_results" in sensitivity_data:
        l_mlp_arc = []
        for l in layers:
            key = f"layer_{l}_mlp"
            val = sensitivity_data["layer_results"].get(key, {}).get("arc", {}).get("recovery_rate", 0)
            l_mlp_arc.append(val * 100)
        ax3.bar(layers, l_mlp_arc, color=llama_color, alpha=0.4, label="Llama MLP (ARC)")

    # Highlight L6-7
    for hl in [6, 7]:
        ax3.axvline(x=hl, color='red', linestyle='--', alpha=0.4, linewidth=1)
    ax3.set_xlabel("Layer Index")
    ax3.set_ylabel("Recovery Rate (%)")
    ax3.set_title("MLP Sensitivity: ARC (Both Models)")
    ax3.legend(fontsize=8)

    # --- Panel 4: Interference matrix heatmap (middle-right) ---
    ax4 = fig.add_subplot(gs[1, 1])
    interf = mistral_data["interference"]
    conditions_if = ["int4_baseline", "gsm8k_opt_attn", "arc_opt_mlp", "all_combined"]
    cond_labels = ["4-bit\nbaseline", "GSM8K-opt\nAttention", "ARC-opt\nMLP", "All\ncombined"]

    matrix = []
    for cond in conditions_if:
        row = [interf[cond][ds]["recovery_rate"] * 100 for ds in datasets]
        matrix.append(row)
    matrix = np.array(matrix)

    im = ax4.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=60)
    ax4.set_xticks(range(len(datasets)))
    ax4.set_xticklabels([ds_labels[ds] for ds in datasets])
    ax4.set_yticks(range(len(conditions_if)))
    ax4.set_yticklabels(cond_labels, fontsize=8)
    for i in range(len(conditions_if)):
        for j in range(len(datasets)):
            ax4.text(j, i, f"{matrix[i, j]:.1f}%", ha='center', va='center', fontsize=9,
                     color='white' if matrix[i, j] > 30 else 'black')
    plt.colorbar(im, ax=ax4, label="Recovery Rate (%)", shrink=0.8)
    ax4.set_title("Mistral-7B: Cross-Task Interference")

    # --- Panel 5: Statistical comparison table (bottom-left) ---
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis("off")
    stats = mistral_data["statistics"]
    table_data = [
        ["", "GSM8K", "ARC", "MATH"],
        ["FP16 mean", f"{stats['gsm8k']['fp16_mean']:.1%}",
         f"{stats['arc']['fp16_mean']:.1%}", f"{stats['math']['fp16_mean']:.1%}"],
        ["4-bit mean", f"{stats['gsm8k']['int4_mean']:.1%}",
         f"{stats['arc']['int4_mean']:.1%}", f"{stats['math']['int4_mean']:.1%}"],
        ["Surgical mean", f"{stats['gsm8k']['surgical_mean']:.1%}",
         f"{stats['arc']['surgical_mean']:.1%}", f"{stats['math']['surgical_mean']:.1%}"],
        ["FP16 vs 4-bit",
         f"p={stats['gsm8k']['fp16_vs_int4']['p_value']:.3f} {'✓' if stats['gsm8k']['fp16_vs_int4']['significant_at_0.05'] else 'ns'}",
         f"p={stats['arc']['fp16_vs_int4']['p_value']:.3f} {'✓' if stats['arc']['fp16_vs_int4']['significant_at_0.05'] else 'ns'}",
         f"p={stats['math']['fp16_vs_int4']['p_value']:.3f} {'✓' if stats['math']['fp16_vs_int4']['significant_at_0.05'] else 'ns'}"],
        ["Surg vs FP16",
         f"p={stats['gsm8k']['fp16_vs_surgical']['p_value']:.3f} {'✓' if stats['gsm8k']['fp16_vs_surgical']['significant_at_0.05'] else 'ns'}",
         f"p={stats['arc']['fp16_vs_surgical']['p_value']:.3f} {'✓' if stats['arc']['fp16_vs_surgical']['significant_at_0.05'] else 'ns'}",
         f"p={stats['math']['fp16_vs_surgical']['p_value']:.3f} {'✓' if stats['math']['fp16_vs_surgical']['significant_at_0.05'] else 'ns'}"],
        ["Surg vs 4-bit",
         f"p={stats['gsm8k']['int4_vs_surgical']['p_value']:.3f} {'✓' if stats['gsm8k']['int4_vs_surgical']['significant_at_0.05'] else 'ns'}",
         f"p={stats['arc']['int4_vs_surgical']['p_value']:.3f} {'✓' if stats['arc']['int4_vs_surgical']['significant_at_0.05'] else 'ns'}",
         f"p={stats['math']['int4_vs_surgical']['p_value']:.3f} {'✓' if stats['math']['int4_vs_surgical']['significant_at_0.05'] else 'ns'}"],
    ]
    table = ax5.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    # Style header row
    for j in range(4):
        table[0, j].set_facecolor('#E8EAF6')
        table[0, j].set_text_props(fontweight='bold')
    # Highlight significant cells
    for i in range(4, 7):
        for j in range(1, 4):
            if '✓' in table_data[i][j]:
                table[i, j].set_facecolor('#C8E6C9')
    ax5.set_title("Mistral-7B: McNemar's Test (3 seeds, n=1500)", fontsize=11, pad=20)

    # --- Panel 6: Cross-model key finding summary (bottom-right) ---
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")
    summary_data = [
        ["Finding", "Llama-3.1-8B", "Mistral-7B"],
        ["Critical attn layer", "L13, L14", "L14, L0"],
        ["Critical MLP layers", "L1, L6, L7, L31", "L3, L5, L6, L7"],
        ["Quantization effect", "Damages (−2.4pp GSM8K)", "Neutral (ns)"],
        ["Surgical vs FP16", "Equivalent (ns)", "Surgical > FP16 (p<0.05)"],
        ["Surgical vs 4-bit", "Surgical > 4-bit (p<0.01)", "Equivalent (ns)"],
        ["ARC dead zone", "L15–31", "L23–31"],
        ["MATH result", "Null (capability)", "Null (capability)"],
        ["Attn monosemantic?", "Yes", "Yes"],
        ["MLP superposition?", "Yes", "Yes"],
    ]
    table2 = ax6.table(cellText=summary_data, loc='center', cellLoc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(8)
    table2.scale(1, 1.35)
    for j in range(3):
        table2[0, j].set_facecolor('#E8EAF6')
        table2[0, j].set_text_props(fontweight='bold')
    # Color the "Yes" cells green
    for i in range(1, len(summary_data)):
        for j in range(1, 3):
            if summary_data[i][j] == "Yes":
                table2[i, j].set_facecolor('#C8E6C9')
            elif "Null" in summary_data[i][j] or "Neutral" in summary_data[i][j]:
                table2[i, j].set_facecolor('#FFF9C4')
    ax6.set_title("Cross-Model Comparison Summary", fontsize=11, pad=20)

    fig.suptitle("Plot 12: Cross-Model Replication — Llama-3.1-8B vs Mistral-7B",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.savefig("plot12_cross_model_comparison.png")
    plt.close()
    print("  Saved: plot12_cross_model_comparison.png")


# =============================================================================
# Plot 13: Restoration vs Hessian Rank Scatter (Phase 21)
# =============================================================================

def plot_restoration_vs_hessian(data):
    """Scatter plot of restoration rank vs Hessian sensitivity rank for all 64 sub-layers.

    Shows the negative correlation (ρ = -0.506) between functional importance
    (restoration-based) and numerical sensitivity (Hessian-based), demonstrating
    that our probe captures structure orthogonal to GPTQ/ResQ methods.
    """
    per_layer = data["per_layer"]

    # Collect data points
    attn_h, attn_r, attn_labels = [], [], []
    mlp_h, mlp_r, mlp_labels = [], [], []
    # critical layers for annotation
    critical = {"L13_attention", "L14_attention", "L6_mlp", "L7_mlp", "L1_mlp", "L0_mlp"}

    for key, v in per_layer.items():
        hrank = v["hessian_rank"]
        rrank = v["restoration_rank"]
        if v["component"] == "attention":
            attn_h.append(hrank)
            attn_r.append(rrank)
            attn_labels.append(key if key in critical else None)
        else:
            mlp_h.append(hrank)
            mlp_r.append(rrank)
            mlp_labels.append(key if key in critical else None)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Scatter: attention
    ax.scatter(attn_h, attn_r, c=COLORS["attention"], s=60, alpha=0.85,
               label="Attention sub-layers", edgecolors="white", linewidths=0.5, zorder=3)
    # Scatter: MLP
    ax.scatter(mlp_h, mlp_r, c=COLORS["mlp"], s=60, alpha=0.85,
               label="MLP sub-layers", edgecolors="white", linewidths=0.5,
               marker="s", zorder=3)

    # Annotate critical layers
    for (labels, hvals, rvals) in [
        (attn_labels, attn_h, attn_r),
        (mlp_labels, mlp_h, mlp_r),
    ]:
        for lbl, hv, rv in zip(labels, hvals, rvals):
            if lbl is None:
                continue
            short = lbl.replace("_attention", "\nattn").replace("_mlp", "\nMLP")
            # Offset annotation to avoid overlap
            xoff = 2 if hv < 40 else -2
            yoff = 2 if rv < 50 else -4
            ax.annotate(
                short,
                xy=(hv, rv),
                xytext=(hv + xoff, rv + yoff),
                fontsize=7.5,
                ha="center",
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
                zorder=5,
            )

    # OLS regression line over full range
    all_h = np.array(attn_h + mlp_h, dtype=float)
    all_r = np.array(attn_r + mlp_r, dtype=float)
    m, b = np.polyfit(all_h, all_r, 1)
    xs = np.linspace(1, 64, 100)
    ax.plot(xs, m * xs + b, color="#E53935", lw=1.5, ls="--", alpha=0.7, zorder=2,
            label=f"Trend (slope={m:.2f})")

    # Quadrant shading
    ax.axhline(32.5, color="grey", lw=0.6, ls=":", alpha=0.5)
    ax.axvline(32.5, color="grey", lw=0.6, ls=":", alpha=0.5)
    ax.text(5, 60, "High functional\nLow Hessian\n(divergent +)",
            fontsize=7.5, color="#558B2F", alpha=0.9, va="top")
    ax.text(40, 8, "Low functional\nHigh Hessian\n(divergent −)",
            fontsize=7.5, color="#C62828", alpha=0.9, va="bottom")

    # Correlation annotation
    rho = data["correlations"]["restoration_aggregate_vs_hessian"]["spearman_rho"]
    p   = data["correlations"]["restoration_aggregate_vs_hessian"]["spearman_p"]
    ax.text(0.97, 0.97,
            f"Spearman ρ = {rho:.3f}\np < 0.001, n = 64",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFF9C4", ec="#F9A825", lw=1))

    ax.set_xlabel("Hessian Sensitivity Rank  (1 = least sensitive, 64 = most sensitive)",
                  fontsize=10)
    ax.set_ylabel("Restoration Rank  (1 = lowest recovery, 64 = highest recovery)",
                  fontsize=10)
    ax.set_title("Restoration vs. Hessian Sensitivity Rankings\n"
                 "Functional importance is inversely predicted by activation-space sensitivity",
                 fontsize=11, pad=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 66)
    ax.set_ylim(0, 66)
    ax.grid(True, alpha=0.25, lw=0.5)

    plt.tight_layout()
    plt.savefig("plot13_restoration_vs_hessian.png")
    plt.close()
    print("  Saved: plot13_restoration_vs_hessian.png")


# =============================================================================
# Plot 14: GPTQ Cross-Method Validation (Phase 22)
# =============================================================================

def plot_gptq_cross_method(data22, nf4_data):
    """
    Compare NF4 vs GPTQ recovery rates across all 32 layers.
    Shows that L14 attention task-selectivity replicates across quantization methods.
    2×2: Attention GSM8K, Attention ARC, MLP ARC, MLP GSM8K.
    """
    gptq_ls = data22["methods"]["gptq"]["layer_sensitivity"]
    nf4_ls = nf4_data["layer_sensitivity"] if nf4_data else None

    layers = list(range(32))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    panels = [
        (axes[0, 0], "attention", "gsm8k", "Attention — GSM8K (Arithmetic)"),
        (axes[0, 1], "attention", "arc",   "Attention — ARC (Science Knowledge)"),
        (axes[1, 0], "mlp",       "arc",   "MLP — ARC (Science Knowledge)"),
        (axes[1, 1], "mlp",       "gsm8k", "MLP — GSM8K (Arithmetic)"),
    ]

    ds_colors = {"gsm8k": COLORS["gsm8k"], "arc": COLORS["arc"]}

    for ax, comp, ds, title in panels:
        color = ds_colors[ds]
        gptq_vals = [gptq_ls[comp][ds].get(str(i), 0) for i in layers]
        ax.plot(layers, gptq_vals, color=color, marker="o", markersize=4,
                linewidth=2, label="GPTQ", alpha=0.9)

        if nf4_ls:
            nf4_vals = [nf4_ls[comp][ds].get(str(i), 0) for i in layers]
            ax.plot(layers, nf4_vals, color=color, marker="s", markersize=4,
                    linewidth=2, linestyle="--", label="NF4", alpha=0.6)

        if comp == "attention" and ds == "gsm8k":
            ax.annotate(f"L14: {gptq_ls['attention']['gsm8k'].get('14', 0):.1%}",
                        xy=(14, gptq_ls["attention"]["gsm8k"].get("14", 0)),
                        xytext=(18, 0.42),
                        arrowprops=dict(arrowstyle="->", color=color),
                        fontsize=8, color=color, fontweight="bold")
        if comp == "attention" and ds == "arc":
            ax.axvspan(15, 31, alpha=0.05, color="red")
            ax.text(23, ax.get_ylim()[1] * 0.8 if ax.get_ylim()[1] > 0 else 0.15,
                    "Dead Zone\n(L15–31)", fontsize=7, color=COLORS["arc"],
                    ha="center", style="italic", alpha=0.7)
        if comp == "mlp" and ds == "arc":
            ax.annotate(f"L7: {gptq_ls['mlp']['arc'].get('7', 0):.1%}",
                        xy=(7, gptq_ls["mlp"]["arc"].get("7", 0)),
                        xytext=(12, 0.25),
                        arrowprops=dict(arrowstyle="->", color=color),
                        fontsize=8, color=color, fontweight="bold")

        ax.set_title(title, fontweight="bold", loc="left", fontsize=10)
        ax.set_ylabel("Recovery Rate")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend(loc="upper right", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

    for ax in axes[1]:
        ax.set_xlabel("Layer Index")
        ax.set_xticks(range(0, 32, 4))

    fig.suptitle(
        "Cross-Quantization-Method Replication: NF4 vs GPTQ Recovery Rates",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("plot14_gptq_cross_method.png")
    plt.close()
    print("  Saved: plot14_gptq_cross_method.png")


# =============================================================================
# Plot 15: Multi-Model Phase 23 Probe Results
# =============================================================================

def plot_multi_model_probe(data23):
    """
    Two panels:
    - Left: FP16 vs NF4 accuracy for each model × task (GSM8K, MATH).
      Visually shows Qwen2.5-Math and Gemma-2 are quantization-resistant.
    - Right: Top attention-L13/L14 GSM8K recovery across models with damage.
      Replication of task-selective arithmetic attention.
    """
    # Hardcoded reference values for Llama (Phase 9) and Mistral (Phase 16)
    REF_MODELS = {
        "llama": {
            "label": "Llama-3.1-8B",
            "fp16": {"gsm8k": 0.802, "math": 0.620},
            "nf4": {"gsm8k": 0.756, "math": 0.605},
            "top_attn_gsm8k": {"13": 0.484, "14": 0.516},
        },
        "mistral": {
            "label": "Mistral-7B",
            "fp16": {"gsm8k": 0.482, "math": None},
            "nf4": {"gsm8k": 0.482, "math": None},
            "top_attn_gsm8k": {"13": 0.35, "14": 0.42},
        },
    }

    MODEL_LABELS = {
        "qwen_math":   "Qwen2.5-Math-7B",
        "deepseek_r1": "DeepSeek-R1-Distill",
        "gemma2":      "Gemma-2-9B",
    }

    # ── Panel 1: FP16 vs NF4 for GSM8K ────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    all_model_keys = ["llama", "mistral", "deepseek_r1", "qwen_math", "gemma2"]
    all_labels = [
        "Llama-3.1-8B", "Mistral-7B", "DeepSeek-R1\n(Distill)",
        "Qwen2.5-Math-7B", "Gemma-2-9B",
    ]

    fp16_gsm = []
    nf4_gsm = []
    for mk in all_model_keys:
        if mk in REF_MODELS:
            fp16_gsm.append(REF_MODELS[mk]["fp16"]["gsm8k"])
            nf4_gsm.append(REF_MODELS[mk]["nf4"]["gsm8k"])
        else:
            mv = data23["models"][mk]
            fp16_gsm.append(mv["fp16_baselines"]["gsm8k"])
            nf4_gsm.append(mv["nf4_baselines"]["gsm8k"])

    x = np.arange(len(all_model_keys))
    w = 0.35
    bars_fp16 = ax1.bar(x - w / 2, [v * 100 for v in fp16_gsm], w,
                        label="FP16", color=COLORS["fp16"], alpha=0.85)
    bars_nf4 = ax1.bar(x + w / 2, [v * 100 for v in nf4_gsm], w,
                       label="NF4 (4-bit)", color=COLORS["4bit"], alpha=0.75)

    # Annotate gap only where damage is notable (>1pp)
    for i, (fp, nf) in enumerate(zip(fp16_gsm, nf4_gsm)):
        gap = (fp - nf) * 100
        if abs(gap) > 1.0:
            ax1.text(x[i], max(fp, nf) * 100 + 1.0,
                     f"{'−' if gap > 0 else '+'}{abs(gap):.1f}pp",
                     ha="center", va="bottom", fontsize=7, color="black")
        else:
            ax1.text(x[i], max(fp, nf) * 100 + 1.0,
                     "≈0", ha="center", va="bottom", fontsize=7,
                     color="grey", style="italic")

    ax1.set_xticks(x)
    ax1.set_xticklabels(all_labels, fontsize=8)
    ax1.set_ylabel("GSM8K Accuracy (%)")
    ax1.set_title("Quantization Damage by Model\n(FP16 vs NF4, GSM8K)",
                  fontweight="bold", loc="left")
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 110)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.2)

    # ── Panel 2: Top attention-L13 GSM8K recovery across damaged models ────────
    damaged_models = {
        "llama":       ("Llama-3.1-8B",       0.484, 0.516),   # L13, L14
        "mistral":     ("Mistral-7B",          0.350, 0.420),   # approx
        "deepseek_r1": ("DeepSeek-R1\n(Distill)", None, None),
    }

    # Get DeepSeek L13 value from data
    ds_attn = data23["models"]["deepseek_r1"]["layer_sensitivity"]["attention"]["gsm8k"]
    deepseek_l13 = ds_attn.get("13", 0)
    deepseek_l14 = ds_attn.get("14", 0)
    damaged_models["deepseek_r1"] = ("DeepSeek-R1\n(Distill)", deepseek_l13, deepseek_l14)

    model_names_d = [v[0] for v in damaged_models.values()]
    l13_vals = [v[1] * 100 if v[1] is not None else 0 for v in damaged_models.values()]
    l14_vals = [v[2] * 100 if v[2] is not None else 0 for v in damaged_models.values()]

    xd = np.arange(len(model_names_d))
    ax2.bar(xd - 0.2, l13_vals, 0.38, label="L13 Attention", color=COLORS["attention"], alpha=0.85)
    ax2.bar(xd + 0.2, l14_vals, 0.38, label="L14 Attention", color="#FF5722", alpha=0.75)

    ax2.set_xticks(xd)
    ax2.set_xticklabels(model_names_d, fontsize=9)
    ax2.set_ylabel("GSM8K Recovery Rate (%)")
    ax2.set_title("Attention L13/L14 GSM8K Recovery\nAcross Models (Replication)",
                  fontweight="bold", loc="left")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 65)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.2)

    # Annotate Qwen/Gemma as resistant in panel 1
    for i, mk in enumerate(all_model_keys):
        if mk in ("qwen_math", "gemma2"):
            ax1.text(x[i], 5, "resistant", ha="center", va="bottom",
                     fontsize=7, color="grey", style="italic", rotation=90)

    fig.suptitle(
        "Phase 23: Multi-Model Probe — Replication vs. Quantization Resistance",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("plot15_multi_model_probe.png")
    plt.close()
    print("  Saved: plot15_multi_model_probe.png")


# =============================================================================
# Main
# =============================================================================

def plot_hessian_vs_restoration(data24: dict) -> None:
    """Plot 16: Hessian-guided vs Restoration-guided mixed-precision comparison.

    Two-panel figure:
      Left  — absolute accuracy for all 4 conditions × 3 datasets.
      Right — recovery rates (HESS-6 vs REST-6) per dataset.
    """
    results = data24["results"]
    datasets = ["gsm8k", "arc", "math"]
    labels = ["GSM8K", "ARC", "MATH"]

    # --- Accuracy data -------------------------------------------------------
    fp16_acc  = [results[d]["fp16_acc"]             * 100 for d in datasets]
    nf4_acc   = [results[d]["nf4_acc"]              * 100 for d in datasets]
    hess_acc  = [results[d]["hessian_surgical_acc"]  * 100 for d in datasets]
    rest_acc  = [results[d]["restoration_surgical_acc"] * 100 for d in datasets]

    # --- Recovery rate data --------------------------------------------------
    hess_rec = [results[d]["hessian_recovery_rate"]     * 100 for d in datasets]
    rest_rec = [results[d]["restoration_recovery_rate"] * 100 for d in datasets]

    # --- Colours (colorblind-safe Okabe-Ito subset) --------------------------
    C_FP16  = "#0072B2"   # blue
    C_NF4   = "#D55E00"   # vermillion
    C_HESS  = "#999999"   # grey
    C_REST  = "#009E73"   # green

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Panel 1: Absolute accuracy ───────────────────────────────────────────
    ax = axes[0]
    x = np.arange(len(datasets))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    bars_data = [
        (fp16_acc,  C_FP16,  "FP16"),
        (nf4_acc,   C_NF4,   "NF4 (4-bit)"),
        (hess_acc,  C_HESS,  "HESS-6"),
        (rest_acc,  C_REST,  "REST-6 (ours)"),
    ]
    for (vals, color, label), offset in zip(bars_data, offsets):
        bars = ax.bar(x + offset * width, vals, width * 0.9,
                      color=color, label=label, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7, color="black")

    # Significance bracket on ARC (index 1): REST vs NF4 p=0.009
    arc_x = 1
    nf4_h  = nf4_acc[arc_x]
    rest_h = rest_acc[arc_x]
    bracket_top = max(nf4_h, rest_h) + 3.5
    ax.annotate(
        "", xy=(arc_x + 0.5 * width, bracket_top),
        xytext=(arc_x - 0.5 * width, bracket_top),
        arrowprops=dict(arrowstyle="-", color="black", lw=1.2),
    )
    ax.text(arc_x, bracket_top + 0.8, r"$p{=}0.009$", ha="center",
            fontsize=8, color="black", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Absolute Accuracy: 4 Conditions")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_ylim(0, max(fp16_acc) * 1.15)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # ── Panel 2: Recovery rates ───────────────────────────────────────────────
    ax2 = axes[1]
    x2 = np.arange(len(datasets))
    w2 = 0.32
    b_hess = ax2.bar(x2 - w2 / 2, hess_rec, w2 * 0.9,
                     color=C_HESS, label="HESS-6", zorder=3)
    b_rest = ax2.bar(x2 + w2 / 2, rest_rec, w2 * 0.9,
                     color=C_REST, label="REST-6 (ours)", zorder=3)

    for bars in (b_hess, b_rest):
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                     f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    # ARC significance: REST vs HESS p=0.001 (rerun-3 with statsmodels: p=0.0011)
    arc_top = max(hess_rec[1], rest_rec[1]) + 5
    ax2.annotate(
        "", xy=(x2[1] + w2 / 2, arc_top),
        xytext=(x2[1] - w2 / 2, arc_top),
        arrowprops=dict(arrowstyle="-", color="black", lw=1.2),
    )
    ax2.text(x2[1], arc_top + 1.5, r"$p{=}0.001$", ha="center",
             fontsize=8, color="black", style="italic")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Recovery Rate (%)\n(NF4-flipped examples recovered)")
    ax2.set_title("Recovery Rates: HESS-6 vs REST-6")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.set_ylim(0, 105)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax2.set_axisbelow(True)
    ax2.axhline(y=50, color="black", linestyle=":", linewidth=0.8, alpha=0.6)

    fig.suptitle(
        "Phase 24: Hessian-Guided vs Restoration-Guided Mixed Precision\n"
        r"HESS-6 = $\{$L22,24,25,27,28,30 attn$\}$  ·  "
        r"REST-6 = $\{$L13,14 attn; L1,6,7,31 MLP$\}$",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = "plot16_hessian_vs_restoration.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


def main():
    print("=" * 60)
    print("Generating Plots")
    print("=" * 60)

    sensitivity_data = None
    flip_data = None
    interference_data = None
    capstone_data = None
    ablation_data = None
    significance_data = None

    if SENSITIVITY_FILE:
        with open(SENSITIVITY_FILE) as f:
            sensitivity_data = json.load(f)
        print(f"  Loaded: {SENSITIVITY_FILE}")
    else:
        print("  WARNING: layer_sensitivity_heatmap.json not found")

    if FLIP_FILE:
        with open(FLIP_FILE) as f:
            flip_data = json.load(f)
        print(f"  Loaded: {FLIP_FILE}")
    else:
        print("  WARNING: bidirectional_flip_report.json not found")

    if INTERFERENCE_FILE:
        with open(INTERFERENCE_FILE) as f:
            interference_data = json.load(f)
        print(f"  Loaded: {INTERFERENCE_FILE}")
    else:
        print("  WARNING: cross_task_interference_report.json not found")

    if CAPSTONE_FILE:
        with open(CAPSTONE_FILE) as f:
            capstone_data = json.load(f)
        print(f"  Loaded: {CAPSTONE_FILE}")
    else:
        print("  WARNING: optimal_mixed_precision_report.json not found")

    if ABLATION_FILE:
        with open(ABLATION_FILE) as f:
            ablation_data = json.load(f)
        print(f"  Loaded: {ABLATION_FILE}")
    else:
        print("  WARNING: patch_count_ablation_report.json not found")

    if SIGNIFICANCE_FILE:
        with open(SIGNIFICANCE_FILE) as f:
            significance_data = json.load(f)
        print(f"  Loaded: {SIGNIFICANCE_FILE}")
    else:
        print("  WARNING: multi_seed_significance_report.json not found")

    math_strat_data = None
    if MATH_STRAT_FILE:
        with open(MATH_STRAT_FILE) as f:
            math_strat_data = json.load(f)
        print(f"  Loaded: {MATH_STRAT_FILE}")
    else:
        print("  WARNING: math_difficulty_stratification_report.json not found")

    mistral_data = None
    if MISTRAL_FILE:
        with open(MISTRAL_FILE) as f:
            mistral_data = json.load(f)
        print(f"  Loaded: {MISTRAL_FILE}")
    else:
        print("  WARNING: mistral_replication_report.json not found")

    print()

    if sensitivity_data:
        plot_layer_sensitivity_heatmap(sensitivity_data)
        plot_attention_vs_mlp_top_layers(sensitivity_data)
        plot_layer_sensitivity_lines(sensitivity_data)

    if flip_data:
        plot_bidirectional_flips(flip_data)
        plot_arc_confidence(flip_data)

    if sensitivity_data and flip_data:
        plot_summary_dashboard(sensitivity_data, flip_data)

    if interference_data:
        plot_cross_task_interference(interference_data)

    if capstone_data:
        plot_optimal_mixed_precision(capstone_data)

    if ablation_data:
        plot_patch_count_ablation(ablation_data)

    if significance_data:
        plot_multi_seed_significance(significance_data)

    if math_strat_data:
        plot_math_difficulty_stratification(math_strat_data)

    if mistral_data:
        plot_cross_model_comparison(mistral_data, sensitivity_data)

    phase21_data = None
    if PHASE21_FILE:
        with open(PHASE21_FILE) as f:
            phase21_data = json.load(f)
        print(f"  Loaded: {PHASE21_FILE}")
    else:
        print("  WARNING: sensitivity_ranking_comparison.json not found")

    if phase21_data:
        plot_restoration_vs_hessian(phase21_data)

    phase22_data = None
    if PHASE22_FILE:
        with open(PHASE22_FILE) as f:
            phase22_data = json.load(f)
        print(f"  Loaded: {PHASE22_FILE}")
    else:
        print("  WARNING: multi_quant_probe_ckpt.json not found")

    if phase22_data:
        plot_gptq_cross_method(phase22_data, sensitivity_data)

    phase23_data = None
    if PHASE23_FILE:
        with open(PHASE23_FILE) as f:
            phase23_data = json.load(f)
        print(f"  Loaded: {PHASE23_FILE}")
    else:
        print("  WARNING: multi_model_probe.json not found")

    if phase23_data:
        plot_multi_model_probe(phase23_data)

    phase24_data = None
    if PHASE24_FILE:
        with open(PHASE24_FILE) as f:
            phase24_data = json.load(f)
        print(f"  Loaded: {PHASE24_FILE}")
    else:
        print("  WARNING: hessian_vs_restoration.json not found")

    if phase24_data:
        plot_hessian_vs_restoration(phase24_data)

    print("\nDone! All plots saved.")


if __name__ == "__main__":
    main()
