"""Standalone teaser scatter: Hessian rank vs functional (restoration) rank.

Plots the real per-sub-layer measurements from
sensitivity_ranking_comparison.json. The six layers a Hessian-guided allocator
would protect (HESS-6) and the six the restoration probe selects (REST-6) are
highlighted so the inversion is legible without axis ticks.

Output: figures/fig_teaser_scatter.{png,pdf} on an opaque white ground, so the
panel can be pasted straight over the old one without the teaser's own text
showing through.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

RANKING_PATHS = [
    "sensitivity_ranking_comparison.json",
    "results/llama/sensitivity_ranking_comparison.json",
    "results/sensitivity_ranking_comparison.json",
    os.path.expanduser("~/Downloads/sensitivity_ranking_comparison.json"),
]

REST_6 = {("attention", 13), ("attention", 14),
          ("mlp", 1), ("mlp", 6), ("mlp", 7), ("mlp", 31)}
HESS_6 = {("attention", l) for l in (22, 24, 25, 27, 28, 30)}

COL_ATTN = "#E8934A"      # orange circles
COL_MLP = "#9575CD"       # purple squares
COL_REST = "#1B5E20"      # dark green stars
COL_HESS = "#8B1A1A"      # dark red circles
COL_LINE = "#C62828"      # dashed trend + rho label
COL_BOX = "#1A1A1A"


def load_points():
    for path in RANKING_PATHS:
        if os.path.exists(path):
            with open(path) as f:
                per_layer = json.load(f)["per_layer"]
            pts = []
            for v in per_layer.values():
                pts.append({
                    "h": v["hessian_rank"],
                    "r": v["restoration_rank"],
                    "comp": v["component"],
                    "layer": v["layer"],
                })
            return pts, path
    raise FileNotFoundError(
        "sensitivity_ranking_comparison.json not found; searched: "
        + ", ".join(RANKING_PATHS)
    )


def main() -> None:
    pts, src = load_points()
    h = np.array([p["h"] for p in pts], dtype=float)
    r = np.array([p["r"] for p in pts], dtype=float)
    rho, pval = spearmanr(h, r)

    is_rest = np.array([(p["comp"], p["layer"]) in REST_6 for p in pts])
    is_hess = np.array([(p["comp"], p["layer"]) in HESS_6 for p in pts])
    is_attn = np.array([p["comp"] == "attention" for p in pts])
    plain = ~(is_rest | is_hess)

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    fig.patch.set_facecolor("white")

    # trend line over all 64 points
    z = np.polyfit(h, r, 1)
    xs = np.array([h.min() - 1, h.max() + 1])
    ax.plot(xs, np.poly1d(z)(xs), "--", color=COL_LINE, lw=2.0, zorder=2)

    ax.scatter(h[plain & is_attn], r[plain & is_attn], s=70, c=COL_ATTN,
               marker="o", edgecolor="none", zorder=3)
    ax.scatter(h[plain & ~is_attn], r[plain & ~is_attn], s=62, c=COL_MLP,
               marker="s", edgecolor="none", zorder=3)
    ax.scatter(h[is_hess], r[is_hess], s=115, c=COL_HESS, marker="o",
               edgecolor="#4A0D0D", linewidth=1.1, zorder=4)
    ax.scatter(h[is_rest], r[is_rest], s=230, c=COL_REST, marker="*",
               edgecolor="#0B2D0E", linewidth=0.9, zorder=5)

    ax.set_xlim(0, 66)
    ax.set_ylim(0, 66)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(COL_BOX)
        spine.set_linewidth(1.6)
    ax.set_facecolor("white")

    # The legend sits below the axes: with the real measurements, L13/L14
    # attention occupy the upper-right corner where an inset legend would go.
    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor=COL_ATTN,
               markeredgecolor="none", markersize=10, label="Attention"),
        Line2D([], [], marker="s", color="none", markerfacecolor=COL_MLP,
               markeredgecolor="none", markersize=9, label="MLP"),
        Line2D([], [], marker="*", color="none", markerfacecolor=COL_REST,
               markeredgecolor="none", markersize=16, label="REST-6 (ours)"),
        Line2D([], [], marker="o", color="none", markerfacecolor=COL_HESS,
               markeredgecolor="none", markersize=11, label="HESS-6"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.015),
              ncol=4, frameon=False, fontsize=10.5, handletextpad=0.35,
              columnspacing=1.1, borderpad=0.2)

    ax.set_title(
        rf"$\rho = {rho:.3f}$".replace("-", "−"),
        color=COL_LINE, fontsize=19, fontweight="bold", pad=22,
    )
    ax.text(0.5, 1.015, "(Hessian rank vs functional rank)",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=11.5, color="#1A1A1A")

    os.makedirs("figures", exist_ok=True)
    png = os.path.join("figures", "fig_teaser_scatter.png")
    pdf = os.path.join("figures", "fig_teaser_scatter.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white",
                edgecolor="none", pad_inches=0.12)
    fig.savefig(pdf, bbox_inches="tight", facecolor="white",
                edgecolor="none", pad_inches=0.12)
    plt.close(fig)

    print(f"  source : {src}")
    print(f"  rho    : {rho:.4f}  (p={pval:.2g}, n={len(pts)})")
    print(f"  REST-6 : {int(is_rest.sum())} highlighted  |  HESS-6: {int(is_hess.sum())}")
    print(f"  saved  : {png}")
    print(f"  saved  : {pdf}")


if __name__ == "__main__":
    main()
