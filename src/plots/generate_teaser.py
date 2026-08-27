"""Generate the Figure 1 hero/teaser.

Three-panel narrative:

  Panel 1 - The Problem
    A real Llama-3.1-8B prompt that NF4 quantization breaks.
    Motivates the question: "How do we recover this without paying FP16?"

  Panel 2 - The Diagnostic (centre, blue background)
    Restoration probe finds the inversion: Hessian sensitivity does not
    predict functional importance (rho = -0.537).
    Layer stack on the left, key correlation in the centre, punchline
    at the bottom.

  Panel 3 - The Outcome
    Two stacked boxes comparing equal 6/32 FP16 budgets.
    Red box (HESS-6, current methods): picks dead-zone layers, 2.6%.
    Green box (REST-6, ours): picks functional layers, 52.6%.
    Pill tags characterise each outcome.

Output: figures/fig0_teaser.{png,pdf}
"""

from __future__ import annotations

import os

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

RANKING_PATHS = [
    "sensitivity_ranking_comparison.json",
    "results/llama/sensitivity_ranking_comparison.json",
    "results/sensitivity_ranking_comparison.json",
    os.path.expanduser("~/Downloads/sensitivity_ranking_comparison.json"),
]


def load_rank_pairs():
    """Return (hessian_rank, restoration_rank, is_attention) for all 64 sub-layers.

    The teaser scatter plots the real measurements, not an illustration; if the
    ranking file cannot be found we raise rather than fall back to synthetic
    points, so the figure can never silently misrepresent the data.
    """
    for path in RANKING_PATHS:
        if os.path.exists(path):
            with open(path) as f:
                per_layer = json.load(f)["per_layer"]
            h, r, attn = [], [], []
            for key, v in per_layer.items():
                h.append(v["hessian_rank"])
                r.append(v["restoration_rank"])
                attn.append(v["component"] == "attention")
            return np.array(h), np.array(r), np.array(attn)
    raise FileNotFoundError(
        "sensitivity_ranking_comparison.json not found; searched: "
        + ", ".join(RANKING_PATHS)
    )


# ---------------------------------------------------------------------------
# Colour palette (Okabe-Ito plus accent neutrals)
# ---------------------------------------------------------------------------

COL_HESS_FILL = "#FFE5DC"      # very light red (HESS-6 panel fill)
COL_HESS_EDGE = "#D55E00"      # vermillion (HESS-6 panel edge / glyph)
COL_REST_FILL = "#DDF1E7"      # very light green (REST-6 panel fill)
COL_REST_EDGE = "#009E73"      # bluish green
COL_DIAG_FILL = "#E3EEF8"      # light blue (diagnostic panel)
COL_DIAG_EDGE = "#3D6FA8"      # steel blue
COL_PROBLEM_FILL = "#F7F1E1"   # warm cream (problem panel)
COL_PROBLEM_EDGE = "#7A6A3E"   # warm brown
COL_INACTIVE = "#DCDCDC"
COL_TEXT = "#222222"
COL_MUTED = "#555555"
COL_X = "#B00020"
COL_CHECK = "#1E7A3D"


# ---------------------------------------------------------------------------
# Layer selections
# ---------------------------------------------------------------------------

HESS_6_ATTN = [22, 24, 25, 27, 28, 30]
REST_6_ATTN = [13, 14]
REST_6_MLP = [1, 6, 7, 31]
NUM_LAYERS = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def panel_box(ax, x, y, w, h, *, fill, edge, lw=1.4):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.18",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=1,
    ))


def panel_title_pill(ax, x_center, y, text, *, fg="white", bg, w=2.2, h=0.42):
    ax.add_patch(FancyBboxPatch(
        (x_center - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.20",
        facecolor=bg, edgecolor="none", zorder=3,
    ))
    ax.text(x_center, y, text, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=fg, zorder=4)


def tag_pill(ax, x, y, text, *, color, w=None, fontsize=7.5):
    """Small outlined pill used for outcome tags."""
    if w is None:
        w = 0.08 * len(text) + 0.25
    h = 0.30
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.13",
        facecolor="white", edgecolor=color, linewidth=1.0, zorder=3,
    ))
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=color, zorder=4, fontweight="bold")
    return w


def arrow_down(ax, x, y_top, y_bot, *, color="#444444", lw=1.4):
    ax.annotate("", xy=(x, y_bot), xytext=(x, y_top),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=lw, mutation_scale=12), zorder=5)


def llm_glyph(ax, x_center, y_center, *, label="LLM"):
    """Stylised LLM chip - a rounded rectangle with the label inside."""
    w, h = 1.4, 0.65
    ax.add_patch(FancyBboxPatch(
        (x_center - w / 2, y_center - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor="#222F3E", edgecolor="#0F1822", linewidth=1.2, zorder=3,
    ))
    # Decorative pins on the sides
    for sign in (-1, 1):
        for dy in (-0.18, 0.0, 0.18):
            ax.add_patch(Rectangle(
                (x_center + sign * (w / 2) - 0.02 + (0.0 if sign < 0 else 0.04),
                 y_center + dy - 0.025),
                0.06, 0.05,
                facecolor="#999999", edgecolor="none", zorder=2,
            ))
    ax.text(x_center, y_center, label, ha="center", va="center",
            fontsize=10, fontweight="bold", color="white", zorder=4)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def draw_problem_panel(ax, x0, y0, w, h):
    panel_box(ax, x0, y0, w, h, fill=COL_PROBLEM_FILL, edge=COL_PROBLEM_EDGE)
    cx = x0 + w / 2

    # Panel title
    panel_title_pill(ax, cx, y0 + h - 0.30,
                     "The Problem", bg=COL_PROBLEM_EDGE, w=1.8)

    # Question box (speech bubble effect)
    q_top = y0 + h - 0.80
    q_h = 1.55
    q_w = w - 0.50
    qx = x0 + 0.25
    ax.add_patch(FancyBboxPatch(
        (qx, q_top - q_h), q_w, q_h,
        boxstyle="round,pad=0.03,rounding_size=0.10",
        facecolor="white", edgecolor="#999999", linewidth=0.8, zorder=2,
    ))
    ax.text(qx + 0.14, q_top - 0.18, "ARC-Challenge",
            ha="left", va="top", fontsize=7.5, color=COL_MUTED, style="italic")
    ax.text(qx + 0.14, q_top - 0.42,
            "Q: Which property of a\nmineral can be determined\njust by looking at it?",
            ha="left", va="top", fontsize=8.0, color=COL_TEXT)
    ax.text(qx + 0.14, q_top - 1.22,
            "(A) luster  (B) mass\n(C) weight  (D) hardness",
            ha="left", va="top", fontsize=7.5, color=COL_MUTED)

    # LLM glyph
    llm_y = q_top - q_h - 0.55
    llm_glyph(ax, cx, llm_y, label="Llama-3.1-8B")

    # Subtitle under glyph
    ax.text(cx, llm_y - 0.45, "NF4 4-bit quantization",
            ha="center", va="center", fontsize=7.5, color=COL_MUTED,
            style="italic")

    # Arrow + wrong answer
    arrow_down(ax, cx, llm_y - 0.65, llm_y - 0.95, color=COL_X, lw=1.8)
    ans_y = llm_y - 1.35
    ans_w, ans_h = w - 0.90, 0.50
    ax.add_patch(FancyBboxPatch(
        (cx - ans_w / 2, ans_y - ans_h / 2), ans_w, ans_h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=COL_HESS_FILL, edgecolor=COL_X, linewidth=1.1, zorder=3,
    ))
    ax.text(cx - ans_w / 2 + 0.15, ans_y, r"$\times$",
            ha="left", va="center", fontsize=14, fontweight="bold",
            color=COL_X, zorder=4)
    ax.text(cx + 0.10, ans_y, "Answer: B",
            ha="left", va="center", fontsize=10.5, fontweight="bold",
            color=COL_X, zorder=4)

    # Punchline at bottom of panel
    ax.text(cx, y0 + 0.35,
            "How do we recover this\nwithout paying FP16's cost?",
            ha="center", va="center", fontsize=8.0,
            color=COL_TEXT, style="italic", fontweight="bold")


def draw_diagnostic_panel(ax, x0, y0, w, h):
    panel_box(ax, x0, y0, w, h, fill=COL_DIAG_FILL, edge=COL_DIAG_EDGE)
    cx = x0 + w / 2

    panel_title_pill(ax, cx, y0 + h - 0.30,
                     "Restoration Probe", bg=COL_DIAG_EDGE, w=2.6)

    ax.text(cx, y0 + h - 0.70,
            "Which layers are functionally critical at 4-bit?",
            ha="center", va="center", fontsize=8.0, color=COL_MUTED,
            style="italic")

    # ----- Inner content split into two columns -----
    inner_top = y0 + h - 1.00
    inner_bot = y0 + 1.40

    # --- Left half: small 32-layer recovery profile -----------------------
    left_x0 = x0 + 0.35
    left_x1 = cx - 0.15
    cell_h = (inner_top - inner_bot) / NUM_LAYERS
    cell_w_attn = (left_x1 - left_x0) * 0.45
    cell_w_mlp = (left_x1 - left_x0) * 0.45
    gap_col = (left_x1 - left_x0) * 0.10

    # Header labels
    ax.text(left_x0 + cell_w_attn / 2, inner_top + 0.10,
            "Attn", ha="center", va="bottom", fontsize=7,
            color=COL_MUTED, fontweight="bold")
    ax.text(left_x0 + cell_w_attn + gap_col + cell_w_mlp / 2,
            inner_top + 0.10,
            "MLP", ha="center", va="bottom", fontsize=7,
            color=COL_MUTED, fontweight="bold")
    ax.text(left_x0 - 0.18, inner_top + 0.10,
            "Layer", ha="right", va="bottom", fontsize=6.5,
            color=COL_MUTED)

    # Pre-defined "recovery" values per layer (qualitative encoding of the
    # paper's findings; the figure caption ties exact numbers to the data).
    attn_glow = {13: 0.85, 14: 0.78}
    mlp_glow = {1: 0.55, 6: 0.62, 7: 0.70, 31: 0.50}

    for layer in range(NUM_LAYERS):
        y = inner_bot + (NUM_LAYERS - 1 - layer) * cell_h
        # Attention cell
        ax.add_patch(Rectangle(
            (left_x0, y), cell_w_attn, cell_h * 0.85,
            facecolor=_recovery_color(attn_glow.get(layer, 0.0)),
            edgecolor="#AAAAAA", linewidth=0.3, zorder=2,
        ))
        # MLP cell
        ax.add_patch(Rectangle(
            (left_x0 + cell_w_attn + gap_col, y),
            cell_w_mlp, cell_h * 0.85,
            facecolor=_recovery_color(mlp_glow.get(layer, 0.0)),
            edgecolor="#AAAAAA", linewidth=0.3, zorder=2,
        ))
        # Layer index every 4 layers
        if layer % 8 == 0 or layer == NUM_LAYERS - 1:
            ax.text(left_x0 - 0.10, y + cell_h * 0.42,
                    f"L{layer}", ha="right", va="center",
                    fontsize=6.0, color=COL_MUTED)

    # Annotate the key glowing cells
    ax.annotate("L13, L14\nattn",
                xy=(left_x0 + cell_w_attn,
                    inner_bot + (NUM_LAYERS - 1 - 13.5) * cell_h
                    + cell_h * 0.4),
                xytext=(left_x0 + cell_w_attn + gap_col + cell_w_mlp + 0.30,
                        inner_bot + (NUM_LAYERS - 1 - 13.5) * cell_h
                        + cell_h * 0.4),
                ha="left", va="center", fontsize=7.0, color=COL_CHECK,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COL_CHECK, lw=0.7))
    ax.annotate("L1, L6, L7,\nL31 mlp",
                xy=(left_x0 + cell_w_attn + gap_col,
                    inner_bot + (NUM_LAYERS - 1 - 6) * cell_h
                    + cell_h * 0.4),
                xytext=(left_x0 - 0.55,
                        inner_bot + (NUM_LAYERS - 1 - 21) * cell_h),
                ha="right", va="center", fontsize=7.0, color=COL_CHECK,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COL_CHECK, lw=0.7))

    # --- Right half: rho scatter + punchline -----------------------------
    right_x0 = cx + 0.15
    right_x1 = x0 + w - 0.30

    # Small scatter showing the inversion
    scatter_h = 1.85
    scatter_y_top = inner_top - 0.05
    scatter_y_bot = scatter_y_top - scatter_h

    sx0, sx1 = right_x0 + 0.10, right_x1 - 0.10
    # Scatter axes drawn as plain box
    ax.add_patch(Rectangle(
        (sx0, scatter_y_bot), sx1 - sx0, scatter_h,
        facecolor="white", edgecolor="#888888", linewidth=0.7, zorder=2,
    ))

    h_rank, r_rank, is_attn = load_rank_pairs()
    n = len(h_rank)
    # normalise both rank axes to [0, 1] for placement inside the panel box
    xs = (h_rank - h_rank.min()) / (h_rank.max() - h_rank.min())
    ys = (r_rank - r_rank.min()) / (r_rank.max() - r_rank.min())

    pxs = sx0 + xs * (sx1 - sx0)
    pys = scatter_y_bot + ys * scatter_h
    ax.scatter(pxs[is_attn], pys[is_attn], c=COL_HESS_EDGE, s=10,
               alpha=0.7, edgecolor="none", zorder=3)
    ax.scatter(pxs[~is_attn], pys[~is_attn], c="#7E57C2", s=10,
               alpha=0.7, edgecolor="none", marker="s", zorder=3)

    z = np.polyfit(xs, ys, 1)
    line_x = np.array([0, 1])
    line_y = np.poly1d(z)(line_x)
    ax.plot(sx0 + line_x * (sx1 - sx0),
            scatter_y_bot + line_y * scatter_h,
            "--", color=COL_X, lw=1.1, zorder=4)

    # Scatter labels
    ax.text((sx0 + sx1) / 2, scatter_y_top + 0.10,
            r"$\rho = -0.537$    (Hessian rank vs functional rank)",
            ha="center", va="bottom", fontsize=8.0, fontweight="bold",
            color=COL_X)
    ax.text((sx0 + sx1) / 2, scatter_y_bot - 0.22,
            "Hessian-prioritised layers (red, bottom-right) recover 0%.\n"
            "Functionally-critical layers (top-left) are deprioritised by GPTQ.",
            ha="center", va="top", fontsize=7.0, color=COL_TEXT)

    # Punchline at panel bottom
    pl_y = y0 + 0.55
    pl_w = w - 0.60
    ax.add_patch(FancyBboxPatch(
        (cx - pl_w / 2, pl_y - 0.42), pl_w, 0.84,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor="white", edgecolor=COL_DIAG_EDGE, linewidth=1.0,
        zorder=3,
    ))
    ax.text(cx, pl_y + 0.12,
            "Numerical sensitivity does NOT predict functional importance.",
            ha="center", va="center", fontsize=8.5,
            color=COL_DIAG_EDGE, fontweight="bold", zorder=4)
    ax.text(cx, pl_y - 0.20,
            "A 6/32 FP16 budget chosen by Hessian rank protects the wrong layers.",
            ha="center", va="center", fontsize=7.5,
            color=COL_TEXT, style="italic", zorder=4)


def _recovery_color(strength):
    """Map [0,1] recovery strength to a green-on-grey colour."""
    if strength < 0.02:
        return COL_INACTIVE
    # Blend white -> COL_REST_EDGE proportional to strength
    r = int(220 + (0 - 220) * strength)
    g = int(220 + (158 - 220) * strength)
    b = int(220 + (115 - 220) * strength)
    return f"#{max(r,0):02X}{max(g,0):02X}{max(b,0):02X}"


def draw_outcome_panel(ax, x0, y0, w, h):
    """Two stacked outcome cards (HESS-6 / REST-6)."""
    gap = 0.20
    card_h = (h - gap) / 2
    panel_title_pill(ax, x0 + w / 2, y0 + h + 0.30,
                     "Equal Budget, Different Outcomes",
                     bg=COL_TEXT, w=4.8)

    # ----- Top card: HESS-6 (red, bad) -----
    top_y = y0 + card_h + gap
    _draw_outcome_card(
        ax, x0, top_y, w, card_h,
        title="HESS-6  (current methods)",
        subtitle="Hessian-guided  /  GPTQ, OmniQuant, ResQ",
        layers_text="Protects: L22, L24, L25, L27, L28, L30 attention",
        layers_note="(all in the L15-L31 ARC dead zone)",
        result_pct="2.6%",
        result_caption=r"ARC recovery   (ns vs.\ NF4, $p{=}0.125$)",
        tags=[("Wrong layers", COL_HESS_EDGE),
              ("Wasted budget", COL_HESS_EDGE),
              ("No improvement", COL_HESS_EDGE)],
        fill=COL_HESS_FILL, edge=COL_HESS_EDGE,
        verdict_mark=r"$\times$",
    )

    # ----- "Vs." separator -----
    ax.text(x0 + w / 2, y0 + card_h + gap / 2, "vs.",
            ha="center", va="center", fontsize=10,
            color=COL_MUTED, fontweight="bold", style="italic")

    # ----- Bottom card: REST-6 (green, good) -----
    _draw_outcome_card(
        ax, x0, y0, w, card_h,
        title="REST-6  (ours)",
        subtitle="Restoration-guided  /  functional probing",
        layers_text="Protects: L13, L14 attention  +  L1, L6, L7, L31 MLP",
        layers_note="(spread across early-to-mid layers)",
        result_pct="52.6%",
        result_caption=r"ARC recovery   ($p{=}0.001$ vs.\ HESS-6)",
        tags=[(r"$20\times$ recovery", COL_REST_EDGE),
              ("Same budget", COL_REST_EDGE),
              (r"$\equiv$ FP16", COL_REST_EDGE)],
        fill=COL_REST_FILL, edge=COL_REST_EDGE,
        verdict_mark=r"$\checkmark$",
    )


def _draw_outcome_card(ax, x0, y0, w, h, *,
                       title, subtitle, layers_text, layers_note,
                       result_pct, result_caption, tags, fill, edge,
                       verdict_mark):
    panel_box(ax, x0, y0, w, h, fill=fill, edge=edge, lw=1.5)

    # Title row: bold title + verdict mark
    ax.text(x0 + 0.20, y0 + h - 0.30, title,
            ha="left", va="center", fontsize=10.5,
            fontweight="bold", color=edge)
    ax.text(x0 + w - 0.25, y0 + h - 0.30, verdict_mark,
            ha="right", va="center", fontsize=15,
            fontweight="bold", color=edge)

    # Subtitle
    ax.text(x0 + 0.20, y0 + h - 0.62, subtitle,
            ha="left", va="center", fontsize=7.5,
            color=COL_MUTED, style="italic")

    # Layer list
    ax.text(x0 + 0.20, y0 + h - 1.00, layers_text,
            ha="left", va="center", fontsize=8.5, color=COL_TEXT)
    ax.text(x0 + 0.20, y0 + h - 1.28, layers_note,
            ha="left", va="center", fontsize=7.5, color=COL_MUTED,
            style="italic")

    # Big result number on the right side
    res_x = x0 + w - 1.10
    ax.text(res_x, y0 + h - 1.08, result_pct,
            ha="center", va="center", fontsize=26,
            fontweight="bold", color=edge)
    ax.text(res_x, y0 + h - 1.55, result_caption,
            ha="center", va="center", fontsize=6.5,
            color=COL_TEXT, style="italic")

    # Tag pills along the bottom
    tag_y = y0 + 0.30
    n_tags = len(tags)
    avail = w - 0.40
    slot = avail / n_tags
    for i, (text, color) in enumerate(tags):
        x_center = x0 + 0.20 + slot * (i + 0.5)
        tag_pill(ax, x_center, tag_y, text, color=color, fontsize=7.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["mathtext.fontset"] = "stix"

    fig = plt.figure(figsize=(13.0, 6.5))
    ax = fig.add_axes([0.015, 0.02, 0.97, 0.85])
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    # Outer dashed container
    ax.add_patch(FancyBboxPatch(
        (0.10, 0.10), 14.30, 6.45,
        boxstyle="round,pad=0.02,rounding_size=0.20",
        facecolor="none", edgecolor="#777777", linewidth=1.0,
        linestyle=(0, (4, 3)), zorder=0,
    ))

    # Headline
    fig.text(0.50, 0.965,
             "Hessian sensitivity ranks the layers backwards",
             ha="center", va="top",
             fontsize=15.5, fontweight="bold", color=COL_TEXT)
    fig.text(0.50, 0.925,
             "Same compute budget, wrong layers protected, "
             r"$20\times$ less damage recovered.",
             ha="center", va="top", fontsize=10.0,
             color=COL_MUTED, style="italic")

    # Three panel slots (left = problem, middle = diagnostic, right = outcome)
    # Outcome panel needs a bit more space for two stacked cards.
    p1_x, p1_w = 0.40, 2.95
    p2_x, p2_w = 3.55, 5.30
    p3_x, p3_w = 9.05, 5.20

    panel_y, panel_h = 0.65, 5.45

    # Diagnostic panel title sits on top, so its drawing height starts a
    # bit lower than the others.
    draw_problem_panel(ax, p1_x, panel_y, p1_w, panel_h)
    draw_diagnostic_panel(ax, p2_x, panel_y, p2_w, panel_h)
    draw_outcome_panel(ax, p3_x, panel_y, p3_w, panel_h - 0.40)

    # Connector arrows between panels (left -> middle, middle -> right)
    arrow_y = panel_y + panel_h / 2
    ax.annotate("",
                xy=(p2_x + 0.05, arrow_y),
                xytext=(p1_x + p1_w - 0.05, arrow_y),
                arrowprops=dict(arrowstyle="->", color="#444444", lw=1.4,
                                mutation_scale=14), zorder=5)
    ax.annotate("",
                xy=(p3_x + 0.05, arrow_y),
                xytext=(p2_x + p2_w - 0.05, arrow_y),
                arrowprops=dict(arrowstyle="->", color="#444444", lw=1.4,
                                mutation_scale=14), zorder=5)

    # Output
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, "fig0_teaser.png")
    pdf = os.path.join(out_dir, "fig0_teaser.pdf")
    fig.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {png}")
    print(f"  Saved: {pdf}")


if __name__ == "__main__":
    main()
