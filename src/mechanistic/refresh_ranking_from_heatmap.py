"""Refresh the restoration side of sensitivity_ranking_comparison.json.

The Hessian trace and SQNR measurements are properties of the model and the
quantizer, so they are unaffected by how MATH generations were scored. Only the
restoration recovery rates change when the sweep is re-run. This recomputes the
restoration fields, ranks, divergence scores and correlations from a refreshed
heatmap without touching the Hessian/SQNR columns and without needing a GPU.

Usage:
    python refresh_ranking_from_heatmap.py \
        --heatmap ~/Downloads/layer_sensitivity_heatmap.json \
        --ranking ~/Downloads/sensitivity_ranking_comparison.json
"""

import argparse
import json
import os
from typing import Dict, List

from scipy.stats import spearmanr, pearsonr


def load_restoration(heatmap_path: str):
    with open(heatmap_path) as f:
        data = json.load(f)
    ls = data["layer_sensitivity"]
    datasets = [ds for ds in ("gsm8k", "arc", "math") if ds in ls["attention"]]
    restoration: Dict[str, Dict[str, float]] = {}
    for comp in ("attention", "mlp"):
        for idx in range(32):
            rates = {ds: ls[comp][ds].get(str(idx), 0.0) for ds in datasets}
            entry = dict(rates)
            entry["aggregate"] = sum(rates.values()) / len(rates) if rates else 0.0
            restoration[f"L{idx}_{comp}"] = entry
    return restoration, datasets


def corr_block(xs: List[float], ys: List[float]) -> Dict[str, float]:
    rho, rp = spearmanr(xs, ys)
    r, pp = pearsonr(xs, ys)
    return {
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(f"{rp:.2g}"),
        "pearson_r": round(float(r), 4),
        "pearson_p": float(f"{pp:.2g}"),
        "n": len(xs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heatmap", required=True)
    ap.add_argument("--ranking", required=True)
    ap.add_argument("--out", default=None, help="defaults to overwriting --ranking")
    args = ap.parse_args()

    heatmap = os.path.expanduser(args.heatmap)
    ranking = os.path.expanduser(args.ranking)
    out = os.path.expanduser(args.out) if args.out else ranking

    restoration, datasets = load_restoration(heatmap)
    with open(ranking) as f:
        rank_doc = json.load(f)
    per_layer = rank_doc["per_layer"]

    keys = sorted(per_layer.keys())
    missing = [k for k in keys if k not in restoration]
    if missing:
        raise KeyError(f"heatmap is missing {len(missing)} sub-layers, e.g. {missing[:3]}")

    # --- refresh restoration fields -----------------------------------------
    for k in keys:
        per_layer[k]["restoration_recovery"] = {ds: restoration[k][ds] for ds in datasets}
        per_layer[k]["restoration_aggregate"] = restoration[k]["aggregate"]

    # ranks: ascending aggregate, so rank 1 = least functionally important
    order = sorted(keys, key=lambda k: per_layer[k]["restoration_aggregate"])
    for rank, k in enumerate(order, start=1):
        per_layer[k]["restoration_rank"] = rank

    # divergence: normalised restoration minus normalised Hessian
    agg = {k: per_layer[k]["restoration_aggregate"] for k in keys}
    hess = {k: per_layer[k]["hessian_trace_normalized"] for k in keys}
    max_restore = max(agg.values()) or 1.0
    min_h, max_h = min(hess.values()), max(hess.values())
    h_range = (max_h - min_h) or 1.0
    for k in keys:
        per_layer[k]["divergence_score"] = round(
            agg[k] / max_restore - (hess[k] - min_h) / h_range, 4
        )

    # --- recompute correlations ---------------------------------------------
    h = [hess[k] for k in keys]
    s = [per_layer[k]["sqnr_db"] for k in keys]
    cors = rank_doc.setdefault("correlations", {})
    for ds in ["aggregate"] + datasets:
        vals = [
            per_layer[k]["restoration_aggregate"] if ds == "aggregate"
            else per_layer[k]["restoration_recovery"][ds]
            for k in keys
        ]
        for other, name in ((h, "hessian"), (s, "sqnr")):
            key = f"restoration_{ds}_vs_{name}"
            block = corr_block(vals, other)
            if key in cors and "interpretation" in cors[key]:
                block["interpretation"] = cors[key]["interpretation"]
            cors[key] = block

    # --- refresh the divergence summaries -----------------------------------
    div = {k: per_layer[k]["divergence_score"] for k in keys}
    for field, reverse in (("top_positive_divergence", True),
                           ("top_negative_divergence", False)):
        if field in rank_doc:
            picked = sorted(keys, key=lambda k: div[k], reverse=reverse)[:10]
            rank_doc[field] = [
                {
                    "layer_key": k,
                    "restoration_aggregate": per_layer[k]["restoration_aggregate"],
                    "restoration_recovery": per_layer[k]["restoration_recovery"],
                    "hessian_trace_normalized": per_layer[k]["hessian_trace_normalized"],
                    "divergence_score": per_layer[k]["divergence_score"],
                }
                for k in picked
            ]

    if "critical_layer_divergence" in rank_doc:
        for k in list(rank_doc["critical_layer_divergence"].keys()):
            if k in per_layer:
                rank_doc["critical_layer_divergence"][k] = {
                    "restoration_aggregate": per_layer[k]["restoration_aggregate"],
                    "restoration_recovery": per_layer[k]["restoration_recovery"],
                    "restoration_rank": per_layer[k]["restoration_rank"],
                    "hessian_rank": per_layer[k]["hessian_rank"],
                    "divergence_score": per_layer[k]["divergence_score"],
                }

    agg_rho = cors["restoration_aggregate_vs_hessian"]["spearman_rho"]
    agg_p = cors["restoration_aggregate_vs_hessian"]["spearman_p"]
    rank_doc["summary"] = [
        f"Aggregate restoration vs Hessian: Spearman rho={agg_rho} (p={agg_p})",
        "Restoration source: " + os.path.basename(heatmap),
    ]
    rank_doc.setdefault("provenance", {})["restoration_refreshed_from"] = os.path.basename(heatmap)

    with open(out, "w") as f:
        json.dump(rank_doc, f, indent=2)

    print(f"Refreshed {len(keys)} sub-layers -> {out}")
    for ds in ["aggregate"] + datasets:
        b = cors[f"restoration_{ds}_vs_hessian"]
        print(f"  rho_{ds:<9} vs Hessian = {b['spearman_rho']:+.4f}  (p={b['spearman_p']:.2g})")
    b = cors["restoration_aggregate_vs_sqnr"]
    print(f"  rho_aggregate vs SQNR    = {b['spearman_rho']:+.4f}  (p={b['spearman_p']:.2g})")


if __name__ == "__main__":
    main()
