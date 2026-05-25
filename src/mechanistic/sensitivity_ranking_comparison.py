"""
Phase 21: Restoration vs Hessian Sensitivity Ranking Comparison
================================================================

DIFFERENTIATOR EXPERIMENT vs ResQ / OmniQuant.

Tests whether our restoration-based layer ranking (Phase 9) captures
*functional* structure beyond pure numerical/activation sensitivity.

Two competing hypotheses
-------------------------
H0 (Tautology): Our restoration ranking is just a proxy for Hessian
  sensitivity. Layers that are "functionally important" are simply the
  ones whose quantization error propagates most to the output — the
  same signal that GPTQ / ResQ use for mixed-precision assignment.
  Prediction: high Spearman ρ between restoration ranking and Hessian.

H1 (Functional Structure): Our probe captures task-selective computation.
  Some layers may have large Hessian sensitivity (error amplified by large
  activations) but low restoration recovery (quantization-resistant function),
  and vice versa. Prediction: low Spearman ρ — rankings diverge.

Hessian proxy (standard GPTQ / ResQ approach)
----------------------------------------------
For each sub-layer l with weight matrix W ∈ R^{out×in}:
  H_l  = E_x[x_l x_l^T]          (activation covariance, input to layer l)
  S_l  = Tr(H_l) / d              (normalized trace = mean squared activation)

This is the exact sensitivity proxy used by:
  - GPTQ (Frantar et al. 2022)
  - OmniQuant (Shao et al. 2023)
  - ResQ (Markov et al. 2024) — uses top PCA directions of H_l

We record input activations to each sub-layer (attention block, MLP block)
across a calibration corpus (128 sequences × 2048 tokens from WikiText-2),
compute S_l per sub-layer, and compare against our restoration ranking
using Spearman rank correlation.

Restoration ranking
-------------------
From Phase 9 (layer_sensitivity_heatmap.json):
  R_l = fraction of flipped examples recovered when layer l FP16 weights
        are transplanted into the 4-bit model.
  We use per-dataset and aggregate rankings.

Output: results/sensitivity_ranking_comparison.json

Authentication
--------------
Set HF_TOKEN environment variable for model access.
"""

import json
import os
import gc
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise ImportError("transformers required: pip install transformers")

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("datasets required: pip install datasets")

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration for sensitivity ranking comparison."""
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir: str = "results"
    output_file: str = os.path.join("results", "sensitivity_ranking_comparison.json")

    # Path to Phase 9 restoration results
    heatmap_file: str = "layer_sensitivity_heatmap.json"
    # Path to Phase 18b SQNR results (optional secondary comparison)
    sqnr_file: str = os.path.join("results", "quantization_error_analysis.json")

    num_layers: int = 32
    # Calibration corpus (standard GPTQ protocol: 128 seqs × 2048 tokens)
    num_calibration_seqs: int = 128
    seq_len: int = 2048

    # Load model in FP16 for accurate activations
    torch_dtype: torch.dtype = torch.float16


# =============================================================================
# Restoration ranking (Phase 9 data)
# =============================================================================

def load_restoration_ranking(
    heatmap_path: str,
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Load per-layer restoration recovery rates from Phase 9 heatmap.

    Returns dict: {layer_key -> {dataset -> recovery_rate, "aggregate" -> mean}}
    layer_key format: "L{idx}_{component}"  e.g. "L13_attention", "L6_mlp"
    """
    with open(heatmap_path) as f:
        data = json.load(f)

    layer_sensitivity = data["layer_sensitivity"]
    datasets = list(layer_sensitivity["attention"].keys())  # e.g. ["gsm8k", "arc", "math"]

    restoration: Dict[str, Dict[str, float]] = {}

    for component in ("attention", "mlp"):
        for layer_idx in range(32):
            key = f"L{layer_idx}_{component}"
            entry: Dict[str, float] = {}
            rates = []
            for ds in datasets:
                rate = layer_sensitivity[component][ds].get(str(layer_idx), 0.0)
                entry[ds] = rate
                rates.append(rate)
            entry["aggregate"] = sum(rates) / len(rates) if rates else 0.0
            restoration[key] = entry

    return restoration, datasets


# =============================================================================
# Hessian proxy: activation covariance trace
# =============================================================================

def collect_activation_stats(
    model: nn.Module,
    tokenizer,
    config: Config,
) -> Dict[str, Dict[str, float]]:
    """Collect per-sub-layer input activation statistics via forward hooks.

    For each sub-layer (attention block input, MLP block input), records:
      - mean_sq_norm: Tr(H) / d  =  E[||x||^2] / d  (Hessian sensitivity proxy)
      - mean_norm: E[||x||] / sqrt(d)  (RMS activation magnitude)
      - max_norm: max ||x|| across calibration corpus
      - n_samples: total number of token positions observed

    Follows the same calibration protocol as GPTQ / ResQ: WikiText-2 test
    set, 128 sequences of 2048 tokens each.
    """
    print("\n[Hessian] Setting up calibration corpus (WikiText-2, 128 × 2048)...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    # Slice into calibration sequences
    total_tokens = input_ids.numel()
    seqs = []
    stride = total_tokens // config.num_calibration_seqs
    for i in range(config.num_calibration_seqs):
        start = i * stride
        end = start + config.seq_len
        if end <= total_tokens:
            seqs.append(input_ids[start:end].unsqueeze(0))
    print(f"  {len(seqs)} calibration sequences × {config.seq_len} tokens")

    # -------------------------------------------------------------------------
    # Register hooks on attention block inputs and MLP block inputs
    # -------------------------------------------------------------------------
    # accumulators[key] = {"sum_sq": float, "sum": float, "max": float, "n": int}
    accumulators: Dict[str, Dict] = {}
    hooks = []

    for layer_idx in range(config.num_layers):
        layer = model.model.layers[layer_idx]

        for component in ("attention", "mlp"):
            key = f"L{layer_idx}_{component}"
            accumulators[key] = {
                "sum_sq": 0.0,  # sum of squared L2 norms
                "sum": 0.0,      # sum of L2 norms
                "max_sq": 0.0,   # max squared L2 norm
                "n": 0,          # total token positions
                "d": 0,          # input dimension (filled on first call)
            }

            # We hook the *input* to each sub-module
            target = layer.self_attn if component == "attention" else layer.mlp

            def make_hook(acc_key: str):
                def hook_fn(module, args, kwargs, output):
                    # Positional or keyword input tensor: (batch, seq, d_model)
                    # transformers 5.x may call modules with all-kwargs, leaving
                    # args empty; fall back to the 'hidden_states' kwarg.
                    if args:
                        x = args[0]
                    elif "hidden_states" in kwargs:
                        x = kwargs["hidden_states"]
                    elif "x" in kwargs:
                        x = kwargs["x"]
                    else:
                        return
                    if x is None:
                        return
                    x = x.float()
                    b, s, d = x.shape
                    accumulators[acc_key]["d"] = d
                    flat = x.reshape(-1, d)  # (batch*seq, d)
                    sq_norms = (flat ** 2).sum(dim=-1)  # (batch*seq,)
                    norms = sq_norms.sqrt()
                    accumulators[acc_key]["sum_sq"] += sq_norms.sum().item()
                    accumulators[acc_key]["sum"] += norms.sum().item()
                    cur_max = sq_norms.max().item()
                    if cur_max > accumulators[acc_key]["max_sq"]:
                        accumulators[acc_key]["max_sq"] = cur_max
                    accumulators[acc_key]["n"] += flat.shape[0]
                return hook_fn

            # with_kwargs=True (PyTorch >= 2.0) ensures the hook receives
            # keyword arguments too — needed for transformers 5.x which calls
            # self_attn(hidden_states=...) as kwargs.
            h = target.register_forward_hook(make_hook(key), with_kwargs=True)
            hooks.append(h)

    # -------------------------------------------------------------------------
    # Run forward passes (no grad needed)
    # -------------------------------------------------------------------------
    device = next(model.parameters()).device
    print(f"  Running {len(seqs)} forward passes on {device}...")

    model.eval()
    with torch.no_grad():
        for seq in tqdm(seqs, desc="Calibration"):
            _ = model(seq.to(device))
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Remove hooks
    for h in hooks:
        h.remove()

    # -------------------------------------------------------------------------
    # Compute per-layer Hessian proxy statistics
    # -------------------------------------------------------------------------
    stats: Dict[str, Dict[str, float]] = {}
    for key, acc in accumulators.items():
        n = acc["n"]
        d = acc["d"] if acc["d"] > 0 else 1
        if n == 0:
            stats[key] = {
                "hessian_trace_normalized": 0.0,
                "rms_activation": 0.0,
                "max_activation_norm": 0.0,
                "n_samples": 0,
                "input_dim": d,
            }
            continue

        # Tr(H) / d  =  E[||x||^2] / d
        hessian_trace_norm = acc["sum_sq"] / (n * d)
        # RMS = sqrt(E[||x||^2] / d)
        rms = math.sqrt(hessian_trace_norm) if hessian_trace_norm > 0 else 0.0
        max_norm = math.sqrt(acc["max_sq"])

        stats[key] = {
            "hessian_trace_normalized": round(hessian_trace_norm, 8),
            "rms_activation": round(rms, 6),
            "max_activation_norm": round(max_norm, 4),
            "n_samples": n,
            "input_dim": d,
        }

    return stats


# =============================================================================
# SQNR data (Phase 18b)
# =============================================================================

def load_sqnr_ranking(sqnr_path: str) -> Optional[Dict[str, float]]:
    """Load SQNR-based sensitivity from Phase 18b output.

    Lower SQNR → higher quantization error → higher numerical sensitivity.
    We convert to a sensitivity score = -SQNR (so higher = more sensitive).
    Returns dict: {layer_key -> sqnr_db}
    """
    if not os.path.exists(sqnr_path):
        return None
    with open(sqnr_path) as f:
        data = json.load(f)
    result = {}
    for entry in data.get("error_ranking", []):
        result[entry["key"]] = entry["sqnr_db"]
    return result


# =============================================================================
# Ranking and correlation analysis
# =============================================================================

def compute_rank_vector(values: Dict[str, float], keys: List[str]) -> List[float]:
    """Return rank vector (1=lowest value) for given keys from values dict."""
    present = [(k, values[k]) for k in keys if k in values]
    present.sort(key=lambda x: x[1])
    ranks = {k: i + 1 for i, (k, _) in enumerate(present)}
    return [ranks.get(k, float("nan")) for k in keys]


def spearman_rho(x: List[float], y: List[float]) -> Tuple[float, float, int]:
    """Compute Spearman ρ, p-value, and n (skipping NaN pairs)."""
    pairs = [(xi, yi) for xi, yi in zip(x, y)
             if not (math.isnan(xi) or math.isnan(yi))]
    if len(pairs) < 3:
        return float("nan"), float("nan"), len(pairs)
    xs, ys = zip(*pairs)
    if HAS_SCIPY:
        rho, p = scipy_stats.spearmanr(xs, ys)
        return float(rho), float(p), len(pairs)
    # Fallback: manual Spearman — convert values to ranks first, then apply
    # the standard formula 1 - 6*Σd_i² / (n*(n²-1)) where d_i are rank diffs
    n = len(pairs)

    def _rank(vals: List[float]) -> List[float]:
        sorted_vals = sorted(enumerate(vals), key=lambda t: t[1])
        ranks: List[float] = [0.0] * n
        for rank, (idx, _) in enumerate(sorted_vals, start=1):
            ranks[idx] = float(rank)
        return ranks

    rx = _rank(list(xs))
    ry = _rank(list(ys))
    d2 = sum((ri - rj) ** 2 for ri, rj in zip(rx, ry))
    rho = 1.0 - 6.0 * d2 / (n * (n ** 2 - 1))
    return rho, float("nan"), n


def pearson_r(x: List[float], y: List[float]) -> Tuple[float, float, int]:
    """Compute Pearson r, p-value, and n (skipping NaN pairs)."""
    pairs = [(xi, yi) for xi, yi in zip(x, y)
             if not (math.isnan(xi) or math.isnan(yi))]
    if len(pairs) < 3:
        return float("nan"), float("nan"), len(pairs)
    xs, ys = zip(*pairs)
    if HAS_SCIPY:
        r, p = scipy_stats.pearsonr(xs, ys)
        return float(r), float(p), len(pairs)
    return float("nan"), float("nan"), len(pairs)


def interpret_correlation(rho: float, p: float) -> str:
    """Return qualitative interpretation of a Spearman ρ value."""
    if math.isnan(rho):
        return "Could not compute"
    sig = "significant" if (not math.isnan(p) and p < 0.05) else "not significant"
    if abs(rho) < 0.3:
        strength = "weak"
    elif abs(rho) < 0.6:
        strength = "moderate"
    else:
        strength = "strong"
    direction = "positive" if rho >= 0 else "negative"
    return (
        f"{strength} {direction} correlation (ρ={rho:.3f}, p={p:.4f}, {sig}). "
        + (
            "Rankings DIVERGE — restoration captures functional structure "
            "beyond Hessian sensitivity."
            if abs(rho) < 0.4
            else "Rankings broadly agree — partial Hessian confound."
            if abs(rho) < 0.7
            else "Rankings align — restoration is strongly predicted by Hessian sensitivity."
        )
    )


# =============================================================================
# Main pipeline
# =============================================================================

def run_analysis():
    """Run Phase 21: restoration vs Hessian sensitivity ranking comparison."""
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 70)
    print("PHASE 21: Restoration vs Hessian Sensitivity Ranking Comparison")
    print("=" * 70)

    # =========================================================================
    # Step 1: Load restoration ranking from Phase 9
    # =========================================================================
    print("\n[Step 1] Loading restoration ranking from Phase 9...")
    if not os.path.exists(config.heatmap_file):
        raise FileNotFoundError(
            f"Phase 9 heatmap not found: {config.heatmap_file}\n"
            "Run layer_sensitivity_profiler.py first."
        )
    restoration, datasets = load_restoration_ranking(config.heatmap_file)
    all_keys = sorted(restoration.keys())
    print(f"  Loaded {len(restoration)} sub-layer entries across datasets: {datasets}")

    # Print top-10 restoration layers (aggregate)
    top_restore = sorted(all_keys, key=lambda k: restoration[k]["aggregate"], reverse=True)[:10]
    print("\n  Top-10 layers by aggregate restoration recovery:")
    for k in top_restore:
        print(f"    {k}: aggregate={restoration[k]['aggregate']:.4f}  "
              + "  ".join(f"{ds}={restoration[k][ds]:.3f}" for ds in datasets))

    # =========================================================================
    # Step 2: Collect Hessian proxy via activation hooks
    # =========================================================================
    print("\n[Step 2] Loading FP16 model for activation collection...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=config.torch_dtype,
        device_map="cuda:0",  # load directly to GPU; avoids slow multi-shard staging
    )
    model.eval()

    hessian_stats = collect_activation_stats(model, tokenizer, config)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Print top-10 most Hessian-sensitive layers
    hessian_scores = {k: v["hessian_trace_normalized"] for k, v in hessian_stats.items()}
    top_hessian = sorted(all_keys, key=lambda k: hessian_scores.get(k, 0), reverse=True)[:10]
    print("\n  Top-10 layers by Hessian sensitivity (Tr(H)/d):")
    for k in top_hessian:
        h = hessian_stats.get(k, {})
        print(f"    {k}: Tr(H)/d={h.get('hessian_trace_normalized', 0):.6f}  "
              f"RMS={h.get('rms_activation', 0):.4f}")

    # =========================================================================
    # Step 3: Load SQNR from Phase 18b (optional secondary comparison)
    # =========================================================================
    print("\n[Step 3] Loading SQNR data from Phase 18b...")
    sqnr_data = load_sqnr_ranking(config.sqnr_file)
    if sqnr_data:
        print(f"  Loaded SQNR for {len(sqnr_data)} sub-layers")
    else:
        print(f"  Phase 18b output not found at {config.sqnr_file}; skipping SQNR comparison")

    # =========================================================================
    # Step 4: Rank correlation analysis
    # =========================================================================
    print("\n[Step 4] Computing Spearman rank correlations...")

    # Build unified key list (present in both restoration and hessian_stats)
    common_keys = [k for k in all_keys if k in hessian_stats and hessian_stats[k]["n_samples"] > 0]
    print(f"  {len(common_keys)} sub-layers with both restoration and Hessian data")

    correlations: Dict[str, dict] = {}

    # --- 4a. Restoration vs Hessian (aggregate + per-dataset) ---
    hessian_ranks = [hessian_stats[k]["hessian_trace_normalized"] for k in common_keys]

    for scoring in ["aggregate"] + datasets:
        restore_values = [restoration[k].get(scoring, 0.0) for k in common_keys]
        rho, p, n = spearman_rho(restore_values, hessian_ranks)
        r, rp, _ = pearson_r(restore_values, hessian_ranks)
        label = f"restoration_{scoring}_vs_hessian"
        correlations[label] = {
            "spearman_rho": round(rho, 4) if not math.isnan(rho) else None,
            "spearman_p": round(p, 6) if not math.isnan(p) else None,
            "pearson_r": round(r, 4) if not math.isnan(r) else None,
            "pearson_p": round(rp, 6) if not math.isnan(rp) else None,
            "n": n,
            "interpretation": interpret_correlation(rho, p),
        }
        print(f"  {label}: ρ={rho:.3f} (p={p:.4f}), r={r:.3f}")

    # --- 4b. Restoration vs SQNR (weight-space error) ---
    if sqnr_data:
        sqnr_keys = [k for k in common_keys if k in sqnr_data]
        for scoring in ["aggregate"] + datasets:
            restore_values = [restoration[k].get(scoring, 0.0) for k in sqnr_keys]
            sqnr_values = [-sqnr_data[k] for k in sqnr_keys]  # negate: lower SQNR = more sensitive
            rho, p, n = spearman_rho(restore_values, sqnr_values)
            r, rp, _ = pearson_r(restore_values, sqnr_values)
            label = f"restoration_{scoring}_vs_sqnr"
            correlations[label] = {
                "spearman_rho": round(rho, 4) if not math.isnan(rho) else None,
                "spearman_p": round(p, 6) if not math.isnan(p) else None,
                "pearson_r": round(r, 4) if not math.isnan(r) else None,
                "pearson_p": round(rp, 6) if not math.isnan(rp) else None,
                "n": n,
                "interpretation": interpret_correlation(rho, p),
            }
            print(f"  {label}: ρ={rho:.3f} (p={p:.4f})")

    # --- 4c. Hessian vs SQNR (two numerical sensitivity measures — sanity check) ---
    if sqnr_data:
        hess_check = [hessian_stats[k]["hessian_trace_normalized"] for k in sqnr_keys
                      if k in hessian_stats]
        sqnr_check = [-sqnr_data[k] for k in sqnr_keys if k in hessian_stats]
        rho, p, n = spearman_rho(hess_check, sqnr_check)
        correlations["hessian_vs_sqnr_sanity"] = {
            "spearman_rho": round(rho, 4) if not math.isnan(rho) else None,
            "spearman_p": round(p, 6) if not math.isnan(p) else None,
            "n": n,
            "note": "Sanity check: two numerical sensitivity measures should correlate",
        }
        print(f"  hessian_vs_sqnr_sanity: ρ={rho:.3f} (expected moderate-high positive)")

    # =========================================================================
    # Step 5: Identify "divergence" layers (high restore, low Hessian or vice versa)
    # =========================================================================
    print("\n[Step 5] Identifying structural divergence layers...")

    # Normalize both scales to [0,1] for divergence score
    restore_agg = {k: restoration[k]["aggregate"] for k in common_keys}
    hess_norm = {k: hessian_stats[k]["hessian_trace_normalized"] for k in common_keys}

    max_restore = max(restore_agg.values()) or 1.0
    min_hess = min(hess_norm.values())
    max_hess = max(hess_norm.values()) or 1.0
    hess_range = max_hess - min_hess or 1.0

    divergence_scores = {}
    for k in common_keys:
        r_norm = restore_agg[k] / max_restore
        h_norm = (hess_norm[k] - min_hess) / hess_range
        # Positive divergence: functionally important but numerically robust
        # Negative divergence: numerically sensitive but functionally unimportant
        divergence_scores[k] = round(r_norm - h_norm, 4)

    # Top-10 positive divergence (high restore, low Hessian → functional without numerical signal)
    pos_divergent = sorted(common_keys, key=lambda k: divergence_scores[k], reverse=True)[:10]
    neg_divergent = sorted(common_keys, key=lambda k: divergence_scores[k])[:10]

    print("\n  Top-10 POSITIVE divergence (high restoration, low Hessian sensitivity):")
    print("  → These layers are functionally critical but numerically unassuming")
    for k in pos_divergent:
        print(f"    {k}: restore={restore_agg[k]:.4f}, "
              f"Hessian={hess_norm[k]:.6f}, "
              f"divergence={divergence_scores[k]:+.4f}")

    print("\n  Top-10 NEGATIVE divergence (low restoration, high Hessian sensitivity):")
    print("  → These layers are numerically sensitive but functionally redundant")
    for k in neg_divergent:
        print(f"    {k}: restore={restore_agg[k]:.4f}, "
              f"Hessian={hess_norm[k]:.6f}, "
              f"divergence={divergence_scores[k]:+.4f}")

    # Specific check: are our critical recovery layers (L13_attention, L6_mlp, L7_mlp)
    # in the high-divergence set?  Use precomputed rank dicts from the sort above.
    critical_layers = ["L13_attention", "L14_attention", "L6_mlp", "L7_mlp"]
    _h_rank_tmp = {
        k: rank for rank, k in enumerate(
            sorted(common_keys, key=lambda x: hess_norm[x]), start=1
        )
    }
    _r_rank_tmp = {
        k: rank for rank, k in enumerate(
            sorted(common_keys, key=lambda x: restore_agg[x]), start=1
        )
    }
    print("\n  Critical recovery layers (from Phase 9) — their divergence profiles:")
    for k in critical_layers:
        if k in divergence_scores:
            h_rank = _h_rank_tmp[k]
            r_rank = _r_rank_tmp[k]
            print(f"    {k}: restore_rank={r_rank}/{len(common_keys)}, "
                  f"Hessian_rank={h_rank}/{len(common_keys)}, "
                  f"divergence={divergence_scores[k]:+.4f}")

    # =========================================================================
    # Step 6: Qualitative summary
    # =========================================================================
    print("\n[Step 6] Summary")

    agg_key = "restoration_aggregate_vs_hessian"
    agg_corr = correlations.get(agg_key, {})
    rho_agg = agg_corr.get("spearman_rho")
    p_agg = agg_corr.get("spearman_p")

    summary_lines = []
    if rho_agg is not None:
        summary_lines.append(
            f"Aggregate restoration vs Hessian: Spearman ρ={rho_agg:.3f} (p={p_agg:.4f})"
        )
        if abs(rho_agg) < 0.4:
            summary_lines.append(
                "RESULT: Rankings substantially diverge. Our probe captures functional "
                "structure that is NOT explained by activation-space Hessian sensitivity. "
                "This is the strongest differentiator from ResQ/OmniQuant: those methods "
                "select layers by numerical sensitivity; we reveal task-selective circuits."
            )
        elif abs(rho_agg) < 0.7:
            summary_lines.append(
                "RESULT: Moderate divergence. Some overlap with Hessian ranking, "
                "but functional task-selectivity adds independent information. "
                "Per-dataset correlations likely show task-specific patterns."
            )
        else:
            summary_lines.append(
                "RESULT: Rankings broadly align. Functional recovery is partially "
                "explained by Hessian sensitivity. Task-selectivity findings (e.g., "
                "attention L13-14 GSM8K-specific) should be emphasized as the "
                "qualitative differentiator, not ranking divergence."
            )

    for line in summary_lines:
        print(f"\n  {line}")

    # =========================================================================
    # Step 7: Save report
    # =========================================================================
    # Precompute rank dicts once (ascending: rank 1 = lowest value)
    hess_rank_dict = {
        k: rank for rank, k in enumerate(
            sorted(common_keys, key=lambda x: hess_norm[x]), start=1
        )
    }
    restore_rank_dict = {
        k: rank for rank, k in enumerate(
            sorted(common_keys, key=lambda x: restore_agg[x]), start=1
        )
    }

    per_layer_combined = {}
    for k in common_keys:
        layer_idx = int(k.split("_")[0][1:])
        component = k.split("_")[1]
        h_rank = hess_rank_dict[k]
        r_rank = restore_rank_dict[k]
        per_layer_combined[k] = {
            "layer": layer_idx,
            "component": component,
            "restoration_recovery": {ds: restoration[k].get(ds, 0.0) for ds in datasets},
            "restoration_aggregate": restore_agg[k],
            "restoration_rank": r_rank,
            "hessian_trace_normalized": hess_norm[k],
            "hessian_rms_activation": hessian_stats[k]["rms_activation"],
            "hessian_rank": h_rank,
            "divergence_score": divergence_scores[k],
            "sqnr_db": sqnr_data.get(k) if sqnr_data else None,
        }

    report = {
        "experiment": "Phase 21: Restoration vs Hessian Sensitivity Ranking",
        "purpose": (
            "Test whether restoration-based layer ranking captures functional structure "
            "beyond Hessian/activation sensitivity (GPTQ/ResQ proxy)"
        ),
        "model": config.model_name,
        "calibration": {
            "dataset": "WikiText-2 test",
            "num_sequences": config.num_calibration_seqs,
            "seq_len": config.seq_len,
        },
        "methodology": {
            "restoration_metric": "Fraction of flipped examples recovered per sub-layer (Phase 9)",
            "hessian_proxy": "Tr(H_l) / d = E[||x_l||^2] / d (activation covariance trace, GPTQ standard)",
            "sqnr_metric": "Signal-to-Quantization-Noise Ratio in dB (Phase 18b, weight-space)",
        },
        "correlations": correlations,
        "summary": summary_lines,
        "critical_layer_divergence": {
            k: per_layer_combined.get(k, {}) for k in critical_layers if k in per_layer_combined
        },
        "top_positive_divergence": [
            {"key": k, **per_layer_combined[k]} for k in pos_divergent if k in per_layer_combined
        ],
        "top_negative_divergence": [
            {"key": k, **per_layer_combined[k]} for k in neg_divergent if k in per_layer_combined
        ],
        "per_layer": per_layer_combined,
    }

    with open(config.output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Report saved: {config.output_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_analysis()
