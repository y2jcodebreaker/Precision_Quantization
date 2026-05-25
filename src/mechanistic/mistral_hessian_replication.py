"""Phase 21 (Mistral edition): replicate the Hessian vs restoration ranking
comparison on Mistral-7B-Instruct-v0.3 to close paper weakness W1.

Methodology mirrors paper_a_mechanistic/sensitivity_ranking_comparison.py
(the Llama Phase 21 script) byte-for-byte where possible:

  - Activation-covariance Hessian proxy:  Tr(H_l) / d  =  E[||x_l||^2] / d
  - Calibration corpus: WikiText-2 raw-v1, test split, 128 sequences x
    2048 tokens, sliced from the concatenated corpus (GPTQ protocol).
  - Restoration ranks: per-sub-layer recovery from the existing Mistral
    cross-task sweep already produced by mistral_replication.py.

Inputs (must exist before running):
  results/mistral_replication_report.json
    -> layer_rankings.{gsm8k,arc,math}.{attention_scores,mlp_scores}
       (per-sub-layer restoration recovery rates on Mistral; produced by
       paper_a_mechanistic/../mistral_replication.py or copied from
       ~/Downloads/mistral_replication_report.json)

Outputs:
  results/mistral_hessian_replication.json
    -> Spearman rho (aggregate + per task), per-sub-layer Hessian and
       restoration ranks, audit-trail Hessian trace values.

Run on Lightning.ai:
  pip install scipy statsmodels
  # Make sure mistral_replication_report.json is in results/
  python paper_a_mechanistic/mistral_hessian_replication.py

Authentication:
  Set HF_TOKEN environment variable for gated Mistral access.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    raise ImportError("transformers required: pip install transformers") from e

try:
    from datasets import load_dataset
except ImportError as e:
    raise ImportError("datasets required: pip install datasets") from e

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration mirroring Phase 21 (Llama) but for Mistral."""
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    num_layers: int = 32

    # GPTQ-standard calibration protocol
    num_calibration_seqs: int = 128
    seq_len: int = 2048

    # I/O
    output_dir: str = "results"
    output_file: str = os.path.join("results", "mistral_hessian_replication.json")

    # Restoration ranks come from the existing Mistral cross-task sweep.
    # Primary location is results/; fall back to ~/Downloads/ which is where
    # the Lightning.ai outputs are typically downloaded to locally.
    mistral_report_primary: str = os.path.join(
        "results", "mistral_replication_report.json"
    )
    mistral_report_fallback: str = os.path.expanduser(
        "~/Downloads/mistral_replication_report.json"
    )

    # FP16 model load (Phase 21 protocol: full precision for accurate
    # activation statistics). transformers 5.5+ uses `dtype=`.
    torch_dtype: torch.dtype = torch.float16


# =============================================================================
# Tokenizer setup (Mistral convention)
# =============================================================================

def configure_tokenizer(tok):
    """Mistral prefers unk_token over eos as pad token to avoid early-stop
    edge cases during long-context calibration runs."""
    if tok.pad_token is None:
        if tok.unk_token is not None:
            tok.pad_token = tok.unk_token
        else:
            tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


# =============================================================================
# Restoration ranks (from existing Mistral run)
# =============================================================================

def load_restoration_scores(cfg: Config) -> Dict[str, Dict[str, List[float]]]:
    """Pull per-sub-layer restoration recovery rates from the existing
    Mistral cross-task sweep (mistral_replication_report.json)."""
    path = None
    for candidate in (cfg.mistral_report_primary, cfg.mistral_report_fallback):
        if os.path.exists(candidate):
            path = candidate
            break
    if path is None:
        raise FileNotFoundError(
            "mistral_replication_report.json not found in either:\n"
            f"  {cfg.mistral_report_primary}\n"
            f"  {cfg.mistral_report_fallback}\n"
            "Run mistral_replication.py first, or copy the output JSON to "
            "results/ before running this script."
        )
    logger.info("Loading restoration scores from %s", path)
    with open(path) as f:
        report = json.load(f)

    def to_indexed_list(obj, num_layers: int, label: str) -> List[float]:
        """Accept either a list of length num_layers OR a dict whose keys
        are stringified layer indices ('0'..'num_layers-1'). Return a
        plain list of floats indexed by layer."""
        if isinstance(obj, list):
            if len(obj) != num_layers:
                raise ValueError(
                    f"{label}: expected list of length {num_layers}, got {len(obj)}"
                )
            return [float(v) for v in obj]
        if isinstance(obj, dict):
            if len(obj) != num_layers:
                raise ValueError(
                    f"{label}: expected dict with {num_layers} entries, got {len(obj)}"
                )
            result = [0.0] * num_layers
            for k, v in obj.items():
                idx = int(k)  # keys are stringified layer indices
                if not 0 <= idx < num_layers:
                    raise ValueError(f"{label}: layer index {idx} out of range")
                result[idx] = float(v)
            return result
        raise TypeError(
            f"{label}: expected list or dict, got {type(obj).__name__}"
        )

    lr = report["layer_rankings"]
    out: Dict[str, Dict[str, List[float]]] = {}
    for ds in ("gsm8k", "arc", "math"):
        out[ds] = {
            "attention": to_indexed_list(
                lr[ds]["attention_scores"], cfg.num_layers, f"{ds}.attention_scores"
            ),
            "mlp": to_indexed_list(
                lr[ds]["mlp_scores"], cfg.num_layers, f"{ds}.mlp_scores"
            ),
        }
    return out


# =============================================================================
# Calibration corpus
# =============================================================================

def build_calibration_sequences(tokenizer, cfg: Config) -> List[torch.Tensor]:
    """Build calibration sequences identically to Phase 21:
    WikiText-2 raw-v1 test split, concatenated, sliced into
    `num_calibration_seqs` non-overlapping windows of `seq_len` tokens.
    """
    logger.info("Loading WikiText-2 raw-v1 test split...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0]  # 1D tensor

    total = input_ids.numel()
    stride = total // cfg.num_calibration_seqs
    if stride < cfg.seq_len:
        raise RuntimeError(
            f"WikiText-2 test set is too short for {cfg.num_calibration_seqs} "
            f"x {cfg.seq_len}-token sequences (stride={stride})."
        )
    seqs = []
    for i in range(cfg.num_calibration_seqs):
        start = i * stride
        end = start + cfg.seq_len
        if end <= total:
            seqs.append(input_ids[start:end].unsqueeze(0))  # (1, seq_len)
    logger.info("  Prepared %d calibration sequences x %d tokens",
                len(seqs), cfg.seq_len)
    return seqs


# =============================================================================
# Hessian sensitivity computation (hook-based)
# =============================================================================

def collect_hessian_traces(model, tokenizer, cfg: Config) -> Tuple[List[float], List[float]]:
    """Run calibration corpus and accumulate per-sub-layer activation
    covariance traces:  Tr(H_l) / d  =  E[||x_l||^2] / d.

    Identical to Phase 21 (Llama) — same hook protocol, same accumulation.
    Returns (attention_traces, mlp_traces), each length num_layers.
    """
    seqs = build_calibration_sequences(tokenizer, cfg)

    accumulators: Dict[Tuple[str, int], Dict[str, float]] = {}
    for layer_idx in range(cfg.num_layers):
        for component in ("attention", "mlp"):
            accumulators[(component, layer_idx)] = {
                "sum_sq": 0.0,  # sum of squared L2 norms
                "n": 0,         # token positions observed
                "d": 0,         # input feature dim (set on first call)
            }

    hooks = []

    def make_hook(component: str, layer_idx: int):
        key = (component, layer_idx)

        def hook_fn(module, args, kwargs, output):
            # transformers 5.x calls sub-modules with kwargs
            # (hidden_states=..., position_ids=..., ...). Be robust.
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
            x = x.float()                           # avoid bf16/fp16 precision loss
            flat = x.reshape(-1, x.shape[-1])       # (B*S, d)
            sq_norms = (flat ** 2).sum(dim=-1)      # (B*S,) - squared L2 per position
            accumulators[key]["sum_sq"] += sq_norms.sum().item()
            accumulators[key]["n"] += flat.shape[0]
            accumulators[key]["d"] = flat.shape[1]

        return hook_fn

    # Register hooks. with_kwargs=True is required for transformers 5.x.
    for layer_idx in range(cfg.num_layers):
        layer = model.model.layers[layer_idx]
        # Attention sub-block input
        h_attn = layer.self_attn.register_forward_hook(
            make_hook("attention", layer_idx),
            with_kwargs=True,
        )
        hooks.append(h_attn)
        # MLP sub-block input
        h_mlp = layer.mlp.register_forward_hook(
            make_hook("mlp", layer_idx),
            with_kwargs=True,
        )
        hooks.append(h_mlp)

    device = next(model.parameters()).device
    logger.info("Running %d calibration forward passes on %s...", len(seqs), device)
    model.eval()
    try:
        with torch.no_grad():
            for seq in tqdm(seqs, desc="Calibration"):
                _ = model(seq.to(device))
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()

    # Compute per-layer normalised trace
    attn_traces: List[float] = []
    mlp_traces: List[float] = []
    for layer_idx in range(cfg.num_layers):
        for component, dest in (("attention", attn_traces), ("mlp", mlp_traces)):
            acc = accumulators[(component, layer_idx)]
            if acc["n"] == 0 or acc["d"] == 0:
                logger.warning("No samples for %s L%d; setting trace=0",
                               component, layer_idx)
                dest.append(0.0)
                continue
            # Tr(H)/d = E[||x||^2] / d = (sum_sq / n) / d
            trace = acc["sum_sq"] / (acc["n"] * acc["d"])
            dest.append(trace)
    return attn_traces, mlp_traces


# =============================================================================
# Spearman correlation
# =============================================================================

def spearman(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Spearman rho + p-value, with scipy if available, otherwise manual."""
    if HAS_SCIPY:
        result = scipy_stats.spearmanr(x, y)
        # Newer scipy returns SpearmanrResult with .statistic; older returns tuple.
        rho = getattr(result, "statistic", None)
        if rho is None:
            rho, p = result  # legacy tuple
        else:
            p = result.pvalue
        return float(rho), float(p)
    # Fallback: manual Spearman from ranks.
    n = len(x)
    if n != len(y) or n < 3:
        return float("nan"), float("nan")
    sx = sorted(range(n), key=lambda i: x[i])
    rx = [0.0] * n
    for r, i in enumerate(sx, start=1):
        rx[i] = float(r)
    sy = sorted(range(n), key=lambda i: y[i])
    ry = [0.0] * n
    for r, i in enumerate(sy, start=1):
        ry[i] = float(r)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    rho = 1.0 - 6.0 * d2 / (n * (n * n - 1))
    return float(rho), float("nan")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Mistral Phase 21 replication")
    parser.add_argument(
        "--from-checkpoint", action="store_true",
        help="Skip GPU work; load Hessian traces from "
             "results/mistral_hessian_traces_checkpoint.json and just "
             "recompute the Spearman aggregation.",
    )
    args = parser.parse_args()

    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Mistral Phase 21 replication: Hessian vs restoration ranking")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Restoration scores (from prior Mistral cross-task sweep)
    # ------------------------------------------------------------------
    rest = load_restoration_scores(cfg)

    checkpoint_path = os.path.join(
        cfg.output_dir, "mistral_hessian_traces_checkpoint.json"
    )

    if args.from_checkpoint:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"--from-checkpoint requested but {checkpoint_path} "
                "does not exist. Run without the flag first."
            )
        logger.info("Loading Hessian traces from checkpoint: %s", checkpoint_path)
        with open(checkpoint_path) as f:
            ck = json.load(f)
        attn_trace = ck["attention_traces"]
        mlp_trace = ck["mlp_traces"]
        if len(attn_trace) != cfg.num_layers or len(mlp_trace) != cfg.num_layers:
            raise ValueError("Checkpoint trace length mismatch")
    else:
        # --------------------------------------------------------------
        # 2. Load Mistral FP16 + tokenizer
        # --------------------------------------------------------------
        logger.info("Loading Mistral-7B-Instruct-v0.3 tokenizer + FP16 model...")
        tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(cfg.model_name))
        try:
            # transformers 5.5+ syntax
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                dtype=cfg.torch_dtype,
                device_map="auto",
            )
        except TypeError:
            # Backward compatibility for older transformers
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                torch_dtype=cfg.torch_dtype,
                device_map="auto",
            )
        model.eval()

        # --------------------------------------------------------------
        # 3. Collect Hessian traces
        # --------------------------------------------------------------
        attn_trace, mlp_trace = collect_hessian_traces(model, tokenizer, cfg)

        # Persist the expensive part immediately so any downstream bug
        # does not waste GPU time on a rerun.
        with open(checkpoint_path, "w") as f:
            json.dump({
                "model": cfg.model_name,
                "num_layers": cfg.num_layers,
                "num_calibration_seqs": cfg.num_calibration_seqs,
                "seq_len": cfg.seq_len,
                "attention_traces": attn_trace,
                "mlp_traces": mlp_trace,
            }, f, indent=2)
        logger.info("Checkpoint saved: %s", checkpoint_path)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4. Build aligned 64-element vectors and compute Spearman
    # ------------------------------------------------------------------
    # Layout: indices 0..31 = attention sub-layers; 32..63 = MLP sub-layers.
    # Matches the order used by Phase 21 for direct comparability.
    hessian = attn_trace + mlp_trace

    rest_agg_attn = [
        (rest["gsm8k"]["attention"][i]
         + rest["arc"]["attention"][i]
         + rest["math"]["attention"][i]) / 3.0
        for i in range(cfg.num_layers)
    ]
    rest_agg_mlp = [
        (rest["gsm8k"]["mlp"][i]
         + rest["arc"]["mlp"][i]
         + rest["math"]["mlp"][i]) / 3.0
        for i in range(cfg.num_layers)
    ]
    rest_agg = rest_agg_attn + rest_agg_mlp

    rho_agg, p_agg = spearman(hessian, rest_agg)
    logger.info("Aggregate Spearman rho = %.4f (p = %.2e)", rho_agg, p_agg)

    per_task = {}
    for task in ("gsm8k", "arc", "math"):
        r_t = list(rest[task]["attention"]) + list(rest[task]["mlp"])
        rho_t, p_t = spearman(hessian, r_t)
        per_task[task] = {"rho": rho_t, "p": p_t}
        logger.info("%-5s Spearman rho = %.4f (p = %.2e)", task.upper(), rho_t, p_t)

    # ------------------------------------------------------------------
    # 5. Persist results
    # ------------------------------------------------------------------
    def _per_layer_entry(component: str, idx: int) -> Dict[str, float]:
        trace = attn_trace[idx] if component == "attention" else mlp_trace[idx]
        rest_avg = (
            rest_agg_attn[idx] if component == "attention" else rest_agg_mlp[idx]
        )
        return {
            "hessian_trace_normalized": round(trace, 8),
            "restoration_aggregate": round(rest_avg, 6),
            "restoration_gsm8k": rest["gsm8k"][component][idx],
            "restoration_arc":   rest["arc"][component][idx],
            "restoration_math":  rest["math"][component][idx],
        }

    output = {
        "experiment": "Phase 21 replication on Mistral-7B-Instruct-v0.3",
        "model": cfg.model_name,
        "methodology": {
            "hessian_proxy": "Tr(H_l) / d = E[||x_l||^2] / d (activation covariance trace)",
            "restoration_metric": (
                "Aggregate recovery rate across GSM8K, ARC, MATH; "
                "per-sub-layer means from mistral_replication_report.json"
            ),
            "calibration": (
                f"WikiText-2 raw-v1 (test), {cfg.num_calibration_seqs} sequences "
                f"x {cfg.seq_len} tokens (GPTQ protocol)"
            ),
        },
        "n_sub_layers": 2 * cfg.num_layers,
        "spearman": {
            "aggregate": {"rho": round(rho_agg, 4), "p": round(p_agg, 6)},
            "per_task": {
                t: {"rho": round(v["rho"], 4), "p": round(v["p"], 6)}
                for t, v in per_task.items()
            },
        },
        "per_sub_layer": {
            **{f"L{i}_attention": _per_layer_entry("attention", i)
               for i in range(cfg.num_layers)},
            **{f"L{i}_mlp": _per_layer_entry("mlp", i)
               for i in range(cfg.num_layers)},
        },
    }

    with open(cfg.output_file, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved: %s", cfg.output_file)
    logger.info("Done.")


if __name__ == "__main__":
    main()
