"""
Phase 24: Hessian-guided vs Restoration-guided Mixed-Precision Comparison
==========================================================================

Direct head-to-head test of the paper's central claim:

  Hessian sensitivity—the layer-importance proxy used by GPTQ/ResQ/OmniQuant—
  selects the WRONG layers for mixed-precision precision allocation.

Experimental design
-------------------
Two 6-layer surgical models are constructed from the same NF4 base:

  1. Hessian-guided  (HESS-6): keep the 6 layers with highest Hessian rank
                                (activation covariance trace) at FP16.
                                These are the layers Hessian-based methods
                                would protect. All 6 fall in the functional
                                dead zone.

  2. Restoration-guided (REST-6): keep the 6 layers identified by the
                                Phase-9/12 restoration probe at FP16.
                                (L13_attn, L14_attn, L1_mlp, L6_mlp,
                                 L7_mlp, L31_mlp)

Both conditions keep exactly 6/32 sub-layers at FP16; 26/32 remain at NF4.
The Hessian-guided selection is loaded programmatically from Phase 21 output
(sensitivity_ranking_comparison.json); fallback to hardcoded values.

Evaluation: GSM8K, ARC-Challenge, MATH (500 examples each, seed=42).
Statistics: accuracy per condition + McNemar paired tests.

Output: results/hessian_vs_restoration.json

Dependencies
------------
  pip install transformers bitsandbytes datasets scipy statsmodels torch

Usage
-----
  python hessian_vs_restoration.py             # full run (FP16 + NF4 + HESS + REST + stats)
  python hessian_vs_restoration.py --stats-only
      # Skip inference. Load existing JSON, recompute McNemar stats (incl. any
      # newly-added comparisons) from saved per-example correctness arrays.
      # Requires a prior full run that wrote per_example_correct arrays.
"""

import argparse
import json
import logging
import os
import re
import gc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

try:
    from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class Config:
    """Immutable configuration for Phase 24."""
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    num_layers: int = 32
    n_examples: int = 500
    seed: int = 42
    batch_size: int = 8
    max_new_tokens: int = 256
    output_dir: str = "results"
    output_file: str = os.path.join("results", "hessian_vs_restoration.json")
    # Phase 21 output for programmatic Hessian layer selection
    phase21_file: str = os.path.join(
        "results", "sensitivity_ranking_comparison.json"
    )
    phase21_fallback: str = os.path.expanduser(
        "~/Downloads/sensitivity_ranking_comparison.json"
    )
    # Number of layers kept at FP16 in each surgical model
    n_surgical_layers: int = 6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# Restoration-guided layers: Phase 12 optimal (hardcoded — proven by Phase 9/12)
RESTORATION_LAYERS: List[Tuple[str, int]] = [
    ("attention", 13),
    ("attention", 14),
    ("mlp", 1),
    ("mlp", 6),
    ("mlp", 7),
    ("mlp", 31),
]


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def aggressive_cleanup() -> None:
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def mcnemar_p(correct_a: List[bool], correct_b: List[bool]) -> float:
    """McNemar's test: returns p-value for paired binary comparison."""
    if not HAS_STATSMODELS:
        return float("nan")
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if not a and bb)
    if b + c == 0:
        return 1.0
    import numpy as np
    table = np.array([[0, b], [c, 0]])
    use_exact = (b + c) < 25
    result = sm_mcnemar(table, exact=use_exact, correction=not use_exact)
    return float(result.pvalue)


# =============================================================================
# Hessian layer selection
# =============================================================================

def load_hessian_top_layers(cfg: Config) -> List[Tuple[str, int]]:
    """
    Load Phase 21 JSON and return the top-N layers by Hessian rank.
    Falls back to hardcoded values if file not found.
    """
    path = None
    for candidate in [cfg.phase21_file, cfg.phase21_fallback]:
        if os.path.exists(candidate):
            path = candidate
            break

    if path is None:
        logger.warning(
            "Phase 21 JSON not found; using hardcoded Hessian-top-6 "
            "(L22, L27, L30, L24, L25, L28 attention)"
        )
        return [
            ("attention", 22),
            ("attention", 27),
            ("attention", 30),
            ("attention", 24),
            ("attention", 25),
            ("attention", 28),
        ]

    try:
        with open(path) as f:
            data = json.load(f)
        per_layer: Dict[str, Dict] = data["per_layer"]
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Phase 21 JSON malformed (%s); falling back to hardcoded.", e)
        return [
            ("attention", 22), ("attention", 27), ("attention", 30),
            ("attention", 24), ("attention", 25), ("attention", 28),
        ]

    ranked = sorted(per_layer.items(), key=lambda x: -x[1]["hessian_rank"])
    top_n = ranked[: cfg.n_surgical_layers]

    layers = [(info["component"], info["layer"]) for _, info in top_n]
    logger.info("Hessian-top-%d layers: %s", cfg.n_surgical_layers, layers)
    return layers


# =============================================================================
# Dataset loading
# =============================================================================

def load_gsm8k(n: int) -> List[Dict]:
    ds = load_dataset("gsm8k", "main", split="test")
    return [
        {"index": i, "question": ex["question"], "answer": ex["answer"],
         "dataset": "gsm8k"}
        for i, ex in enumerate(ds)
        if i < n
    ]


def load_arc_challenge(n: int) -> List[Dict]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    examples = []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        options_text = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        examples.append({
            "index": i, "question": ex["question"],
            "choices_labels": labels, "choices_texts": texts,
            "options_text": options_text, "answer": ex["answerKey"],
            "dataset": "arc",
        })
    return examples


def load_math(n: int) -> List[Dict]:
    for source, split in [
        ("hendrycks/competition_math", "test"),
        ("DigitalLearningGmbH/MATH-lighteval", "test"),
        ("hendrycks/competition_math", "train"),
    ]:
        try:
            ds = load_dataset(source, split=split)
            logger.info("MATH loaded from %s (%s)", source, split)
            break
        except Exception as e:
            logger.warning("MATH source %s/%s failed: %s", source, split, e)
            continue
    else:
        raise RuntimeError("Could not load MATH dataset")
    return [
        {"index": i, "problem": ex["problem"], "solution": ex["solution"],
         "dataset": "math"}
        for i, ex in enumerate(ds)
        if i < n
    ]


# =============================================================================
# Prompt formatting & answer extraction
# =============================================================================

def format_prompt(example: Dict) -> str:
    ds = example["dataset"]
    if ds == "gsm8k":
        return (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Solve this math problem. End with your numerical answer after '####'.\n\n"
            f"{example['question']}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    if ds == "arc":
        return (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Answer this science question by selecting the correct option.\n\n"
            f"Question: {example['question']}\n\n"
            f"Options:\n{example['options_text']}\n\n"
            f"Respond with just the letter ({', '.join(example['choices_labels'])})."
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    if ds == "math":
        return (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Solve this problem. Put your final answer in \\boxed{{}}.\n\n"
            f"{example['problem']}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    return ""


def extract_answer(response: str, example: Dict) -> Optional[str]:
    ds = example["dataset"]
    text = response.replace("**", "").replace("*", "")
    if ds == "gsm8k":
        m = re.search(r"\\boxed\{([^}]+)\}", text)
        if m:
            val = m.group(1).strip().replace(",", "")
            try:
                float(val); return val
            except ValueError:
                pass
        m = re.search(r"####\s*\$?([+-]?\d[\d,]*\.?\d*)", text)
        if m:
            return m.group(1).replace(",", "")
        nums = re.findall(r"(?<!\d)([+-]?\d[\d,]*\.?\d*)(?!\d)", text)
        return nums[-1].replace(",", "") if nums else None
    if ds == "arc":
        valid = example.get("choices_labels", ["A", "B", "C", "D"])
        vp = "".join(valid)
        if text and text[0].upper() in valid:
            return text[0].upper()
        for pat in [
            rf"answer\s+is[:\s]*([{vp}])\b",
            rf"\b([{vp}])\s+is\s+(?:the\s+)?(?:correct|right)",
            rf"^([{vp}])\s*[\.:\)]",
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).upper()
        for lbl in valid:
            if re.search(rf"\b{lbl}\b", text, re.IGNORECASE):
                return lbl.upper()
        return None
    if ds == "math":
        m = re.search(r"\\boxed\{([^}]+)\}", text)
        if m:
            return m.group(1).strip()
        m = re.search(r"answer\s+is[:\s]*(.+?)(?:\.|$)", text, re.IGNORECASE)
        return m.group(1).strip() if m else None
    return None


def extract_ground_truth(example: Dict) -> str:
    ds = example["dataset"]
    if ds == "gsm8k":
        m = re.search(r"####\s*([+-]?\d[\d,]*\.?\d*)", example["answer"])
        return m.group(1).replace(",", "") if m else ""
    if ds == "arc":
        return example["answer"]
    if ds == "math":
        m = re.search(r"\\boxed\{([^}]+)\}", example["solution"])
        return m.group(1).strip() if m else ""
    return ""


def check_correct(model_ans: Optional[str], gt: str, ds: str) -> bool:
    if not model_ans or not gt:
        return False
    if ds == "gsm8k":
        try:
            return abs(float(model_ans) - float(gt)) < 1e-6
        except ValueError:
            return False
    if ds == "arc":
        return model_ans.upper() == gt.upper()
    if ds == "math":
        return model_ans.strip().lower() == gt.strip().lower()
    return model_ans == gt


# =============================================================================
# Model loading
# =============================================================================

def load_fp16(model_name: str):
    logger.info("Loading FP16 model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    return model, tokenizer


def load_nf4(model_name: str):
    logger.info("Loading NF4 model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config,
        dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    return model, tokenizer


def cache_fp16_weights(model_name: str, num_layers: int) -> Dict:
    logger.info("Caching FP16 weights for %d layers...", num_layers)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16,
        device_map="cpu", low_cpu_mem_usage=True,
    )
    cache: Dict = {"attention": {}, "mlp": {}}
    for idx in tqdm(range(num_layers), desc="  Caching"):
        layer = model.model.layers[idx]
        cache["attention"][idx] = {
            proj: {
                "weight": getattr(layer.self_attn, proj).weight.data.clone().cpu(),
                "bias": (getattr(layer.self_attn, proj).bias.data.clone().cpu()
                         if getattr(layer.self_attn, proj).bias is not None else None),
                "in_features": getattr(layer.self_attn, proj).in_features,
                "out_features": getattr(layer.self_attn, proj).out_features,
            }
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj")
        }
        cache["mlp"][idx] = {
            proj: {
                "weight": getattr(layer.mlp, proj).weight.data.clone().cpu(),
                "bias": (getattr(layer.mlp, proj).bias.data.clone().cpu()
                         if getattr(layer.mlp, proj).bias is not None else None),
                "in_features": getattr(layer.mlp, proj).in_features,
                "out_features": getattr(layer.mlp, proj).out_features,
            }
            for proj in ("gate_proj", "up_proj", "down_proj")
        }
    del model
    aggressive_cleanup()
    logger.info("Cached %d layers", num_layers)
    return cache


# =============================================================================
# Multi-layer transplant / restore
# =============================================================================

def transplant_layer_set(
    model,
    fp16_cache: Dict,
    layer_specs: List[Tuple[str, int]],
    device: str,
) -> Dict[Tuple[str, int, str], Any]:
    """
    Transplant FP16 weights for all (component, layer_idx) pairs.
    Returns saved originals keyed by (component, layer_idx, proj_name).
    """
    originals: Dict[Tuple[str, int, str], Any] = {}
    for comp, idx in layer_specs:
        layer = model.model.layers[idx]
        proj_names = (
            ("q_proj", "k_proj", "v_proj", "o_proj") if comp == "attention"
            else ("gate_proj", "up_proj", "down_proj")
        )
        target = layer.self_attn if comp == "attention" else layer.mlp
        for proj in proj_names:
            originals[(comp, idx, proj)] = getattr(target, proj)
            info = fp16_cache[comp][idx][proj]
            new_linear = nn.Linear(
                info["in_features"], info["out_features"],
                bias=info["bias"] is not None,
                dtype=torch.bfloat16,
                device=device,
            )
            new_linear.weight.data = info["weight"].to(device=device, dtype=torch.bfloat16)
            if info["bias"] is not None:
                new_linear.bias.data = info["bias"].to(device=device, dtype=torch.bfloat16)
            setattr(target, proj, new_linear)
    return originals


def restore_layer_set(model, originals: Dict[Tuple[str, int, str], Any]) -> None:
    """Reverse all transplants, restoring original NF4 modules."""
    for (comp, idx, proj), original in originals.items():
        layer = model.model.layers[idx]
        target = layer.self_attn if comp == "attention" else layer.mlp
        setattr(target, proj, original)
    originals.clear()
    aggressive_cleanup()


def assert_nf4_clean(model, num_layers: int) -> None:
    """Sanity check: all decoder layer projections must be Linear4bit after restore."""
    from bitsandbytes.nn import Linear4bit
    for idx in range(num_layers):
        layer = model.model.layers[idx]
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            m = getattr(layer.self_attn, proj)
            assert isinstance(m, Linear4bit), \
                f"L{idx}.self_attn.{proj} is {type(m)} — restore_layer_set may have failed"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            m = getattr(layer.mlp, proj)
            assert isinstance(m, Linear4bit), \
                f"L{idx}.mlp.{proj} is {type(m)} — restore_layer_set may have failed"


# =============================================================================
# Inference
# =============================================================================

def run_inference(
    model,
    tokenizer,
    examples: List[Dict],
    batch_size: int,
    max_new_tokens: int,
) -> List[Dict]:
    results = []
    for batch_start in tqdm(range(0, len(examples), batch_size), desc="  inference", leave=False):
        batch = examples[batch_start: batch_start + batch_size]
        prompts = [format_prompt(ex) for ex in batch]
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=1024,
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for ex, inp_ids, out_ids in zip(
            batch, inputs["input_ids"], outputs
        ):
            response = tokenizer.decode(
                out_ids[inp_ids.shape[0]:], skip_special_tokens=True
            )
            gt = extract_ground_truth(ex)
            ans = extract_answer(response, ex)
            results.append({
                "index": ex["index"],
                "correct": check_correct(ans, gt, ex["dataset"]),
                "predicted": ans,
                "ground_truth": gt,
            })
    return results


def accuracy(results: List[Dict]) -> float:
    return sum(r["correct"] for r in results) / len(results) if results else 0.0


# =============================================================================
# Stats-only mode (no GPU inference)
# =============================================================================

def recovery_rate(fp_c, nf_c, cond_c) -> float:
    """Fraction of forward-flipped examples that a condition answers correctly.

    A forward-flipped example is one the FP16 model answers correctly and the
    NF4 model answers incorrectly -- i.e. an item quantization demonstrably
    broke. Both numerator and denominator are restricted to that set:

        recovery = |FP16 right AND NF4 wrong AND cond right|
                 / |FP16 right AND NF4 wrong|

    NOTE: an earlier version of this function omitted the "FP16 right" term from
    the numerator, which counted items FP16 also failed and inflated recovery.
    The per-example arrays saved in the results JSON are the ground truth; this
    function is the single definition used everywhere.
    """
    flipped = [(f, c) for f, n, c in zip(fp_c, nf_c, cond_c) if f and not n]
    if not flipped:
        return 0.0
    return sum(1 for _, c in flipped if c) / len(flipped)


def _recompute_ds_block(block: Dict[str, Any]) -> Dict[str, Any]:
    pe = block.get("per_example_correct")
    if not pe:
        raise KeyError(
            "per_example_correct missing — run a full pass once with the "
            "updated script before --stats-only can be used"
        )
    fp_c, nf_c = pe["fp16"], pe["nf4"]
    h_c,  r_c  = pe["hessian"], pe["restoration"]

    fp_acc = sum(fp_c) / len(fp_c)
    nf_acc = sum(nf_c) / len(nf_c)
    h_acc  = sum(h_c)  / len(h_c)
    r_acc  = sum(r_c)  / len(r_c)

    flipped = sum(1 for a, b in zip(fp_c, nf_c) if a and not b)
    h_recovery = recovery_rate(fp_c, nf_c, h_c)
    r_recovery = recovery_rate(fp_c, nf_c, r_c)

    return {
        "fp16_acc": round(fp_acc, 4),
        "nf4_acc": round(nf_acc, 4),
        "hessian_surgical_acc": round(h_acc, 4),
        "restoration_surgical_acc": round(r_acc, 4),
        "flipped_count": flipped,
        "hessian_recovery_rate": round(h_recovery, 4),
        "restoration_recovery_rate": round(r_recovery, 4),
        "mcnemar": {
            "rest_vs_nf4_p":  round(mcnemar_p(r_c, nf_c), 4),
            "hess_vs_nf4_p":  round(mcnemar_p(h_c, nf_c), 4),
            "rest_vs_fp16_p": round(mcnemar_p(r_c, fp_c), 4),
            "hess_vs_fp16_p": round(mcnemar_p(h_c, fp_c), 4),
            "rest_vs_hess_p": round(mcnemar_p(r_c, h_c), 4),
        },
        "per_example_correct": pe,
    }


def recompute_stats_from_json(json_path: str) -> None:
    """Reload existing results JSON and recompute all stats from per-example arrays.

    No GPU inference. Used to add new McNemar comparisons or fix stale stats
    after the experiment has already produced per-example correctness arrays.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(json_path)
    if not HAS_STATSMODELS:
        raise RuntimeError(
            "statsmodels not installed — `pip install statsmodels` required "
            "for --stats-only mode"
        )

    with open(json_path) as f:
        data = json.load(f)

    for ds_name in ("gsm8k", "arc", "math"):
        if ds_name not in data.get("results", {}):
            logger.warning("%s missing from results; skipping", ds_name)
            continue
        logger.info("Recomputing stats for %s...", ds_name)
        data["results"][ds_name] = _recompute_ds_block(data["results"][ds_name])

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Stats recomputed and saved: %s", json_path)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 24: Hessian vs Restoration")
    parser.add_argument(
        "--stats-only", action="store_true",
        help="Skip inference; recompute stats from saved per-example arrays in "
             "the existing results JSON. Requires a prior full run.",
    )
    args = parser.parse_args()

    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    if args.stats_only:
        logger.info("=" * 60)
        logger.info("Phase 24 --stats-only: no GPU inference")
        logger.info("=" * 60)
        recompute_stats_from_json(cfg.output_file)
        return

    set_seed(cfg.seed)

    logger.info("=" * 60)
    logger.info("Phase 24: Hessian-guided vs Restoration-guided")
    logger.info("=" * 60)

    assert len(RESTORATION_LAYERS) == cfg.n_surgical_layers, (
        f"RESTORATION_LAYERS has {len(RESTORATION_LAYERS)} entries but "
        f"cfg.n_surgical_layers={cfg.n_surgical_layers} — unfair comparison"
    )

    # ── Hessian layer selection ────────────────────────────────────────────────
    hessian_layers = load_hessian_top_layers(cfg)
    logger.info("Hessian-guided layers  : %s", hessian_layers)
    logger.info("Restoration-guided     : %s", RESTORATION_LAYERS)

    # ── Load datasets ──────────────────────────────────────────────────────────
    logger.info("Loading datasets...")
    all_examples = {
        "gsm8k": load_gsm8k(cfg.n_examples),
        "arc":   load_arc_challenge(cfg.n_examples),
        "math":  load_math(cfg.n_examples),
    }

    # ── FP16 baseline ──────────────────────────────────────────────────────────
    logger.info("Evaluating FP16 baseline...")
    fp16_model, tokenizer = load_fp16(cfg.model_name)
    fp16_results: Dict[str, List[Dict]] = {}
    for ds_name, examples in all_examples.items():
        logger.info("  FP16 %s...", ds_name)
        fp16_results[ds_name] = run_inference(
            fp16_model, tokenizer, examples, cfg.batch_size, cfg.max_new_tokens
        )
        logger.info("    acc = %.2f%%", accuracy(fp16_results[ds_name]) * 100)
    del fp16_model
    aggressive_cleanup()

    # ── Cache FP16 weights ─────────────────────────────────────────────────────
    fp16_cache = cache_fp16_weights(cfg.model_name, cfg.num_layers)

    # ── Load NF4 model (reused across conditions) ──────────────────────────────
    logger.info("Loading NF4 model (shared across conditions)...")
    nf4_model, _ = load_nf4(cfg.model_name)

    # ── NF4 baseline ──────────────────────────────────────────────────────────
    logger.info("Evaluating NF4 baseline...")
    nf4_results: Dict[str, List[Dict]] = {}
    for ds_name, examples in all_examples.items():
        logger.info("  NF4 %s...", ds_name)
        nf4_results[ds_name] = run_inference(
            nf4_model, tokenizer, examples, cfg.batch_size, cfg.max_new_tokens
        )
        logger.info("    acc = %.2f%%", accuracy(nf4_results[ds_name]) * 100)

    # ── Hessian-guided surgical model ─────────────────────────────────────────
    logger.info("Evaluating Hessian-guided surgical (top-%d Hessian layers)...",
                cfg.n_surgical_layers)
    originals_h = transplant_layer_set(nf4_model, fp16_cache, hessian_layers, cfg.device)
    hess_results: Dict[str, List[Dict]] = {}
    for ds_name, examples in all_examples.items():
        logger.info("  Hessian-surgical %s...", ds_name)
        hess_results[ds_name] = run_inference(
            nf4_model, tokenizer, examples, cfg.batch_size, cfg.max_new_tokens
        )
        logger.info("    acc = %.2f%%", accuracy(hess_results[ds_name]) * 100)
    restore_layer_set(nf4_model, originals_h)
    assert_nf4_clean(nf4_model, cfg.num_layers)

    # ── Restoration-guided surgical model ─────────────────────────────────────
    logger.info("Evaluating Restoration-guided surgical (Phase 12 optimal)...")
    originals_r = transplant_layer_set(
        nf4_model, fp16_cache, RESTORATION_LAYERS, cfg.device
    )
    rest_results: Dict[str, List[Dict]] = {}
    for ds_name, examples in all_examples.items():
        logger.info("  Restoration-surgical %s...", ds_name)
        rest_results[ds_name] = run_inference(
            nf4_model, tokenizer, examples, cfg.batch_size, cfg.max_new_tokens
        )
        logger.info("    acc = %.2f%%", accuracy(rest_results[ds_name]) * 100)
    restore_layer_set(nf4_model, originals_r)
    assert_nf4_clean(nf4_model, cfg.num_layers)

    del nf4_model
    aggressive_cleanup()

    # ── Statistics ────────────────────────────────────────────────────────────
    output: Dict[str, Any] = {
        "experiment": "Phase 24: Hessian-guided vs Restoration-guided",
        "model": cfg.model_name,
        "n_examples": cfg.n_examples,
        "seed": cfg.seed,
        "n_surgical_layers": cfg.n_surgical_layers,
        "hessian_layers": [{"component": c, "layer": i} for c, i in hessian_layers],
        "restoration_layers": [{"component": c, "layer": i} for c, i in RESTORATION_LAYERS],
        "results": {},
    }

    for ds_name in ("gsm8k", "arc", "math"):
        fp_acc  = accuracy(fp16_results[ds_name])
        nf_acc  = accuracy(nf4_results[ds_name])
        h_acc   = accuracy(hess_results[ds_name])
        r_acc   = accuracy(rest_results[ds_name])

        fp_c  = [x["correct"] for x in fp16_results[ds_name]]
        nf_c  = [x["correct"] for x in nf4_results[ds_name]]
        h_c   = [x["correct"] for x in hess_results[ds_name]]
        r_c   = [x["correct"] for x in rest_results[ds_name]]

        flipped = sum(1 for a, b in zip(fp_c, nf_c) if a and not b)

        h_recovery = recovery_rate(fp_c, nf_c, h_c)
        r_recovery = recovery_rate(fp_c, nf_c, r_c)

        output["results"][ds_name] = {
            "fp16_acc": round(fp_acc, 4),
            "nf4_acc": round(nf_acc, 4),
            "hessian_surgical_acc": round(h_acc, 4),
            "restoration_surgical_acc": round(r_acc, 4),
            "flipped_count": flipped,
            "hessian_recovery_rate": round(h_recovery, 4),
            "restoration_recovery_rate": round(r_recovery, 4),
            "mcnemar": {
                "rest_vs_nf4_p":  round(mcnemar_p(r_c, nf_c), 4),
                "hess_vs_nf4_p":  round(mcnemar_p(h_c, nf_c), 4),
                "rest_vs_fp16_p": round(mcnemar_p(r_c, fp_c), 4),
                "hess_vs_fp16_p": round(mcnemar_p(h_c, fp_c), 4),
                "rest_vs_hess_p": round(mcnemar_p(r_c, h_c), 4),
            },
            "per_example_correct": {
                "fp16": fp_c,
                "nf4":  nf_c,
                "hessian": h_c,
                "restoration": r_c,
            },
        }

        logger.info(
            "%s — FP16=%.1f%%  NF4=%.1f%%  HESS=%.1f%%  REST=%.1f%%  "
            "flipped=%d  HESS_rec=%.1f%%  REST_rec=%.1f%%",
            ds_name.upper(),
            fp_acc * 100, nf_acc * 100,
            h_acc * 100, r_acc * 100,
            flipped,
            h_recovery * 100, r_recovery * 100,
        )

    out_path = cfg.output_file
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
