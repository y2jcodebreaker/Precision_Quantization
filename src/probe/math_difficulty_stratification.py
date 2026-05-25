"""
MATH Difficulty Stratification Experiment
==========================================

The multi-seed experiment revealed that MATH showed no significant difference
between conditions (FP16 vs 4-bit gap ~1.1pp, well below noise floor at n=500).

Hypothesis: quantization damage is difficulty-gated. Easy problems (Level 1-3)
are solved robustly regardless of precision; hard problems (Level 4-5) operate
near the model's capability ceiling where small precision differences matter.

This script runs FP16, 4-bit NF4, and Surgical Mix on two difficulty tiers:
  - Easy:  Levels 1-3
  - Hard:  Levels 4-5

Method:
- Load MATH dataset with `level` field, merge into two pools
- n = min(500, available) examples per tier, seed=42
- Conditions: FP16, 4-bit, Surgical Mix (same 6-layer patch as Phase 12-14)
- Statistical test: McNemar's per tier (paired per-example)
- Output: results/math_difficulty_stratification_report.json

Authentication:
---------------
Set HF_TOKEN environment variable for model access.
"""

import json
import os
import re
import gc
import time
import random
import math
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found. McNemar's test will use manual chi2 approximation.")

import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
except ImportError:
    raise ImportError("transformers required: pip install transformers")

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("datasets required: pip install datasets")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir: str = "results"
    output_file: str = os.path.join("results", "math_difficulty_stratification_report.json")
    checkpoint_file: str = os.path.join("results", "math_difficulty_stratification_checkpoint.json")

    num_examples_per_tier: int = 500
    seed: int = 42

    # Two tiers: easy (L1-3), hard (L4-5)
    tiers: Dict[str, Tuple[int, ...]] = field(default_factory=lambda: {
        "easy": (1, 2, 3),
        "hard": (4, 5),
    })

    # The 6 critical layers (same as Phase 12-14)
    attn_fp16_layers: Tuple[int, ...] = (13, 14)
    mlp_fp16_layers: Tuple[int, ...] = (1, 6, 7, 31)

    attn_projections: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    mlp_projections: Tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")

    batch_size: int = 64
    max_new_tokens: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Memory Management
# =============================================================================

def aggressive_cleanup():
    gc.collect()
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# Dataset Loading
# =============================================================================

def load_math_by_tier(tiers: Dict[str, Tuple[int, ...]]) -> Dict[str, List[Dict]]:
    """Load MATH and group into difficulty tiers."""
    print("  Loading MATH dataset with level metadata...")

    dataset = None
    for source, split in [
        ("hendrycks/competition_math", "test"),
        ("DigitalLearningGmbH/MATH-lighteval", "test"),
        ("hendrycks/competition_math", "train"),
        ("DigitalLearningGmbH/MATH-lighteval", "train"),
    ]:
        try:
            dataset = load_dataset(source, split=split)
            print(f"    Loaded from {source} ({split} split)")
            break
        except Exception:
            continue

    if dataset is None:
        raise RuntimeError("Could not load MATH dataset from any source")

    # Invert tiers dict: level_int -> tier_name
    level_to_tier: Dict[int, str] = {}
    for tier_name, levels in tiers.items():
        for lvl in levels:
            level_to_tier[lvl] = tier_name

    tier_pools: Dict[str, List[Dict]] = {name: [] for name in tiers}
    level_counts: Dict[int, int] = {}

    for i, item in enumerate(dataset):
        raw_level = item.get("level", "")
        try:
            lvl = int(str(raw_level).replace("Level", "").strip())
        except (ValueError, AttributeError):
            continue
        tier_name = level_to_tier.get(lvl)
        if tier_name is None:
            continue
        tier_pools[tier_name].append({
            "index": i,
            "problem": item["problem"],
            "solution": item["solution"],
            "level": lvl,
            "type": item.get("type", ""),
            "dataset": "math",
        })
        level_counts[lvl] = level_counts.get(lvl, 0) + 1

    for lvl in sorted(level_counts):
        tier_name = level_to_tier[lvl]
        print(f"    Level {lvl} ({tier_name}): {level_counts[lvl]} examples")
    for name, pool in tier_pools.items():
        levels_str = "+".join(f"L{l}" for l in tiers[name])
        print(f"    Tier '{name}' ({levels_str}): {len(pool)} total")

    return tier_pools


def subsample(examples: List[Dict], seed: int, n: int) -> List[Dict]:
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    return shuffled[:n]


# =============================================================================
# Prompt Formatting & Answer Extraction
# =============================================================================

def format_prompt(example: Dict) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Solve this problem. Put your final answer in \\boxed{{}}.\n\n"
        f"{example['problem']}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def extract_answer(response: str) -> Optional[str]:
    text = response.replace("**", "").replace("*", "")
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    match = re.search(
        r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*(.+?)(?:\.|$)",
        text, re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return None


def extract_ground_truth(example: Dict) -> str:
    match = re.search(r"\\boxed\{([^}]+)\}", example["solution"])
    if match:
        return match.group(1).strip()
    return ""


def check_match(model_answer: Optional[str], ground_truth: str) -> bool:
    if model_answer is None or not ground_truth:
        return False
    return model_answer.strip().lower() == ground_truth.strip().lower()


# =============================================================================
# Inference
# =============================================================================

def run_inference(
    model,
    tokenizer,
    examples: List[Dict],
    batch_size: int,
    max_new_tokens: int,
    label: str,
) -> Dict[str, Any]:
    if not examples:
        return {"accuracy": 0.0, "correct": 0, "total": 0,
                "per_example_correct": [], "wall_time_s": 0.0}

    correct = 0
    total = len(examples)
    per_example_correct = []
    total_generated_tokens = 0
    num_batches = (total + batch_size - 1) // batch_size
    wall_start = time.time()

    for batch_idx in tqdm(range(num_batches), desc=f"    {label}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total)
        batch_examples = examples[start_idx:end_idx]

        prompts = [format_prompt(ex) for ex in batch_examples]
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for i, (example, output) in enumerate(zip(batch_examples, outputs)):
            input_len = inputs["input_ids"][i].shape[0]
            generated_tokens = output[input_len:]
            total_generated_tokens += len(generated_tokens)
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            ground_truth = extract_ground_truth(example)
            model_answer = extract_answer(response)
            is_correct = check_match(model_answer, ground_truth)
            if is_correct:
                correct += 1
            per_example_correct.append(1 if is_correct else 0)

        if (batch_idx + 1) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    wall_time = time.time() - wall_start
    accuracy = correct / total
    throughput = total_generated_tokens / wall_time if wall_time > 0 else 0

    print(f"      {label}: {accuracy:.2%} ({correct}/{total}) | {throughput:.1f} tok/s")

    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "per_example_correct": per_example_correct,
        "wall_time_s": round(wall_time, 2),
        "total_generated_tokens": total_generated_tokens,
        "throughput_tok_per_s": round(throughput, 2),
    }


# =============================================================================
# Model Loading & Surgery
# =============================================================================

def load_fp16_model(model_name: str):
    print("\n  Loading FP16 model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()
    return model, tokenizer


def load_4bit_model(model_name: str):
    print("\n  Loading 4-bit model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto",
    )
    model.eval()
    return model, tokenizer


def unload_model(model, tokenizer=None):
    del model
    if tokenizer:
        del tokenizer
    aggressive_cleanup()


def cache_fp16_weights(model_name: str, config: Config) -> Dict:
    """Cache FP16 weights for the 6 critical layers on CPU."""
    print("\n  Caching FP16 weights for surgical patch layers...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map="cpu", low_cpu_mem_usage=True,
    )
    cache = {"attention": {}, "mlp": {}}

    for layer_idx in config.attn_fp16_layers:
        layer = model.model.layers[layer_idx]
        cache["attention"][layer_idx] = {}
        for proj_name in config.attn_projections:
            proj = getattr(layer.self_attn, proj_name)
            cache["attention"][layer_idx][proj_name] = {
                "weight": proj.weight.data.clone().cpu(),
                "bias": proj.bias.data.clone().cpu() if proj.bias is not None else None,
                "in_features": proj.in_features,
                "out_features": proj.out_features,
            }
        print(f"    Cached attention L{layer_idx}")

    for layer_idx in config.mlp_fp16_layers:
        layer = model.model.layers[layer_idx]
        cache["mlp"][layer_idx] = {}
        for proj_name in config.mlp_projections:
            proj = getattr(layer.mlp, proj_name)
            cache["mlp"][layer_idx][proj_name] = {
                "weight": proj.weight.data.clone().cpu(),
                "bias": proj.bias.data.clone().cpu() if proj.bias is not None else None,
                "in_features": proj.in_features,
                "out_features": proj.out_features,
            }
        print(f"    Cached MLP L{layer_idx}")

    del model
    aggressive_cleanup()
    return cache


def apply_surgical_patch(model, fp16_cache: Dict, config: Config):
    """Replace the 6 critical layers' Linear4bit modules with FP16 nn.Linear."""
    print("\n  Applying surgical FP16 patches...")
    for layer_idx in config.attn_fp16_layers:
        layer = model.model.layers[layer_idx]
        for proj_name in config.attn_projections:
            w = fp16_cache["attention"][layer_idx][proj_name]
            new_linear = nn.Linear(
                w["in_features"], w["out_features"],
                bias=w["bias"] is not None, dtype=torch.float16, device=config.device,
            )
            new_linear.weight.data = w["weight"].to(config.device)
            if w["bias"] is not None:
                new_linear.bias.data = w["bias"].to(config.device)
            setattr(layer.self_attn, proj_name, new_linear)
        print(f"    Patched attention L{layer_idx}")

    for layer_idx in config.mlp_fp16_layers:
        layer = model.model.layers[layer_idx]
        for proj_name in config.mlp_projections:
            w = fp16_cache["mlp"][layer_idx][proj_name]
            new_linear = nn.Linear(
                w["in_features"], w["out_features"],
                bias=w["bias"] is not None, dtype=torch.float16, device=config.device,
            )
            new_linear.weight.data = w["weight"].to(config.device)
            if w["bias"] is not None:
                new_linear.bias.data = w["bias"].to(config.device)
            setattr(layer.mlp, proj_name, new_linear)
        print(f"    Patched MLP L{layer_idx}")


# =============================================================================
# Statistical Tests
# =============================================================================

def mcnemar_test(correct_a: List[int], correct_b: List[int]) -> Dict:
    """McNemar's test for paired binary outcomes."""
    assert len(correct_a) == len(correct_b), "Lists must be same length"
    b = sum(1 for a, bv in zip(correct_a, correct_b) if a == 1 and bv == 0)
    c = sum(1 for a, bv in zip(correct_a, correct_b) if a == 0 and bv == 1)

    if b + c == 0:
        return {"b_count": 0, "c_count": 0, "chi2": 0.0, "p_value": 1.0,
                "significant_at_0.05": False, "note": "no discordant pairs"}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)  # with continuity correction

    if HAS_SCIPY:
        p_value = float(1 - scipy_stats.chi2.cdf(chi2, df=1))
    else:
        x = chi2 / 2.0
        p_value = 1.0 - math.erf(math.sqrt(x)) if x > 0 else 1.0

    return {
        "b_count": b,
        "c_count": c,
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "significant_at_0.05": p_value < 0.05,
    }


def gap_recovery_pct(fp16_acc: float, int4_acc: float, surgical_acc: float) -> Optional[float]:
    """Return gap recovery % or None if gap < 1pp (meaningless division)."""
    gap = fp16_acc - int4_acc
    if abs(gap) < 0.01:
        return None
    return round((surgical_acc - int4_acc) / gap * 100, 1)


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(data: Dict, filepath: str):
    tmp = filepath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, filepath)
    print(f"  [checkpoint] Saved: {filepath}")


def load_checkpoint(filepath: str) -> Optional[Dict]:
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"  WARNING: Corrupt checkpoint, ignoring: {filepath}")
            return None
        return data
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  WARNING: Corrupt checkpoint, ignoring: {filepath} ({e})")
        return None


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)

    tier_labels = {
        "easy": "Easy (L1-3)",
        "hard": "Hard (L4-5)",
    }

    print("=" * 70)
    print("MATH Difficulty Stratification Experiment")
    print("=" * 70)
    print(f"Model:    {config.model_name}")
    print(f"Device:   {config.device}")
    print(f"Tiers:    easy=L1-3, hard=L4-5")
    print(f"n/tier:   {config.num_examples_per_tier}")
    print(f"Seed:     {config.seed}")

    checkpoint = load_checkpoint(config.checkpoint_file)
    tier_results = checkpoint.get("tier_results", {}) if checkpoint else {}
    completed_substep = checkpoint.get("completed_substep", "") if checkpoint else ""

    # =========================================================================
    # Load MATH by tier
    # =========================================================================
    print(f"\n{'='*70}")
    print("Loading MATH Dataset")
    print(f"{'='*70}")

    math_by_tier = load_math_by_tier(config.tiers)
    tier_samples = {}
    for tier_name, pool in math_by_tier.items():
        n = min(config.num_examples_per_tier, len(pool))
        tier_samples[tier_name] = subsample(pool, config.seed, n)
        print(f"  {tier_labels[tier_name]}: {n} examples sampled (pool: {len(pool)})")

    save_checkpoint({
        "tier_results": tier_results,
        "completed_substep": "datasets_loaded",
    }, config.checkpoint_file)

    # =========================================================================
    # Cache FP16 weights for surgical patch
    # =========================================================================
    fp16_cache = cache_fp16_weights(config.model_name, config)
    save_checkpoint({
        "tier_results": tier_results,
        "completed_substep": "fp16_cached",
    }, config.checkpoint_file)

    tier_names = list(config.tiers.keys())

    # =========================================================================
    # Phase A: FP16 baselines
    # =========================================================================
    fp16_done = completed_substep.startswith("fp16_all_done") or all(
        tier_results.get(f"tier_{t}", {}).get("fp16") for t in tier_names
    )

    if not fp16_done:
        print(f"\n{'='*70}")
        print("PHASE A: FP16 Baselines")
        print(f"{'='*70}")

        model, tokenizer = load_fp16_model(config.model_name)
        save_checkpoint({"tier_results": tier_results, "completed_substep": "fp16_model_loaded"},
                        config.checkpoint_file)

        for tier_name in tier_names:
            tier_key = f"tier_{tier_name}"
            if tier_results.get(tier_key, {}).get("fp16"):
                print(f"\n  {tier_labels[tier_name]} FP16 — loaded from checkpoint")
                continue

            print(f"\n  --- {tier_labels[tier_name]} (FP16, n={len(tier_samples[tier_name])}) ---")
            if tier_key not in tier_results:
                tier_results[tier_key] = {"tier": tier_name}

            result = run_inference(
                model, tokenizer, tier_samples[tier_name],
                config.batch_size, config.max_new_tokens,
                f"{tier_labels[tier_name]} FP16",
            )
            tier_results[tier_key]["fp16"] = result

            save_checkpoint({"tier_results": tier_results, "completed_substep": f"fp16_tier_{tier_name}"},
                            config.checkpoint_file)

        unload_model(model, tokenizer)
        save_checkpoint({"tier_results": tier_results, "completed_substep": "fp16_all_done"},
                        config.checkpoint_file)
        print("\n  FP16 complete — model unloaded")

    # =========================================================================
    # Phase B: 4-bit NF4 baselines
    # =========================================================================
    int4_done = completed_substep.startswith("int4_all_done") or all(
        tier_results.get(f"tier_{t}", {}).get("int4") for t in tier_names
    )

    if not int4_done:
        print(f"\n{'='*70}")
        print("PHASE B: 4-bit NF4 Baselines")
        print(f"{'='*70}")

        model, tokenizer = load_4bit_model(config.model_name)
        save_checkpoint({"tier_results": tier_results, "completed_substep": "int4_model_loaded"},
                        config.checkpoint_file)

        for tier_name in tier_names:
            tier_key = f"tier_{tier_name}"
            if tier_results.get(tier_key, {}).get("int4"):
                print(f"\n  {tier_labels[tier_name]} 4-bit — loaded from checkpoint")
                continue

            print(f"\n  --- {tier_labels[tier_name]} (4-bit, n={len(tier_samples[tier_name])}) ---")
            result = run_inference(
                model, tokenizer, tier_samples[tier_name],
                config.batch_size, config.max_new_tokens,
                f"{tier_labels[tier_name]} 4-bit",
            )
            tier_results[tier_key]["int4"] = result

            save_checkpoint({"tier_results": tier_results, "completed_substep": f"int4_tier_{tier_name}"},
                            config.checkpoint_file)

        unload_model(model, tokenizer)
        save_checkpoint({"tier_results": tier_results, "completed_substep": "int4_all_done"},
                        config.checkpoint_file)
        print("\n  4-bit complete — model unloaded")

    # =========================================================================
    # Phase C: Surgical Mix
    # =========================================================================
    surgical_done = completed_substep.startswith("surgical_all_done") or all(
        tier_results.get(f"tier_{t}", {}).get("surgical") for t in tier_names
    )

    if not surgical_done:
        print(f"\n{'='*70}")
        print("PHASE C: Surgical Mix")
        print(f"{'='*70}")

        model, tokenizer = load_4bit_model(config.model_name)
        apply_surgical_patch(model, fp16_cache, config)
        save_checkpoint({"tier_results": tier_results, "completed_substep": "surgical_model_ready"},
                        config.checkpoint_file)

        for tier_name in tier_names:
            tier_key = f"tier_{tier_name}"
            if tier_results.get(tier_key, {}).get("surgical"):
                print(f"\n  {tier_labels[tier_name]} Surgical — loaded from checkpoint")
                continue

            print(f"\n  --- {tier_labels[tier_name]} (Surgical, n={len(tier_samples[tier_name])}) ---")
            result = run_inference(
                model, tokenizer, tier_samples[tier_name],
                config.batch_size, config.max_new_tokens,
                f"{tier_labels[tier_name]} Surgical",
            )
            tier_results[tier_key]["surgical"] = result

            save_checkpoint({"tier_results": tier_results, "completed_substep": f"surgical_tier_{tier_name}"},
                            config.checkpoint_file)

        unload_model(model, tokenizer)
        save_checkpoint({"tier_results": tier_results, "completed_substep": "surgical_all_done"},
                        config.checkpoint_file)
        print("\n  Surgical complete — model unloaded")

    # =========================================================================
    # Phase D: Statistical analysis
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE D: Statistical Analysis")
    print(f"{'='*70}")

    tier_summary = []
    statistical_tests = {}

    for tier_name in tier_names:
        tier_key = f"tier_{tier_name}"
        tr = tier_results[tier_key]

        fp16_acc  = tr["fp16"]["accuracy"]
        int4_acc  = tr["int4"]["accuracy"]
        surg_acc  = tr["surgical"]["accuracy"]
        gap_pp    = round((fp16_acc - int4_acc) * 100, 2)
        surg_gain = round((surg_acc - int4_acc) * 100, 2)
        recovery  = gap_recovery_pct(fp16_acc, int4_acc, surg_acc)

        fp16_vec  = tr["fp16"]["per_example_correct"]
        int4_vec  = tr["int4"]["per_example_correct"]
        surg_vec  = tr["surgical"]["per_example_correct"]

        stat_fp16_vs_int4 = mcnemar_test(fp16_vec, int4_vec)
        stat_fp16_vs_surg = mcnemar_test(fp16_vec, surg_vec)
        stat_int4_vs_surg = mcnemar_test(int4_vec, surg_vec)

        statistical_tests[tier_key] = {
            "fp16_vs_int4":     stat_fp16_vs_int4,
            "fp16_vs_surgical": stat_fp16_vs_surg,
            "int4_vs_surgical": stat_int4_vs_surg,
        }

        tier_summary.append({
            "tier":              tier_name,
            "tier_label":        tier_labels[tier_name],
            "levels":            list(config.tiers[tier_name]),
            "n":                 tr["fp16"]["total"],
            "fp16_accuracy":     fp16_acc,
            "int4_accuracy":     int4_acc,
            "surgical_accuracy": surg_acc,
            "gap_pp":            gap_pp,
            "surgical_gain_pp":  surg_gain,
            "gap_recovery_pct":  recovery,
            "fp16_vs_int4_p":    stat_fp16_vs_int4["p_value"],
            "fp16_vs_int4_sig":  stat_fp16_vs_int4["significant_at_0.05"],
            "int4_vs_surg_p":    stat_int4_vs_surg["p_value"],
            "int4_vs_surg_sig":  stat_int4_vs_surg["significant_at_0.05"],
        })

        sig_marker = "✓" if stat_fp16_vs_int4["significant_at_0.05"] else "ns"
        print(
            f"  {tier_labels[tier_name]}: FP16={fp16_acc:.2%}  4-bit={int4_acc:.2%}  "
            f"Surgical={surg_acc:.2%}  gap={gap_pp:+.1f}pp  "
            f"recovery={recovery if recovery is not None else 'N/A'}%  [{sig_marker}]"
        )

    # =========================================================================
    # Save final report
    # =========================================================================
    report = {
        "config": {
            "model": config.model_name,
            "num_examples_per_tier": config.num_examples_per_tier,
            "seed": config.seed,
            "tiers": {k: list(v) for k, v in config.tiers.items()},
            "attn_fp16_layers": list(config.attn_fp16_layers),
            "mlp_fp16_layers": list(config.mlp_fp16_layers),
        },
        "tier_results": {
            k: {
                "tier":     v["tier"],
                "fp16":     {kk: vv for kk, vv in v["fp16"].items() if kk != "per_example_correct"},
                "int4":     {kk: vv for kk, vv in v["int4"].items() if kk != "per_example_correct"},
                "surgical": {kk: vv for kk, vv in v["surgical"].items() if kk != "per_example_correct"},
            }
            for k, v in tier_results.items()
        },
        "tier_summary": tier_summary,
        "statistical_tests": statistical_tests,
    }

    tmp_out = config.output_file + ".tmp"
    with open(tmp_out, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp_out, config.output_file)
    print(f"\n  Final report saved: {config.output_file}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY: Easy vs Hard MATH")
    print(f"{'='*70}")
    print(f"{'Tier':<16} {'FP16':>8} {'4-bit':>8} {'Surgical':>10} {'Gap':>8} {'Recovery':>10} {'Sig?':>6}")
    print("-" * 68)
    for row in tier_summary:
        rec = f"{row['gap_recovery_pct']}%" if row['gap_recovery_pct'] is not None else "N/A"
        sig = "✓" if row["fp16_vs_int4_sig"] else "ns"
        print(
            f"  {row['tier_label']:<14} {row['fp16_accuracy']:>8.2%} {row['int4_accuracy']:>8.2%} "
            f"{row['surgical_accuracy']:>10.2%} {row['gap_pp']:>+7.1f}pp "
            f"{rec:>10} {sig:>6}"
        )

    print(f"\nDone. Output: {config.output_file}")


if __name__ == "__main__":
    main()
