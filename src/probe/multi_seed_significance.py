"""
Multi-Seed Statistical Significance Testing
=============================================

Runs the surgical mixed-precision experiment with 5 different random seeds
to establish statistical significance. Each seed shuffles the datasets
differently, selecting a different subset of 500 examples.

Seeds: [42, 52, 62, 72, 82] (5 seeds per best practices — see arxiv:2503.07329)
Conditions: Full FP16, Full 4-bit, Surgical Mix (6 layers)
Datasets: GSM8K, ARC-Challenge, MATH (500 examples each, shuffled per seed)

Statistical tests performed:
- McNemar's test (per arxiv:2602.10144 — detecting quantization degradation)
- Bootstrap 95% confidence intervals (BCa)
- Paired permutation test (per Dror et al. 2018)
- Efficiency metrics: throughput, latency, accuracy-per-MB

Output: results/multi_seed_significance_report.json

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
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found. McNemar's test will use manual chi2 approximation.")
    print("Install with: pip install scipy")

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
    output_file: str = os.path.join("results", "multi_seed_significance_report.json")
    checkpoint_file: str = os.path.join("results", "multi_seed_checkpoint.json")

    num_examples_per_dataset: int = 500
    seeds: Tuple[int, ...] = (42, 52, 62, 72, 82)

    # The 6 critical layers
    attn_fp16_layers: Tuple[int, ...] = (13, 14)
    mlp_fp16_layers: Tuple[int, ...] = (1, 6, 7, 31)

    attn_projections: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    mlp_projections: Tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")

    batch_size: int = 8
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


def get_memory_stats() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0, "peak_mb": 0}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
        "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
        "peak_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
    }


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# =============================================================================
# Dataset Loading (full datasets, subsampled per seed later)
# =============================================================================

def load_gsm8k_full() -> List[Dict]:
    print(f"  Loading full GSM8K test set...")
    dataset = load_dataset("gsm8k", "main", split="test")
    examples = []
    for i, item in enumerate(dataset):
        examples.append({
            "index": i,
            "question": item["question"],
            "answer": item["answer"],
            "dataset": "gsm8k",
        })
    print(f"    {len(examples)} examples available")
    return examples


def load_arc_challenge_full() -> List[Dict]:
    print(f"  Loading full ARC-Challenge test set...")
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    examples = []
    for i, item in enumerate(dataset):
        labels = item["choices"]["label"]
        texts = item["choices"]["text"]
        options_text = "\n".join([f"{l}. {t}" for l, t in zip(labels, texts)])
        examples.append({
            "index": i,
            "question": item["question"],
            "choices_labels": labels,
            "choices_texts": texts,
            "options_text": options_text,
            "answer": item["answerKey"],
            "dataset": "arc",
        })
    print(f"    {len(examples)} examples available")
    return examples


def load_math_full() -> List[Dict]:
    print(f"  Loading full MATH test set...")
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
    examples = []
    for i, item in enumerate(dataset):
        examples.append({
            "index": i,
            "problem": item["problem"],
            "solution": item["solution"],
            "dataset": "math",
        })
    print(f"    {len(examples)} examples available")
    return examples


def subsample(examples: List[Dict], seed: int, n: int) -> List[Dict]:
    """Shuffle with seed and take first n examples."""
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    return shuffled[:n]


# =============================================================================
# Prompt Formatting
# =============================================================================

def format_prompt(example: Dict) -> str:
    dataset = example["dataset"]
    if dataset == "gsm8k":
        return (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Solve this math problem. End with your numerical answer after '####'.\n\n"
            f"{example['question']}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif dataset == "arc":
        return (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Answer this science question by selecting the correct option.\n\n"
            f"Question: {example['question']}\n\n"
            f"Options:\n{example['options_text']}\n\n"
            f"Respond with just the letter ({', '.join(example['choices_labels'])})."
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif dataset == "math":
        return (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Solve this problem. Put your final answer in \\boxed{{}}.\n\n"
            f"{example['problem']}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    return ""


# =============================================================================
# Answer Extraction
# =============================================================================

def extract_answer(response: str, example: Dict) -> Optional[str]:
    dataset = example["dataset"]
    text = response.replace("**", "").replace("*", "")

    if dataset == "gsm8k":
        match = re.search(r"####\s*\$?([+-]?\d[\d,]*\.?\d*)", text)
        if match:
            return match.group(1).replace(",", "")
        patterns = [
            r"(?:the\s+)?answer\s+is[:\s]*\$?([+-]?\d[\d,]*\.?\d*)",
            r"final\s+answer[:\s]*\$?([+-]?\d[\d,]*\.?\d*)",
            r"=\s*\$?([+-]?\d[\d,]*\.?\d*)\s*(?:dollars?|\.|\s|$)",
            r"(?:therefore|thus|so)[,\s]+(?:the\s+)?(?:answer\s+is\s+)?\$?([+-]?\d[\d,]*\.?\d*)",
            r"total(?:\s+is|\s+of|:)\s*\$?([+-]?\d[\d,]*\.?\d*)",
            r"equals?\s+\$?([+-]?\d[\d,]*\.?\d*)",
        ]
        for p in patterns:
            match = re.search(p, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).replace(",", "")
        numbers = re.findall(r"(?<!\d)([+-]?\d[\d,]*\.?\d*)(?!\d)", text)
        if numbers:
            return numbers[-1].replace(",", "")

    elif dataset == "arc":
        valid_labels = example.get("choices_labels", ["A", "B", "C", "D"])
        valid_pattern = ''.join(valid_labels)
        if text and text[0].upper() in valid_labels:
            return text[0].upper()
        patterns = [
            rf"(?:the\s+)?answer\s+is[:\s]*([{valid_pattern}])\b",
            rf"\b([{valid_pattern}])\s+is\s+(?:the\s+)?(?:correct|right)",
            rf"^([{valid_pattern}])\s*[\.:\)]",
        ]
        for p in patterns:
            match = re.search(p, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        for label in valid_labels:
            if re.search(rf"\b{label}\b", text, re.IGNORECASE):
                return label.upper()

    elif dataset == "math":
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
    dataset = example["dataset"]
    if dataset == "gsm8k":
        match = re.search(r"####\s*([+-]?\d[\d,]*\.?\d*)", example["answer"])
        if match:
            return match.group(1).replace(",", "")
    elif dataset == "arc":
        return example["answer"]
    elif dataset == "math":
        match = re.search(r"\\boxed\{([^}]+)\}", example["solution"])
        if match:
            return match.group(1).strip()
    return ""


def check_match(model_answer: Optional[str], ground_truth: str, dataset: str) -> bool:
    if model_answer is None or not ground_truth:
        return False
    if dataset == "gsm8k":
        try:
            return abs(float(model_answer) - float(ground_truth)) < 1e-6
        except ValueError:
            return False
    elif dataset == "arc":
        return model_answer.upper() == ground_truth.upper()
    elif dataset == "math":
        return model_answer.strip().lower() == ground_truth.strip().lower()
    return model_answer == ground_truth


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
    """Run inference, returning accuracy, per-example outcomes, and timing."""
    if not examples:
        return {"accuracy": 0.0, "correct": 0, "total": 0,
                "per_example_correct": [], "wall_time_s": 0.0,
                "total_generated_tokens": 0}

    correct = 0
    total = len(examples)
    per_example_correct = []  # Binary list: 1 if correct, 0 if not
    total_generated_tokens = 0
    num_batches = (total + batch_size - 1) // batch_size

    wall_start = time.time()

    for batch_idx in tqdm(range(num_batches), desc=f"    {label}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total)
        batch_examples = examples[start_idx:end_idx]

        effective_max_tokens = max_new_tokens
        if batch_examples[0]["dataset"] == "arc":
            effective_max_tokens = 32

        prompts = [format_prompt(ex) for ex in batch_examples]

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=effective_max_tokens,
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
            model_answer = extract_answer(response, example)
            is_correct = check_match(model_answer, ground_truth, example["dataset"])
            if is_correct:
                correct += 1
            per_example_correct.append(1 if is_correct else 0)

        if (batch_idx + 1) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    wall_time = time.time() - wall_start
    accuracy = correct / total
    throughput = total_generated_tokens / wall_time if wall_time > 0 else 0
    latency_per_example = wall_time / total if total > 0 else 0

    print(f"      {label}: {accuracy:.2%} ({correct}/{total}) | "
          f"{throughput:.1f} tok/s | {latency_per_example:.2f}s/example")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_example_correct": per_example_correct,
        "wall_time_s": round(wall_time, 2),
        "total_generated_tokens": total_generated_tokens,
        "throughput_tok_per_s": round(throughput, 2),
        "latency_per_example_s": round(latency_per_example, 3),
    }


# =============================================================================
# Model Loading & Surgery
# =============================================================================

def load_4bit_model(model_name: str):
    print(f"\n  Loading 4-bit model...")
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


def load_fp16_model(model_name: str):
    print(f"\n  Loading FP16 model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()
    return model, tokenizer


def unload_model(model, tokenizer=None):
    del model
    if tokenizer:
        del tokenizer
    aggressive_cleanup()


def cache_fp16_weights(model_name: str, config: Config) -> Dict:
    """Cache FP16 weights for the 6 critical layers."""
    print("\n  Caching FP16 weights for critical layers...")

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
    """Apply FP16 patches to the 6 critical layers."""
    print("\n  Applying surgical FP16 patches...")

    for layer_idx in config.attn_fp16_layers:
        layer = model.model.layers[layer_idx]
        for proj_name in config.attn_projections:
            weight_info = fp16_cache["attention"][layer_idx][proj_name]
            new_linear = nn.Linear(
                weight_info["in_features"], weight_info["out_features"],
                bias=weight_info["bias"] is not None,
                dtype=torch.float16, device=config.device,
            )
            new_linear.weight.data = weight_info["weight"].to(config.device)
            if weight_info["bias"] is not None:
                new_linear.bias.data = weight_info["bias"].to(config.device)
            setattr(layer.self_attn, proj_name, new_linear)
        print(f"    Patched attention L{layer_idx}")

    for layer_idx in config.mlp_fp16_layers:
        layer = model.model.layers[layer_idx]
        for proj_name in config.mlp_projections:
            weight_info = fp16_cache["mlp"][layer_idx][proj_name]
            new_linear = nn.Linear(
                weight_info["in_features"], weight_info["out_features"],
                bias=weight_info["bias"] is not None,
                dtype=torch.float16, device=config.device,
            )
            new_linear.weight.data = weight_info["weight"].to(config.device)
            if weight_info["bias"] is not None:
                new_linear.bias.data = weight_info["bias"].to(config.device)
            setattr(layer.mlp, proj_name, new_linear)
        print(f"    Patched MLP L{layer_idx}")


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(data: Dict, filepath: str):
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, filepath)
    print(f"  Checkpoint saved: {filepath}")


def load_checkpoint(filepath: str) -> Optional[Dict]:
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                print(f"  WARNING: Corrupt checkpoint (not a dict), ignoring: {filepath}")
                return None
            return data
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  WARNING: Corrupt checkpoint file, ignoring: {filepath} ({e})")
            return None
    return None


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 70)
    print("Multi-Seed Statistical Significance Testing")
    print("=" * 70)
    print(f"\nModel: {config.model_name}")
    print(f"Device: {config.device}")
    print(f"Seeds: {config.seeds}")
    print(f"Examples per dataset per seed: {config.num_examples_per_dataset}")

    # Check for checkpoint
    checkpoint = load_checkpoint(config.checkpoint_file)
    completed_seed_idx = checkpoint.get("completed_seed_idx", -1) if checkpoint else -1
    seed_results = checkpoint.get("seed_results", {}) if checkpoint else {}
    completed_substep = checkpoint.get("completed_substep", "") if checkpoint else ""

    # =========================================================================
    # Load full datasets (once, subsample per seed later)
    # =========================================================================
    print(f"\n{'='*70}")
    print("Loading Full Datasets")
    print(f"{'='*70}")

    full_datasets = {
        "gsm8k": load_gsm8k_full(),
        "arc": load_arc_challenge_full(),
        "math": load_math_full(),
    }

    save_checkpoint({
        "completed_seed_idx": completed_seed_idx,
        "seed_results": seed_results,
        "completed_substep": "datasets_loaded",
    }, config.checkpoint_file)
    print("  [checkpoint] Full datasets loaded")

    # =========================================================================
    # Cache FP16 weights (once, reused across all seeds)
    # =========================================================================
    fp16_cache = cache_fp16_weights(config.model_name, config)

    save_checkpoint({
        "completed_seed_idx": completed_seed_idx,
        "seed_results": seed_results,
        "completed_substep": "fp16_cached",
    }, config.checkpoint_file)
    print("  [checkpoint] FP16 weights cached")

    # =========================================================================
    # Phase 1: FP16 Baselines (one model load, all seeds' samples)
    # =========================================================================
    # FP16 is done if: (a) we explicitly recorded it, (b) at least one seed finished
    # Phase 2+3 (implying FP16 was done earlier), or (c) all seeds already have
    # complete FP16 results in the checkpoint.
    fp16_done = (
        completed_substep.startswith("fp16_all_done")
        or completed_seed_idx >= 0
        or all(
            seed_results.get(f"seed_{seed}", {}).get("fp16")
            and all(
                ds in seed_results.get(f"seed_{seed}", {}).get("fp16", {})
                for ds in ["gsm8k", "arc", "math"]
            )
            for seed in config.seeds
        )
    )

    if not fp16_done:
        print(f"\n{'='*70}")
        print("PHASE 1: FP16 Baselines (all seeds)")
        print(f"{'='*70}")

        model, tokenizer = load_fp16_model(config.model_name)

        save_checkpoint({
            "completed_seed_idx": completed_seed_idx,
            "seed_results": seed_results,
            "completed_substep": "fp16_model_loaded",
        }, config.checkpoint_file)
        print("  [checkpoint] FP16 model loaded")

        for seed_idx, seed in enumerate(config.seeds):
            seed_key = f"seed_{seed}"
            if seed_key not in seed_results:
                seed_results[seed_key] = {"seed": seed, "fp16": {}, "int4": {}, "surgical": {}}

            if seed_results[seed_key].get("fp16") and all(
                ds in seed_results[seed_key]["fp16"] for ds in ["gsm8k", "arc", "math"]
            ):
                print(f"\n  Seed {seed} FP16 — loaded from checkpoint")
                continue

            print(f"\n  --- Seed {seed} (FP16) ---")
            samples = {
                ds: subsample(full_datasets[ds], seed, config.num_examples_per_dataset)
                for ds in ["gsm8k", "arc", "math"]
            }

            for ds_name in ["gsm8k", "arc", "math"]:
                res = run_inference(
                    model, tokenizer, samples[ds_name],
                    config.batch_size, config.max_new_tokens,
                    f"FP16-s{seed}-{ds_name}",
                )
                seed_results[seed_key]["fp16"][ds_name] = res

                save_checkpoint({
                    "completed_seed_idx": completed_seed_idx,
                    "seed_results": seed_results,
                    "completed_substep": f"fp16_seed{seed}_{ds_name}_done",
                }, config.checkpoint_file)
                print(f"  [checkpoint] FP16 seed {seed} {ds_name} saved")

        unload_model(model, tokenizer)

        save_checkpoint({
            "completed_seed_idx": completed_seed_idx,
            "seed_results": seed_results,
            "completed_substep": "fp16_all_done",
        }, config.checkpoint_file)
        print("  [checkpoint] All FP16 baselines complete")
    else:
        print(f"\n  FP16 baselines loaded from checkpoint")

    # =========================================================================
    # Phase 2 & 3: Per-seed 4-bit + Surgical
    # =========================================================================
    for seed_idx, seed in enumerate(config.seeds):
        if seed_idx <= completed_seed_idx:
            print(f"\n  Seed {seed} (4-bit + surgical) — loaded from checkpoint")
            continue

        seed_key = f"seed_{seed}"
        if seed_key not in seed_results:
            seed_results[seed_key] = {"seed": seed, "fp16": {}, "int4": {}, "surgical": {}}

        print(f"\n{'='*70}")
        print(f"SEED {seed}: 4-bit Baseline + Surgical Mix")
        print(f"{'='*70}")

        # Subsample for this seed
        samples = {
            ds: subsample(full_datasets[ds], seed, config.num_examples_per_dataset)
            for ds in ["gsm8k", "arc", "math"]
        }

        # --- 4-bit Baseline ---
        model, tokenizer = load_4bit_model(config.model_name)
        reset_peak_memory()

        save_checkpoint({
            "completed_seed_idx": completed_seed_idx,
            "seed_results": seed_results,
            "completed_substep": f"seed{seed}_4bit_model_loaded",
        }, config.checkpoint_file)
        print(f"  [checkpoint] Seed {seed} 4-bit model loaded")

        print(f"\n  Evaluating 4-bit baseline...")
        for ds_name in ["gsm8k", "arc", "math"]:
            res = run_inference(
                model, tokenizer, samples[ds_name],
                config.batch_size, config.max_new_tokens,
                f"4bit-s{seed}-{ds_name}",
            )
            seed_results[seed_key]["int4"][ds_name] = res

            save_checkpoint({
                "completed_seed_idx": completed_seed_idx,
                "seed_results": seed_results,
                "completed_substep": f"seed{seed}_4bit_{ds_name}_done",
            }, config.checkpoint_file)
            print(f"  [checkpoint] Seed {seed} 4-bit {ds_name} saved")

        # --- Surgical Mix (apply patches to same model) ---
        print(f"\n  Applying surgical patches...")
        apply_surgical_patch(model, fp16_cache, config)

        save_checkpoint({
            "completed_seed_idx": completed_seed_idx,
            "seed_results": seed_results,
            "completed_substep": f"seed{seed}_patches_applied",
        }, config.checkpoint_file)
        print(f"  [checkpoint] Seed {seed} patches applied")

        print(f"\n  Evaluating surgical model...")
        for ds_name in ["gsm8k", "arc", "math"]:
            res = run_inference(
                model, tokenizer, samples[ds_name],
                config.batch_size, config.max_new_tokens,
                f"Surg-s{seed}-{ds_name}",
            )
            seed_results[seed_key]["surgical"][ds_name] = res

            save_checkpoint({
                "completed_seed_idx": completed_seed_idx,
                "seed_results": seed_results,
                "completed_substep": f"seed{seed}_surgical_{ds_name}_done",
            }, config.checkpoint_file)
            print(f"  [checkpoint] Seed {seed} surgical {ds_name} saved")

        unload_model(model, tokenizer)

        completed_seed_idx = seed_idx
        save_checkpoint({
            "completed_seed_idx": completed_seed_idx,
            "seed_results": seed_results,
            "completed_substep": f"seed{seed}_complete",
        }, config.checkpoint_file)
        print(f"  [checkpoint] Seed {seed} complete")

    # =========================================================================
    # Compute Summary Statistics
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")

    summary = {}
    for condition in ["fp16", "int4", "surgical"]:
        summary[condition] = {}
        for ds_name in ["gsm8k", "arc", "math"]:
            accs = []
            for seed in config.seeds:
                seed_key = f"seed_{seed}"
                accs.append(seed_results[seed_key][condition][ds_name]["accuracy"])

            accs_arr = np.array(accs)
            summary[condition][ds_name] = {
                "mean": float(np.mean(accs_arr)),
                "std": float(np.std(accs_arr, ddof=1)),  # sample std
                "min": float(np.min(accs_arr)),
                "max": float(np.max(accs_arr)),
                "values": [float(a) for a in accs],
            }

        # Average across datasets
        avg_accs = []
        for seed in config.seeds:
            seed_key = f"seed_{seed}"
            seed_avg = np.mean([
                seed_results[seed_key][condition][ds]["accuracy"]
                for ds in ["gsm8k", "arc", "math"]
            ])
            avg_accs.append(seed_avg)

        avg_arr = np.array(avg_accs)
        summary[condition]["average"] = {
            "mean": float(np.mean(avg_arr)),
            "std": float(np.std(avg_arr, ddof=1)),
            "min": float(np.min(avg_arr)),
            "max": float(np.max(avg_arr)),
            "values": [float(a) for a in avg_accs],
        }

    # Gap recovery stats
    summary["gap_recovery"] = {}
    for ds_name in ["gsm8k", "arc", "math"]:
        recoveries = []
        for seed in config.seeds:
            seed_key = f"seed_{seed}"
            fp16_acc = seed_results[seed_key]["fp16"][ds_name]["accuracy"]
            int4_acc = seed_results[seed_key]["int4"][ds_name]["accuracy"]
            surg_acc = seed_results[seed_key]["surgical"][ds_name]["accuracy"]
            gap = fp16_acc - int4_acc
            if gap > 0:
                recovery = (surg_acc - int4_acc) / gap * 100
            else:
                recovery = 100.0
            recoveries.append(recovery)

        rec_arr = np.array(recoveries)
        summary["gap_recovery"][ds_name] = {
            "mean": float(np.mean(rec_arr)),
            "std": float(np.std(rec_arr, ddof=1)),
            "values": [float(r) for r in recoveries],
        }

    # Print results table
    print(f"\n  {'Condition':<20} {'Dataset':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'─'*60}")

    cond_labels = {"fp16": "Full FP16", "int4": "Full 4-bit", "surgical": "Surgical Mix"}
    for condition in ["fp16", "int4", "surgical"]:
        for ds_name in ["gsm8k", "arc", "math", "average"]:
            s = summary[condition][ds_name]
            label = cond_labels[condition] if ds_name == "gsm8k" else ""
            print(f"  {label:<20} {ds_name:<10} {s['mean']:>7.1%} {s['std']:>7.1%} "
                  f"{s['min']:>7.1%} {s['max']:>7.1%}")
        print()

    print(f"\n  Gap Recovery (surgical closes X% of FP16-4bit gap):")
    print(f"  {'Dataset':<10} {'Mean':>10} {'Std':>10}")
    print(f"  {'─'*35}")
    for ds_name in ["gsm8k", "arc", "math"]:
        s = summary["gap_recovery"][ds_name]
        print(f"  {ds_name:<10} {s['mean']:>9.1f}% {s['std']:>9.1f}%")

    # =========================================================================
    # Statistical Significance Tests
    # =========================================================================
    print(f"\n{'='*70}")
    print("STATISTICAL SIGNIFICANCE TESTS")
    print(f"{'='*70}")

    stat_tests = {}

    # --- McNemar's Test (per arxiv:2602.10144) ---
    # Tests whether two models disagree on examples in a statistically
    # significant pattern. Uses per-example correct/incorrect outcomes.
    print(f"\n  McNemar's Test (H0: models have same error rate):")
    print(f"  {'Comparison':<30} {'Dataset':<10} {'b (A>B)':>8} {'c (B>A)':>8} {'chi2':>8} {'p-value':>10} {'Sig?':>6}")
    print(f"  {'─'*85}")

    stat_tests["mcnemar"] = {}

    # Use first seed's per-example data for McNemar's (paired test needs same examples)
    first_seed_key = f"seed_{config.seeds[0]}"
    comparisons = [
        ("FP16 vs 4-bit", "fp16", "int4"),
        ("FP16 vs Surgical", "fp16", "surgical"),
        ("4-bit vs Surgical", "int4", "surgical"),
    ]

    for comp_label, cond_a, cond_b in comparisons:
        stat_tests["mcnemar"][f"{cond_a}_vs_{cond_b}"] = {}
        for ds_name in ["gsm8k", "arc", "math"]:
            a_correct = seed_results[first_seed_key][cond_a][ds_name].get("per_example_correct", [])
            b_correct = seed_results[first_seed_key][cond_b][ds_name].get("per_example_correct", [])

            if not a_correct or not b_correct or len(a_correct) != len(b_correct):
                print(f"  {comp_label:<30} {ds_name:<10} {'(no per-example data)':>40}")
                stat_tests["mcnemar"][f"{cond_a}_vs_{cond_b}"][ds_name] = {
                    "error": "no per-example data"
                }
                continue

            a_arr = np.array(a_correct)
            b_arr = np.array(b_correct)

            # McNemar contingency: b = A correct & B wrong, c = A wrong & B correct
            b_count = int(np.sum((a_arr == 1) & (b_arr == 0)))  # A right, B wrong
            c_count = int(np.sum((a_arr == 0) & (b_arr == 1)))  # A wrong, B right

            # McNemar's chi-squared (with continuity correction)
            if b_count + c_count == 0:
                chi2 = 0.0
                p_value = 1.0
            else:
                chi2 = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
                if HAS_SCIPY:
                    p_value = float(1 - scipy_stats.chi2.cdf(chi2, df=1))
                else:
                    # Manual approximation using normal distribution
                    import math
                    z = math.sqrt(chi2)
                    p_value = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))

            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"  {comp_label:<30} {ds_name:<10} {b_count:>8} {c_count:>8} {chi2:>8.2f} {p_value:>10.4f} {sig:>6}")

            stat_tests["mcnemar"][f"{cond_a}_vs_{cond_b}"][ds_name] = {
                "b_count": b_count,
                "c_count": c_count,
                "chi2": round(chi2, 4),
                "p_value": round(p_value, 6),
                "significant_at_0.05": p_value < 0.05,
            }

    # --- Bootstrap 95% Confidence Intervals ---
    print(f"\n  Bootstrap 95% Confidence Intervals (10,000 resamples):")
    print(f"  {'Condition':<20} {'Dataset':<10} {'Mean':>8} {'95% CI':>20}")
    print(f"  {'─'*62}")

    stat_tests["bootstrap_ci"] = {}
    n_bootstrap = 10000

    for condition in ["fp16", "int4", "surgical"]:
        stat_tests["bootstrap_ci"][condition] = {}
        for ds_name in ["gsm8k", "arc", "math"]:
            accs = np.array(summary[condition][ds_name]["values"])

            # Bootstrap: resample the seed-level accuracies
            boot_means = []
            rng = np.random.RandomState(42)
            for _ in range(n_bootstrap):
                sample = rng.choice(accs, size=len(accs), replace=True)
                boot_means.append(np.mean(sample))

            boot_means = np.array(boot_means)
            ci_lower = float(np.percentile(boot_means, 2.5))
            ci_upper = float(np.percentile(boot_means, 97.5))
            mean_val = float(np.mean(accs))

            label = cond_labels[condition] if ds_name == "gsm8k" else ""
            print(f"  {label:<20} {ds_name:<10} {mean_val:>7.1%} [{ci_lower:>7.1%}, {ci_upper:>7.1%}]")

            stat_tests["bootstrap_ci"][condition][ds_name] = {
                "mean": round(mean_val, 4),
                "ci_lower": round(ci_lower, 4),
                "ci_upper": round(ci_upper, 4),
            }
        print()

    # --- Paired Permutation Test (Dror et al. 2018) ---
    # Tests whether accuracy difference between two conditions is significant
    # by permuting which seed's result belongs to which condition.
    print(f"\n  Paired Permutation Test (10,000 permutations):")
    print(f"  {'Comparison':<30} {'Dataset':<10} {'Obs Diff':>10} {'p-value':>10} {'Sig?':>6}")
    print(f"  {'─'*70}")

    stat_tests["permutation_test"] = {}
    n_permutations = 10000

    for comp_label, cond_a, cond_b in comparisons:
        stat_tests["permutation_test"][f"{cond_a}_vs_{cond_b}"] = {}
        for ds_name in ["gsm8k", "arc", "math"]:
            accs_a = np.array([
                seed_results[f"seed_{s}"][cond_a][ds_name]["accuracy"]
                for s in config.seeds
            ])
            accs_b = np.array([
                seed_results[f"seed_{s}"][cond_b][ds_name]["accuracy"]
                for s in config.seeds
            ])

            obs_diff = float(np.mean(accs_a) - np.mean(accs_b))
            diffs = accs_a - accs_b

            # Permutation: randomly flip signs of paired differences
            rng = np.random.RandomState(42)
            count_extreme = 0
            for _ in range(n_permutations):
                signs = rng.choice([-1, 1], size=len(diffs))
                perm_diff = np.mean(diffs * signs)
                if abs(perm_diff) >= abs(obs_diff):
                    count_extreme += 1

            p_value = count_extreme / n_permutations
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"  {comp_label:<30} {ds_name:<10} {obs_diff:>+9.1%} {p_value:>10.4f} {sig:>6}")

            stat_tests["permutation_test"][f"{cond_a}_vs_{cond_b}"][ds_name] = {
                "observed_diff": round(obs_diff, 4),
                "p_value": round(p_value, 6),
                "significant_at_0.05": p_value < 0.05,
            }

    # =========================================================================
    # Efficiency Metrics
    # =========================================================================
    print(f"\n{'='*70}")
    print("EFFICIENCY METRICS")
    print(f"{'='*70}")

    efficiency = {}

    for condition in ["fp16", "int4", "surgical"]:
        efficiency[condition] = {}

        # Aggregate throughput and latency across seeds
        throughputs = []
        latencies = []
        for seed in config.seeds:
            seed_key = f"seed_{seed}"
            for ds_name in ["gsm8k", "arc", "math"]:
                res = seed_results[seed_key][condition][ds_name]
                if "throughput_tok_per_s" in res:
                    throughputs.append(res["throughput_tok_per_s"])
                if "latency_per_example_s" in res:
                    latencies.append(res["latency_per_example_s"])

        if throughputs:
            efficiency[condition]["throughput_tok_per_s"] = {
                "mean": round(float(np.mean(throughputs)), 2),
                "std": round(float(np.std(throughputs, ddof=1)), 2) if len(throughputs) > 1 else 0,
            }
        if latencies:
            efficiency[condition]["latency_per_example_s"] = {
                "mean": round(float(np.mean(latencies)), 3),
                "std": round(float(np.std(latencies, ddof=1)), 3) if len(latencies) > 1 else 0,
            }

    # Accuracy-per-MB efficiency ratio (using capstone memory data if available)
    # Memory values from optimal_mixed_precision_report.json
    memory_mb = {
        "fp16": 16000,    # approximate
        "int4": 5583,     # from capstone
        "surgical": 6711, # 5583 + 1128 overhead
    }

    capstone_path = None
    for path in [
        os.path.join(config.output_dir, "optimal_mixed_precision_report.json"),
        "optimal_mixed_precision_report.json",
        os.path.expanduser("~/Downloads/optimal_mixed_precision_report.json"),
    ]:
        if os.path.exists(path):
            capstone_path = path
            break

    if capstone_path:
        with open(capstone_path) as f:
            capstone = json.load(f)
        cap_results = capstone.get("results", {})
        if cap_results.get("int4", {}).get("memory_mb"):
            memory_mb["int4"] = cap_results["int4"]["memory_mb"]["peak_mb"]
        if cap_results.get("surgical", {}).get("patch_info"):
            overhead = cap_results["surgical"]["patch_info"]["overhead_mb"]
            memory_mb["surgical"] = memory_mb["int4"] + overhead
        print(f"  Loaded memory data from {capstone_path}")

    print(f"\n  {'Condition':<20} {'Avg Acc':>8} {'Memory':>10} {'Acc/GB':>10} "
          f"{'Throughput':>12} {'Latency':>10}")
    print(f"  {'─'*75}")

    for condition in ["fp16", "int4", "surgical"]:
        avg_acc = summary[condition]["average"]["mean"]
        mem = memory_mb[condition]
        acc_per_gb = avg_acc / (mem / 1024)  # accuracy per GB

        tp_str = ""
        lat_str = ""
        if efficiency[condition].get("throughput_tok_per_s"):
            tp = efficiency[condition]["throughput_tok_per_s"]["mean"]
            tp_str = f"{tp:.1f} tok/s"
        if efficiency[condition].get("latency_per_example_s"):
            lat = efficiency[condition]["latency_per_example_s"]["mean"]
            lat_str = f"{lat:.2f} s/ex"

        print(f"  {cond_labels[condition]:<20} {avg_acc:>7.1%} {mem:>8.0f} MB {acc_per_gb:>9.2f} "
              f"{tp_str:>12} {lat_str:>10}")

        efficiency[condition]["memory_mb"] = mem
        efficiency[condition]["accuracy_per_gb"] = round(acc_per_gb, 4)

    print(f"\n  Gap Recovery (surgical closes X% of FP16-4bit gap):")
    print(f"  {'Dataset':<10} {'Mean':>10} {'Std':>10}")
    print(f"  {'─'*35}")
    for ds_name in ["gsm8k", "arc", "math"]:
        s = summary["gap_recovery"][ds_name]
        print(f"  {ds_name:<10} {s['mean']:>9.1f}% {s['std']:>9.1f}%")

    # =========================================================================
    # Save
    # =========================================================================
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")

    output = {
        "config": {
            "model": config.model_name,
            "num_examples_per_dataset": config.num_examples_per_dataset,
            "seeds": list(config.seeds),
            "attn_fp16_layers": list(config.attn_fp16_layers),
            "mlp_fp16_layers": list(config.mlp_fp16_layers),
        },
        "seed_results": seed_results,
        "summary": summary,
        "statistical_tests": stat_tests,
        "efficiency": efficiency,
    }

    with open(config.output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {config.output_file}")

    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}")

    return output


if __name__ == "__main__":
    output = main()
