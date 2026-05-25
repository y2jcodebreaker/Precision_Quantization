"""
Patch Count Ablation: How Many Critical Layers Do You Actually Need?
====================================================================

Tests the diminishing returns curve by incrementally adding the 6 critical
layers in order of importance (from cross-task interference analysis).

Layer importance ranking:
  1. MLP L6   — ARC-optimal, highest cross-task spillover (helps ALL tasks)
  2. MLP L7   — ARC-optimal companion
  3. Attn L13 — GSM8K arithmetic bottleneck
  4. Attn L14 — GSM8K arithmetic companion
  5. MLP L1   — MATH bookend (early feature extraction)
  6. MLP L31  — MATH bookend (output formation)

Tests 7 conditions (0 through 6 layers patched), evaluating on GSM8K,
ARC-Challenge, and MATH (500 examples each) for every condition.

Output: results/patch_count_ablation_report.json

Authentication:
---------------
Set HF_TOKEN environment variable for model access.
"""

import json
import os
import re
import gc
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

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
    output_file: str = os.path.join("results", "patch_count_ablation_report.json")
    checkpoint_file: str = os.path.join("results", "patch_count_ablation_checkpoint.json")

    num_examples_per_dataset: int = 500

    # Layers in order of importance (added incrementally)
    # Each entry: (component, layer_idx)
    layer_order: Tuple[Tuple[str, int], ...] = (
        ("mlp", 6),       # ARC-optimal, highest cross-task spillover
        ("mlp", 7),       # ARC-optimal companion
        ("attention", 13), # GSM8K arithmetic bottleneck
        ("attention", 14), # GSM8K arithmetic companion
        ("mlp", 1),       # MATH bookend (early features)
        ("mlp", 31),      # MATH bookend (output formation)
    )

    attn_projections: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    mlp_projections: Tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")

    batch_size: int = 8
    max_new_tokens: int = 512
    seed: int = 42
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
# Dataset Loading
# =============================================================================

def load_gsm8k(num_examples: int) -> List[Dict]:
    print(f"  Loading GSM8K ({num_examples} examples)...")
    dataset = load_dataset("gsm8k", "main", split="test")
    examples = []
    for i, item in enumerate(dataset):
        if i >= num_examples:
            break
        examples.append({
            "index": i,
            "question": item["question"],
            "answer": item["answer"],
            "dataset": "gsm8k",
        })
    return examples


def load_arc_challenge(num_examples: int) -> List[Dict]:
    print(f"  Loading ARC-Challenge ({num_examples} examples)...")
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    examples = []
    for i, item in enumerate(dataset):
        if i >= num_examples:
            break
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
    return examples


def load_math(num_examples: int) -> List[Dict]:
    print(f"  Loading MATH ({num_examples} examples)...")
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
        if i >= num_examples:
            break
        examples.append({
            "index": i,
            "problem": item["problem"],
            "solution": item["solution"],
            "dataset": "math",
        })
    return examples


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
    if not examples:
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    correct = 0
    total = len(examples)
    num_batches = (total + batch_size - 1) // batch_size

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
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            ground_truth = extract_ground_truth(example)
            model_answer = extract_answer(response, example)
            if check_match(model_answer, ground_truth, example["dataset"]):
                correct += 1

        if (batch_idx + 1) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    accuracy = correct / total
    print(f"      {label}: {accuracy:.2%} ({correct}/{total})")
    return {"accuracy": accuracy, "correct": correct, "total": total}


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


def unload_model(model, tokenizer=None):
    del model
    if tokenizer:
        del tokenizer
    aggressive_cleanup()


def cache_all_fp16_weights(model_name: str, config: Config) -> Dict:
    """Cache FP16 weights for all 6 critical layers."""
    print("\n  Caching FP16 weights for all critical layers...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map="cpu", low_cpu_mem_usage=True,
    )

    cache = {"attention": {}, "mlp": {}}

    for component, layer_idx in config.layer_order:
        layer = model.model.layers[layer_idx]

        if component == "attention" and layer_idx not in cache["attention"]:
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

        elif component == "mlp" and layer_idx not in cache["mlp"]:
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


def apply_patches(model, fp16_cache: Dict, layers_to_patch: List[Tuple[str, int]], config: Config) -> Dict:
    """Apply FP16 patches for the specified subset of layers."""
    fp16_param_count = 0

    for component, layer_idx in layers_to_patch:
        layer = model.model.layers[layer_idx]

        if component == "attention":
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
                fp16_param_count += weight_info["in_features"] * weight_info["out_features"]
            print(f"    Patched attention L{layer_idx}")

        elif component == "mlp":
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
                fp16_param_count += weight_info["in_features"] * weight_info["out_features"]
            print(f"    Patched MLP L{layer_idx}")

    fp16_bytes = fp16_param_count * 2
    nf4_bytes = fp16_param_count * 0.5
    overhead_mb = (fp16_bytes - nf4_bytes) / 1024 / 1024

    return {
        "fp16_param_count": fp16_param_count,
        "overhead_mb": overhead_mb,
    }


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
    print("Patch Count Ablation: Diminishing Returns Curve")
    print("=" * 70)
    print(f"\nModel: {config.model_name}")
    print(f"Device: {config.device}")
    print(f"\nLayer addition order (by importance):")
    for i, (component, layer_idx) in enumerate(config.layer_order):
        print(f"  {i+1}. {component.upper()} L{layer_idx}")

    torch.manual_seed(config.seed)

    # Check for checkpoint
    checkpoint = load_checkpoint(config.checkpoint_file)
    completed_condition = checkpoint.get("completed_condition", -1) if checkpoint else -1
    results = checkpoint.get("results", {}) if checkpoint else {}

    # =========================================================================
    # Load Datasets
    # =========================================================================
    print(f"\n{'='*70}")
    print("Loading Datasets")
    print(f"{'='*70}")

    all_examples = {
        "gsm8k": load_gsm8k(config.num_examples_per_dataset),
        "arc": load_arc_challenge(config.num_examples_per_dataset),
        "math": load_math(config.num_examples_per_dataset),
    }

    for ds, examples in all_examples.items():
        print(f"  {ds}: {len(examples)} examples")

    save_checkpoint({
        "completed_condition": completed_condition,
        "results": results,
        "substep": "datasets_loaded",
    }, config.checkpoint_file)
    print("  [checkpoint] Datasets loaded")

    # =========================================================================
    # Cache FP16 weights
    # =========================================================================
    fp16_cache = cache_all_fp16_weights(config.model_name, config)

    save_checkpoint({
        "completed_condition": completed_condition,
        "results": results,
        "substep": "fp16_weights_cached",
    }, config.checkpoint_file)
    print("  [checkpoint] FP16 weights cached")

    # =========================================================================
    # Run 7 conditions (0 through 6 layers)
    # =========================================================================
    num_conditions = len(config.layer_order) + 1  # 0 through 6

    for cond_idx in range(num_conditions):
        if cond_idx <= completed_condition:
            print(f"\n  Condition {cond_idx} ({cond_idx} layers) — loaded from checkpoint")
            continue

        layers_to_patch = list(config.layer_order[:cond_idx])
        layer_desc = "pure 4-bit" if cond_idx == 0 else ", ".join(
            f"{c[0]}L{c[1]}" for c in layers_to_patch
        )

        print(f"\n{'='*70}")
        print(f"CONDITION {cond_idx}: {cond_idx} layers patched")
        print(f"  Layers: {layer_desc}")
        print(f"{'='*70}")

        # Load fresh 4-bit model
        model, tokenizer = load_4bit_model(config.model_name)
        reset_peak_memory()

        save_checkpoint({
            "completed_condition": completed_condition,
            "results": results,
            "substep": f"cond{cond_idx}_model_loaded",
        }, config.checkpoint_file)
        print(f"  [checkpoint] Condition {cond_idx} model loaded")

        # Apply patches (if any)
        patch_info = None
        if cond_idx > 0:
            print(f"\n  Applying {cond_idx} FP16 patches...")
            patch_info = apply_patches(model, fp16_cache, layers_to_patch, config)
            print(f"    FP16 params: {patch_info['fp16_param_count']:,}")
            print(f"    Overhead: {patch_info['overhead_mb']:.1f} MB")

            save_checkpoint({
                "completed_condition": completed_condition,
                "results": results,
                "substep": f"cond{cond_idx}_patches_applied",
            }, config.checkpoint_file)
            print(f"  [checkpoint] Condition {cond_idx} patches applied")

        # Evaluate all datasets
        cond_key = f"layers_{cond_idx}"
        results[cond_key] = {
            "num_layers": cond_idx,
            "layers_patched": [(c, l) for c, l in layers_to_patch],
            "patch_info": patch_info,
        }

        for ds_name, examples in all_examples.items():
            res = run_inference(
                model, tokenizer, examples,
                config.batch_size, config.max_new_tokens,
                f"C{cond_idx}-{ds_name}",
            )
            results[cond_key][ds_name] = res

            save_checkpoint({
                "completed_condition": completed_condition,
                "results": results,
                "substep": f"cond{cond_idx}_{ds_name}_done",
            }, config.checkpoint_file)
            print(f"  [checkpoint] Condition {cond_idx} {ds_name} saved")

        results[cond_key]["memory_mb"] = get_memory_stats()

        # Unload
        unload_model(model, tokenizer)

        completed_condition = cond_idx
        save_checkpoint({
            "completed_condition": completed_condition,
            "results": results,
            "substep": f"cond{cond_idx}_complete",
        }, config.checkpoint_file)
        print(f"  [checkpoint] Condition {cond_idx} complete")

    # =========================================================================
    # RESULTS
    # =========================================================================
    print(f"\n{'='*70}")
    print("RESULTS: Patch Count Ablation")
    print(f"{'='*70}")

    # Accuracy table
    print(f"\n  {'Layers':<8} {'Patched':<35} {'GSM8K':>8} {'ARC':>8} {'MATH':>8} {'Avg':>8}")
    print(f"  {'─'*80}")

    baseline_accs = {}
    for cond_idx in range(num_conditions):
        cond_key = f"layers_{cond_idx}"
        r = results[cond_key]
        gsm = r["gsm8k"]["accuracy"]
        arc = r["arc"]["accuracy"]
        math = r["math"]["accuracy"]
        avg = (gsm + arc + math) / 3

        if cond_idx == 0:
            baseline_accs = {"gsm8k": gsm, "arc": arc, "math": math, "avg": avg}

        layers_desc = "none (4-bit)" if cond_idx == 0 else ", ".join(
            f"{c}L{l}" for c, l in r["layers_patched"]
        )
        # Truncate if too long
        if len(layers_desc) > 33:
            layers_desc = layers_desc[:30] + "..."

        print(f"  {cond_idx:<8} {layers_desc:<35} {gsm:>7.1%} {arc:>7.1%} {math:>7.1%} {avg:>7.1%}")

    # Delta from baseline
    print(f"\n  Delta from pure 4-bit baseline:")
    print(f"  {'Layers':<8} {'GSM8K':>8} {'ARC':>8} {'MATH':>8} {'Avg':>8}")
    print(f"  {'─'*40}")

    for cond_idx in range(num_conditions):
        cond_key = f"layers_{cond_idx}"
        r = results[cond_key]
        d_gsm = r["gsm8k"]["accuracy"] - baseline_accs["gsm8k"]
        d_arc = r["arc"]["accuracy"] - baseline_accs["arc"]
        d_math = r["math"]["accuracy"] - baseline_accs["math"]
        d_avg = (d_gsm + d_arc + d_math) / 3
        print(f"  {cond_idx:<8} {d_gsm:>+7.1%} {d_arc:>+7.1%} {d_math:>+7.1%} {d_avg:>+7.1%}")

    # Marginal gains
    print(f"\n  Marginal gain per layer added:")
    print(f"  {'Added':<20} {'GSM8K':>8} {'ARC':>8} {'MATH':>8} {'Avg':>8}")
    print(f"  {'─'*55}")

    for cond_idx in range(1, num_conditions):
        cond_key = f"layers_{cond_idx}"
        prev_key = f"layers_{cond_idx - 1}"
        r = results[cond_key]
        p = results[prev_key]
        component, layer_idx = config.layer_order[cond_idx - 1]
        added = f"+{component}L{layer_idx}"

        m_gsm = r["gsm8k"]["accuracy"] - p["gsm8k"]["accuracy"]
        m_arc = r["arc"]["accuracy"] - p["arc"]["accuracy"]
        m_math = r["math"]["accuracy"] - p["math"]["accuracy"]
        m_avg = (m_gsm + m_arc + m_math) / 3
        print(f"  {added:<20} {m_gsm:>+7.1%} {m_arc:>+7.1%} {m_math:>+7.1%} {m_avg:>+7.1%}")

    # =========================================================================
    # Build diminishing returns curve data
    # =========================================================================
    curve = []
    for cond_idx in range(num_conditions):
        cond_key = f"layers_{cond_idx}"
        r = results[cond_key]
        gsm = r["gsm8k"]["accuracy"]
        arc = r["arc"]["accuracy"]
        math_acc = r["math"]["accuracy"]
        avg = (gsm + arc + math_acc) / 3
        overhead = r["patch_info"]["overhead_mb"] if r.get("patch_info") else 0

        curve.append({
            "num_layers": cond_idx,
            "layers": [(c, l) for c, l in r["layers_patched"]] if cond_idx > 0 else [],
            "gsm8k": gsm,
            "arc": arc,
            "math": math_acc,
            "average": avg,
            "overhead_mb": overhead,
        })

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
            "seed": config.seed,
            "layer_order": [(c, l) for c, l in config.layer_order],
            "layer_order_rationale": [
                "MLP L6: ARC-optimal, highest cross-task spillover",
                "MLP L7: ARC-optimal companion",
                "Attn L13: GSM8K arithmetic bottleneck",
                "Attn L14: GSM8K arithmetic companion",
                "MLP L1: MATH bookend (early features)",
                "MLP L31: MATH bookend (output formation)",
            ],
        },
        "results": results,
        "diminishing_returns_curve": curve,
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
