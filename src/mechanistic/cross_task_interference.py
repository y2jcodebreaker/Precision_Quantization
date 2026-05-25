"""
Cross-Task Interference Matrix
================================

Tests whether task-specific precision-sensitive circuits are functionally
independent or share computational substrates.

Method:
- Apply 4 patch configurations to the 4-bit model:
  1. GSM8K-optimal: FP16 attention in L13 + L14
  2. ARC-optimal: FP16 MLP in L6 + L7
  3. MATH-optimal: FP16 MLP in L1 + L31
  4. All combined: all of the above simultaneously
- Evaluate each configuration on ALL 3 datasets (flipped examples only)
- Build a 4×3 interference matrix

If task-optimal patches are neutral for other tasks → functionally independent circuits.
If they help/hurt other tasks → shared computational substrates.

Datasets:
- GSM8K: Math word problems
- ARC-Challenge: Science reasoning
- MATH: Competition mathematics

Output: cross_task_interference_report.json

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
    """Configuration for cross-task interference experiment."""
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_file: str = "cross_task_interference_report.json"
    checkpoint_file: str = "cross_task_interference_checkpoint.json"

    # Pre-computed results from bidirectional flip analysis (skip Phase 1 if found)
    flip_report_file: str = "bidirectional_flip_report.json"

    num_examples_per_dataset: int = 500
    num_layers: int = 32

    # Patch configurations derived from layer sensitivity profiling
    # GSM8K-optimal: attention in layers 13, 14 (top attention layers for GSM8K)
    gsm8k_patch_layers: Tuple[int, ...] = (13, 14)
    gsm8k_patch_component: str = "attention"

    # ARC-optimal: MLP in layers 6, 7 (top MLP layers for ARC)
    arc_patch_layers: Tuple[int, ...] = (6, 7)
    arc_patch_component: str = "mlp"

    # MATH-optimal: MLP in layers 1, 31 (bookend pattern)
    math_patch_layers: Tuple[int, ...] = (1, 31)
    math_patch_component: str = "mlp"

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
    """Aggressively clear memory."""
    gc.collect()
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# Dataset Loading (same as profiler)
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
# Model Loading
# =============================================================================

def load_fp16_model(model_name: str):
    print(f"\nLoading FP16 model...")
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
    print(f"\nLoading 4-bit model...")
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


# =============================================================================
# FP16 Weight Caching (selective — only cache layers we need)
# =============================================================================

def cache_needed_fp16_weights(model_name: str, config: Config) -> Dict:
    """
    Cache FP16 weights only for the layers we actually need to patch.
    Much more memory-efficient than caching all 32 layers.
    """
    print("\nCaching FP16 weights for target layers...")

    # Determine unique layers needed
    all_attn_layers = set()
    all_mlp_layers = set()

    if config.gsm8k_patch_component == "attention":
        all_attn_layers.update(config.gsm8k_patch_layers)
    else:
        all_mlp_layers.update(config.gsm8k_patch_layers)

    if config.arc_patch_component == "attention":
        all_attn_layers.update(config.arc_patch_layers)
    else:
        all_mlp_layers.update(config.arc_patch_layers)

    if config.math_patch_component == "attention":
        all_attn_layers.update(config.math_patch_layers)
    else:
        all_mlp_layers.update(config.math_patch_layers)

    print(f"  Attention layers to cache: {sorted(all_attn_layers)}")
    print(f"  MLP layers to cache: {sorted(all_mlp_layers)}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map="cpu", low_cpu_mem_usage=True,
    )

    cache = {"attention": {}, "mlp": {}}

    for layer_idx in sorted(all_attn_layers):
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
        print(f"  Cached attention L{layer_idx}")

    for layer_idx in sorted(all_mlp_layers):
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
        print(f"  Cached MLP L{layer_idx}")

    del model
    aggressive_cleanup()
    return cache


# =============================================================================
# Multi-Layer Transplant Operations
# =============================================================================

def apply_patch(
    model,
    fp16_cache: Dict,
    patch_layers: Tuple[int, ...],
    patch_component: str,
    config: Config,
) -> Dict[int, Dict[str, Any]]:
    """
    Apply FP16 patch to multiple layers at once.
    Returns dict of {layer_idx: {proj_name: original_module}} for restoration.
    """
    all_originals = {}

    if patch_component == "attention":
        projections = config.attn_projections
        for layer_idx in patch_layers:
            layer = model.model.layers[layer_idx]
            originals = {}
            for proj_name in projections:
                originals[proj_name] = getattr(layer.self_attn, proj_name)
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
            all_originals[layer_idx] = originals

    elif patch_component == "mlp":
        projections = config.mlp_projections
        for layer_idx in patch_layers:
            layer = model.model.layers[layer_idx]
            originals = {}
            for proj_name in projections:
                originals[proj_name] = getattr(layer.mlp, proj_name)
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
            all_originals[layer_idx] = originals

    return all_originals


def restore_patch(
    model,
    all_originals: Dict[int, Dict[str, Any]],
    patch_component: str,
):
    """Restore original 4-bit modules after a patch test."""
    for layer_idx, originals in all_originals.items():
        layer = model.model.layers[layer_idx]
        for proj_name, original_module in originals.items():
            if patch_component == "attention":
                current = getattr(layer.self_attn, proj_name)
                del current
                setattr(layer.self_attn, proj_name, original_module)
            elif patch_component == "mlp":
                current = getattr(layer.mlp, proj_name)
                del current
                setattr(layer.mlp, proj_name, original_module)

    aggressive_cleanup()


def apply_combined_patch(
    model,
    fp16_cache: Dict,
    config: Config,
) -> List[Tuple[Dict, str]]:
    """
    Apply ALL patches simultaneously (GSM8K + ARC + MATH optimal).
    Returns list of (originals_dict, component) for restoration.
    """
    restore_list = []

    originals = apply_patch(
        model, fp16_cache,
        config.gsm8k_patch_layers, config.gsm8k_patch_component, config,
    )
    restore_list.append((originals, config.gsm8k_patch_component))

    originals = apply_patch(
        model, fp16_cache,
        config.arc_patch_layers, config.arc_patch_component, config,
    )
    restore_list.append((originals, config.arc_patch_component))

    originals = apply_patch(
        model, fp16_cache,
        config.math_patch_layers, config.math_patch_component, config,
    )
    restore_list.append((originals, config.math_patch_component))

    return restore_list


def restore_combined_patch(model, restore_list: List[Tuple[Dict, str]]):
    """Restore all patches from a combined application."""
    for originals, component in restore_list:
        restore_patch(model, originals, component)


# =============================================================================
# Inference
# =============================================================================

def run_inference_on_subset(
    model,
    tokenizer,
    examples: List[Dict],
    batch_size: int,
    max_new_tokens: int,
) -> List[Dict]:
    """Run inference on a subset of examples."""
    if not examples:
        return []

    results = []
    num_batches = (len(examples) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(examples))
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
            is_correct = check_match(model_answer, ground_truth, example["dataset"])

            results.append({
                "index": example["index"],
                "dataset": example["dataset"],
                "correct": is_correct,
            })

        if (batch_idx + 1) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def evaluate_all_datasets(
    model,
    tokenizer,
    flipped_examples: Dict[str, List[Dict]],
    all_examples: Dict[str, List[Dict]],
    config: Config,
    condition_name: str,
) -> Dict[str, Dict]:
    """
    Evaluate on all 3 datasets.
    Returns {dataset_name: {"recovery_rate": float, "accuracy": float, ...}}
    """
    results = {}

    for ds_name in ["gsm8k", "arc", "math"]:
        flipped = flipped_examples[ds_name]
        full = all_examples[ds_name]

        # Recovery rate: test on flipped examples
        if flipped:
            flipped_res = run_inference_on_subset(
                model, tokenizer, flipped,
                config.batch_size, config.max_new_tokens,
            )
            recovered = sum(1 for r in flipped_res if r["correct"])
            recovery_rate = recovered / len(flipped)
        else:
            recovered = 0
            recovery_rate = 0.0

        # Full accuracy: test on all examples
        full_res = run_inference_on_subset(
            model, tokenizer, full,
            config.batch_size, config.max_new_tokens,
        )
        full_correct = sum(1 for r in full_res if r["correct"])
        full_accuracy = full_correct / len(full)

        results[ds_name] = {
            "recovery_rate": recovery_rate,
            "recovered_count": recovered,
            "flipped_total": len(flipped),
            "full_accuracy": full_accuracy,
            "full_correct": full_correct,
            "full_total": len(full),
        }

        print(f"      {ds_name}: Recovery={recovery_rate:.1%} ({recovered}/{len(flipped)}), "
              f"Accuracy={full_accuracy:.1%} ({full_correct}/{len(full)})")

    return results


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(data: Dict, filepath: str):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Checkpoint saved: {filepath}")


def load_checkpoint(filepath: str) -> Optional[Dict]:
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        print(f"  Loaded checkpoint: {filepath}")
        return data
    return None


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    config = Config()

    print("=" * 70)
    print("Cross-Task Interference Matrix")
    print("=" * 70)
    print(f"\nModel: {config.model_name}")
    print(f"Device: {config.device}")
    print(f"\nPatch Configurations:")
    print(f"  GSM8K-optimal: {config.gsm8k_patch_component} L{list(config.gsm8k_patch_layers)}")
    print(f"  ARC-optimal:   {config.arc_patch_component} L{list(config.arc_patch_layers)}")
    print(f"  MATH-optimal:  {config.math_patch_component} L{list(config.math_patch_layers)}")

    torch.manual_seed(config.seed)

    # Check for checkpoint
    checkpoint = load_checkpoint(config.checkpoint_file)
    resumed_phase = checkpoint.get("completed_phase", 0) if checkpoint else 0

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

    # =========================================================================
    # Phase 1: Baselines & flipped examples (try to load from prior results)
    # =========================================================================
    flip_report_loaded = False

    if resumed_phase < 1 and os.path.exists(config.flip_report_file):
        print(f"\n{'='*70}")
        print("PHASE 1: Loading baselines from bidirectional_flip_report.json")
        print(f"{'='*70}")

        with open(config.flip_report_file, "r") as f:
            flip_report = json.load(f)

        baselines = {}
        flipped_indices = {}

        for ds_name in ["gsm8k", "arc", "math"]:
            a = flip_report["analyses"][ds_name]
            baselines[ds_name] = {
                "fp16_accuracy": a["fp16_accuracy"],
                "int4_accuracy": a["int4_accuracy"],
                "accuracy_drop": a["accuracy_drop"],
            }
            flipped_indices[ds_name] = a["forward_flip_indices"]
            print(f"  {ds_name}: FP16={a['fp16_accuracy']:.1%}, "
                  f"4-bit={a['int4_accuracy']:.1%}, "
                  f"flipped={len(a['forward_flip_indices'])}")

        flip_report_loaded = True
        resumed_phase = 1  # Mark phase 1 as done

        save_checkpoint({
            "completed_phase": 1,
            "baselines": baselines,
            "flipped_indices": flipped_indices,
        }, config.checkpoint_file)

    elif resumed_phase < 1:
        print(f"\n{'='*70}")
        print("PHASE 1: Baselines & Flipped Example Identification")
        print(f"  (No prior flip report found — running full evaluation)")
        print(f"{'='*70}")

        # FP16 baseline
        model, tokenizer = load_fp16_model(config.model_name)

        fp16_results = {}
        for ds_name, examples in all_examples.items():
            print(f"\n  FP16 — {ds_name}...")
            res = run_inference_on_subset(
                model, tokenizer, examples,
                config.batch_size, config.max_new_tokens,
            )
            fp16_results[ds_name] = res
            acc = sum(1 for r in res if r["correct"]) / len(res)
            print(f"    Accuracy: {acc:.2%}")

        unload_model(model, tokenizer)

        # 4-bit baseline
        model, tokenizer = load_4bit_model(config.model_name)

        int4_results = {}
        baselines = {}
        flipped_indices = {}

        for ds_name, examples in all_examples.items():
            print(f"\n  4-bit — {ds_name}...")
            res = run_inference_on_subset(
                model, tokenizer, examples,
                config.batch_size, config.max_new_tokens,
            )
            int4_results[ds_name] = res

            fp16_acc = sum(1 for r in fp16_results[ds_name] if r["correct"]) / len(fp16_results[ds_name])
            int4_acc = sum(1 for r in res if r["correct"]) / len(res)

            baselines[ds_name] = {
                "fp16_accuracy": fp16_acc,
                "int4_accuracy": int4_acc,
                "accuracy_drop": fp16_acc - int4_acc,
            }

            # Find flipped indices
            fp16_by_idx = {r["index"]: r for r in fp16_results[ds_name]}
            int4_by_idx = {r["index"]: r for r in res}
            flipped = [
                idx for idx in fp16_by_idx
                if fp16_by_idx[idx]["correct"] and not int4_by_idx[idx]["correct"]
            ]
            flipped_indices[ds_name] = flipped

            print(f"    Accuracy: {int4_acc:.2%} (drop: {fp16_acc - int4_acc:.2%})")
            print(f"    Flipped examples: {len(flipped)}")

        unload_model(model, tokenizer)

        save_checkpoint({
            "completed_phase": 1,
            "baselines": baselines,
            "flipped_indices": flipped_indices,
        }, config.checkpoint_file)
    else:
        print(f"\n  Skipping Phase 1 (loaded from checkpoint)")
        baselines = checkpoint["baselines"]
        flipped_indices = checkpoint["flipped_indices"]

    # Build flipped example subsets
    flipped_examples = {}
    for ds_name in all_examples:
        idx_set = set(flipped_indices[ds_name])
        flipped_examples[ds_name] = [
            ex for ex in all_examples[ds_name] if ex["index"] in idx_set
        ]
        print(f"  {ds_name} flipped: {len(flipped_examples[ds_name])}")

    # =========================================================================
    # Phase 2: Cache FP16 weights & run interference tests
    # =========================================================================
    if resumed_phase < 2:
        print(f"\n{'='*70}")
        print("PHASE 2: Cache FP16 Weights")
        print(f"{'='*70}")

        fp16_cache = cache_needed_fp16_weights(config.model_name, config)

        print(f"\n{'='*70}")
        print("PHASE 3: Interference Matrix Tests")
        print(f"{'='*70}")

        model, tokenizer = load_4bit_model(config.model_name)

        interference_matrix = {}

        # --- Test 1: 4-bit baseline (no patch) ---
        print(f"\n  [1/5] Condition: 4-bit Baseline (no patch)")
        interference_matrix["4bit_baseline"] = evaluate_all_datasets(
            model, tokenizer, flipped_examples, all_examples, config,
            "4-bit Baseline",
        )

        # --- Test 2: GSM8K-optimal patch ---
        print(f"\n  [2/5] Condition: GSM8K-optimal ({config.gsm8k_patch_component} L{list(config.gsm8k_patch_layers)})")
        originals = apply_patch(
            model, fp16_cache,
            config.gsm8k_patch_layers, config.gsm8k_patch_component, config,
        )
        interference_matrix["gsm8k_optimal"] = evaluate_all_datasets(
            model, tokenizer, flipped_examples, all_examples, config,
            "GSM8K-optimal",
        )
        restore_patch(model, originals, config.gsm8k_patch_component)

        # Save intermediate checkpoint
        save_checkpoint({
            "completed_phase": 1,
            "baselines": baselines,
            "flipped_indices": flipped_indices,
            "partial_matrix": interference_matrix,
        }, config.checkpoint_file)

        # --- Test 3: ARC-optimal patch ---
        print(f"\n  [3/5] Condition: ARC-optimal ({config.arc_patch_component} L{list(config.arc_patch_layers)})")
        originals = apply_patch(
            model, fp16_cache,
            config.arc_patch_layers, config.arc_patch_component, config,
        )
        interference_matrix["arc_optimal"] = evaluate_all_datasets(
            model, tokenizer, flipped_examples, all_examples, config,
            "ARC-optimal",
        )
        restore_patch(model, originals, config.arc_patch_component)

        # --- Test 4: MATH-optimal patch ---
        print(f"\n  [4/5] Condition: MATH-optimal ({config.math_patch_component} L{list(config.math_patch_layers)})")
        originals = apply_patch(
            model, fp16_cache,
            config.math_patch_layers, config.math_patch_component, config,
        )
        interference_matrix["math_optimal"] = evaluate_all_datasets(
            model, tokenizer, flipped_examples, all_examples, config,
            "MATH-optimal",
        )
        restore_patch(model, originals, config.math_patch_component)

        # Save intermediate checkpoint
        save_checkpoint({
            "completed_phase": 1,
            "baselines": baselines,
            "flipped_indices": flipped_indices,
            "partial_matrix": interference_matrix,
        }, config.checkpoint_file)

        # --- Test 5: All combined ---
        print(f"\n  [5/5] Condition: All Combined")
        restore_list = apply_combined_patch(model, fp16_cache, config)
        interference_matrix["all_combined"] = evaluate_all_datasets(
            model, tokenizer, flipped_examples, all_examples, config,
            "All Combined",
        )
        restore_combined_patch(model, restore_list)

        unload_model(model, tokenizer)
    else:
        print(f"\n  Skipping Phase 2-3 (loaded from checkpoint)")
        interference_matrix = checkpoint.get("interference_matrix", checkpoint.get("partial_matrix", {}))

    # =========================================================================
    # Analysis & Output
    # =========================================================================
    print(f"\n{'='*70}")
    print("RESULTS: Cross-Task Interference Matrix")
    print(f"{'='*70}")

    conditions = [
        ("4bit_baseline", "4-bit Baseline"),
        ("gsm8k_optimal", f"GSM8K-opt (attn L{list(config.gsm8k_patch_layers)})"),
        ("arc_optimal", f"ARC-opt (MLP L{list(config.arc_patch_layers)})"),
        ("math_optimal", f"MATH-opt (MLP L{list(config.math_patch_layers)})"),
        ("all_combined", "All Combined"),
    ]

    # Print recovery rate matrix
    print(f"\n  Recovery Rate Matrix (% of flipped examples recovered):")
    print(f"  {'Condition':<35} {'GSM8K':>8} {'ARC':>8} {'MATH':>8}")
    print(f"  {'─'*63}")

    for key, label in conditions:
        if key in interference_matrix:
            m = interference_matrix[key]
            gsm = m["gsm8k"]["recovery_rate"]
            arc = m["arc"]["recovery_rate"]
            math = m["math"]["recovery_rate"]
            print(f"  {label:<35} {gsm:>7.1%} {arc:>7.1%} {math:>7.1%}")

    # Print full accuracy matrix
    print(f"\n  Full Accuracy Matrix (on all {config.num_examples_per_dataset} examples):")
    print(f"  {'Condition':<35} {'GSM8K':>8} {'ARC':>8} {'MATH':>8}")
    print(f"  {'─'*63}")

    for key, label in conditions:
        if key in interference_matrix:
            m = interference_matrix[key]
            gsm = m["gsm8k"]["full_accuracy"]
            arc = m["arc"]["full_accuracy"]
            math = m["math"]["full_accuracy"]
            print(f"  {label:<35} {gsm:>7.1%} {arc:>7.1%} {math:>7.1%}")

    # Compute deltas from 4-bit baseline
    if "4bit_baseline" in interference_matrix:
        print(f"\n  Delta from 4-bit Baseline (full accuracy):")
        print(f"  {'Condition':<35} {'GSM8K':>8} {'ARC':>8} {'MATH':>8}")
        print(f"  {'─'*63}")

        base = interference_matrix["4bit_baseline"]
        for key, label in conditions[1:]:  # Skip baseline
            if key in interference_matrix:
                m = interference_matrix[key]
                d_gsm = m["gsm8k"]["full_accuracy"] - base["gsm8k"]["full_accuracy"]
                d_arc = m["arc"]["full_accuracy"] - base["arc"]["full_accuracy"]
                d_math = m["math"]["full_accuracy"] - base["math"]["full_accuracy"]
                print(f"  {label:<35} {d_gsm:>+7.1%} {d_arc:>+7.1%} {d_math:>+7.1%}")

    # =========================================================================
    # Save Results
    # =========================================================================
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")

    output = {
        "config": {
            "model": config.model_name,
            "num_examples_per_dataset": config.num_examples_per_dataset,
            "seed": config.seed,
            "patches": {
                "gsm8k_optimal": {
                    "component": config.gsm8k_patch_component,
                    "layers": list(config.gsm8k_patch_layers),
                },
                "arc_optimal": {
                    "component": config.arc_patch_component,
                    "layers": list(config.arc_patch_layers),
                },
                "math_optimal": {
                    "component": config.math_patch_component,
                    "layers": list(config.math_patch_layers),
                },
            },
        },
        "baselines": baselines,
        "flipped_counts": {ds: len(indices) for ds, indices in flipped_indices.items()},
        "interference_matrix": interference_matrix,
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
