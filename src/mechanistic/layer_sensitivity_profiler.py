"""
Layer Sensitivity Profiler: Finding Critical Cognitive Bottlenecks
===================================================================

This script performs a comprehensive sweep through all 32 layers of Llama-3.1-8B
to identify which individual layers have the greatest impact on recovering
quantization damage across different cognitive tasks.

Methodology:
1. Establish baselines (FP16 and 4-bit) for each dataset
2. Identify "flipped" examples (FP16 correct, 4-bit wrong)
3. For each layer L (0-31):
   - Transplant FP16 Attention into layer L only → measure recovery
   - Transplant FP16 MLP into layer L only → measure recovery
   - Restore original 4-bit weights
4. Generate sensitivity heatmap showing layer importance per dataset

Datasets:
- GSM8K: Mathematical reasoning
- ARC-Challenge: Scientific/semantic reasoning
- MATH: Competition mathematics

Memory Optimization:
- Store original 4-bit module references for restoration
- Aggressive garbage collection between each layer test
- Clean VRAM after every modification

Output: layer_sensitivity_heatmap.json

Authentication:
---------------
Set HF_TOKEN environment variable for model access.
"""

import json
import os
import re
import gc
import copy
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

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
    """Configuration for layer sensitivity profiling."""
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_file: str = "layer_sensitivity_heatmap.json"
    checkpoint_file: str = "layer_sensitivity_checkpoint.json"

    num_examples_per_dataset: int = 500
    num_layers: int = 32  # Llama-3.1-8B has 32 layers

    attn_projections: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    mlp_projections: Tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")

    batch_size: int = 4
    max_new_tokens: int = 256
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


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


# =============================================================================
# Dataset Loading
# =============================================================================

def load_gsm8k(num_examples: int) -> List[Dict]:
    """Load GSM8K examples."""
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
    """Load ARC-Challenge examples."""
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
    """Load MATH competition examples."""
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
    """Format example into prompt based on dataset type."""
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
    """Extract answer based on dataset type."""
    dataset = example["dataset"]
    text = response.replace("**", "").replace("*", "")

    if dataset == "gsm8k":
        # Try #### pattern
        match = re.search(r"####\s*\$?([+-]?\d[\d,]*\.?\d*)", text)
        if match:
            return match.group(1).replace(",", "")
        # Fallback patterns
        patterns = [
            r"(?:the\s+)?answer\s+is[:\s]*\$?([+-]?\d[\d,]*\.?\d*)",
            r"=\s*\$?([+-]?\d[\d,]*\.?\d*)\s*$",
        ]
        for p in patterns:
            match = re.search(p, text, re.IGNORECASE)
            if match:
                return match.group(1).replace(",", "")
        # Last number
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
        # Try \boxed{}
        match = re.search(r"\\boxed\{([^}]+)\}", text)
        if match:
            return match.group(1).strip()
        # Fallback
        match = re.search(r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*(.+?)(?:\.|$)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


def extract_ground_truth(example: Dict) -> str:
    """Extract ground truth answer from example."""
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
    """Check if answers match."""
    if model_answer is None or not ground_truth:
        return False

    if dataset in ["gsm8k"]:
        try:
            return abs(float(model_answer) - float(ground_truth)) < 1e-6
        except ValueError:
            return False
    elif dataset == "arc":
        return model_answer.upper() == ground_truth.upper()
    elif dataset == "math":
        # Simplified comparison
        return model_answer.strip().lower() == ground_truth.strip().lower()

    return model_answer == ground_truth


# =============================================================================
# Model Loading
# =============================================================================

def load_fp16_model(model_name: str):
    """Load model in FP16."""
    print(f"\nLoading FP16 model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def load_4bit_model(model_name: str):
    """Load model in 4-bit NF4."""
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
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def unload_model(model, tokenizer=None):
    """Unload model."""
    del model
    if tokenizer:
        del tokenizer
    aggressive_cleanup()


# =============================================================================
# FP16 Weight Cache
# =============================================================================

def cache_all_fp16_weights(model_name: str, num_layers: int) -> Dict:
    """
    Load FP16 model once and cache ALL attention and MLP weights.
    This avoids reloading FP16 model for each layer test.
    """
    print("\nCaching all FP16 weights (one-time operation)...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    cache = {"attention": {}, "mlp": {}}

    for layer_idx in tqdm(range(num_layers), desc="  Caching layers"):
        layer = model.model.layers[layer_idx]

        # Cache attention weights
        cache["attention"][layer_idx] = {}
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            proj = getattr(layer.self_attn, proj_name)
            cache["attention"][layer_idx][proj_name] = {
                "weight": proj.weight.data.clone().cpu(),
                "bias": proj.bias.data.clone().cpu() if proj.bias is not None else None,
                "in_features": proj.in_features,
                "out_features": proj.out_features,
            }

        # Cache MLP weights
        cache["mlp"][layer_idx] = {}
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(layer.mlp, proj_name)
            cache["mlp"][layer_idx][proj_name] = {
                "weight": proj.weight.data.clone().cpu(),
                "bias": proj.bias.data.clone().cpu() if proj.bias is not None else None,
                "in_features": proj.in_features,
                "out_features": proj.out_features,
            }

    del model
    aggressive_cleanup()

    print(f"  Cached weights for {num_layers} layers")
    return cache


# =============================================================================
# Layer Transplant Operations
# =============================================================================

def transplant_attention_layer(
    model,
    layer_idx: int,
    fp16_cache: Dict,
    device: str,
) -> Dict[str, Any]:
    """
    Transplant FP16 attention weights into a single layer.
    Returns the original 4-bit modules for later restoration.
    """
    layer = model.model.layers[layer_idx]
    original_modules = {}

    for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        # Store original module
        original_modules[proj_name] = getattr(layer.self_attn, proj_name)

        # Create FP16 replacement
        weight_info = fp16_cache["attention"][layer_idx][proj_name]
        new_linear = nn.Linear(
            in_features=weight_info["in_features"],
            out_features=weight_info["out_features"],
            bias=weight_info["bias"] is not None,
            dtype=torch.float16,
            device=device,
        )
        new_linear.weight.data = weight_info["weight"].to(device)
        if weight_info["bias"] is not None:
            new_linear.bias.data = weight_info["bias"].to(device)

        setattr(layer.self_attn, proj_name, new_linear)

    return original_modules


def transplant_mlp_layer(
    model,
    layer_idx: int,
    fp16_cache: Dict,
    device: str,
) -> Dict[str, Any]:
    """
    Transplant FP16 MLP weights into a single layer.
    Returns the original 4-bit modules for later restoration.
    """
    layer = model.model.layers[layer_idx]
    original_modules = {}

    for proj_name in ("gate_proj", "up_proj", "down_proj"):
        # Store original module
        original_modules[proj_name] = getattr(layer.mlp, proj_name)

        # Create FP16 replacement
        weight_info = fp16_cache["mlp"][layer_idx][proj_name]
        new_linear = nn.Linear(
            in_features=weight_info["in_features"],
            out_features=weight_info["out_features"],
            bias=weight_info["bias"] is not None,
            dtype=torch.float16,
            device=device,
        )
        new_linear.weight.data = weight_info["weight"].to(device)
        if weight_info["bias"] is not None:
            new_linear.bias.data = weight_info["bias"].to(device)

        setattr(layer.mlp, proj_name, new_linear)

    return original_modules


def restore_attention_layer(model, layer_idx: int, original_modules: Dict[str, Any]):
    """Restore original 4-bit attention modules."""
    layer = model.model.layers[layer_idx]

    for proj_name, original_module in original_modules.items():
        # Get current FP16 module and delete it
        current = getattr(layer.self_attn, proj_name)
        del current

        # Restore original
        setattr(layer.self_attn, proj_name, original_module)

    aggressive_cleanup()


def restore_mlp_layer(model, layer_idx: int, original_modules: Dict[str, Any]):
    """Restore original 4-bit MLP modules."""
    layer = model.model.layers[layer_idx]

    for proj_name, original_module in original_modules.items():
        current = getattr(layer.mlp, proj_name)
        del current
        setattr(layer.mlp, proj_name, original_module)

    aggressive_cleanup()


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

        prompts = [format_prompt(ex) for ex in batch_examples]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
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
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            ground_truth = extract_ground_truth(example)
            model_answer = extract_answer(response, example)
            is_correct = check_match(model_answer, ground_truth, example["dataset"])

            results.append({
                "index": example["index"],
                "dataset": example["dataset"],
                "correct": is_correct,
            })

    return results


def calculate_recovery_rate(results: List[Dict]) -> float:
    """Calculate recovery rate from results."""
    if not results:
        return 0.0
    correct = sum(1 for r in results if r["correct"])
    return correct / len(results)


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(data: Dict, filepath: str):
    """Save checkpoint."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_checkpoint(filepath: str) -> Optional[Dict]:
    """Load checkpoint if exists."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    config = Config()

    print("=" * 70)
    print("Layer Sensitivity Profiler: Finding Critical Cognitive Bottlenecks")
    print("=" * 70)
    print(f"\nModel: {config.model_name}")
    print(f"Layers to test: {config.num_layers}")
    print(f"Tests per layer: 2 (Attention + MLP)")
    print(f"Total tests: {config.num_layers * 2}")
    print(f"Device: {config.device}")

    torch.manual_seed(config.seed)

    # Results structure
    results = {
        "config": {
            "model": config.model_name,
            "num_layers": config.num_layers,
            "num_examples_per_dataset": config.num_examples_per_dataset,
        },
        "baselines": {},
        "flipped_indices": {},
        "layer_sensitivity": {
            "attention": {ds: {} for ds in ["gsm8k", "arc", "math"]},
            "mlp": {ds: {} for ds in ["gsm8k", "arc", "math"]},
        },
    }

    # =========================================================================
    # PHASE 0: Load Datasets
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 0: Loading Datasets")
    print(f"{'='*70}")

    all_examples = {
        "gsm8k": load_gsm8k(config.num_examples_per_dataset),
        "arc": load_arc_challenge(config.num_examples_per_dataset),
        "math": load_math(config.num_examples_per_dataset),
    }

    for ds, examples in all_examples.items():
        print(f"  {ds}: {len(examples)} examples")

    # =========================================================================
    # PHASE 1: FP16 Baseline
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 1: FP16 Baseline Evaluation")
    print(f"{'='*70}")

    model, tokenizer = load_fp16_model(config.model_name)

    fp16_results = {}
    for ds_name, examples in all_examples.items():
        print(f"\n  Testing {ds_name}...")
        res = run_inference_on_subset(
            model, tokenizer, examples,
            config.batch_size, config.max_new_tokens
        )
        fp16_results[ds_name] = res
        acc = calculate_recovery_rate(res)
        results["baselines"][f"{ds_name}_fp16"] = acc
        print(f"    FP16 Accuracy: {acc:.2%}")

    unload_model(model, tokenizer)

    # =========================================================================
    # PHASE 2: 4-bit Baseline & Damage Identification
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 2: 4-bit Baseline & Damage Identification")
    print(f"{'='*70}")

    model, tokenizer = load_4bit_model(config.model_name)

    int4_results = {}
    for ds_name, examples in all_examples.items():
        print(f"\n  Testing {ds_name}...")
        res = run_inference_on_subset(
            model, tokenizer, examples,
            config.batch_size, config.max_new_tokens
        )
        int4_results[ds_name] = res
        acc = calculate_recovery_rate(res)
        results["baselines"][f"{ds_name}_4bit"] = acc
        print(f"    4-bit Accuracy: {acc:.2%}")

        # Identify flipped indices
        fp16_by_idx = {r["index"]: r for r in fp16_results[ds_name]}
        int4_by_idx = {r["index"]: r for r in res}

        flipped = [
            idx for idx in fp16_by_idx
            if fp16_by_idx[idx]["correct"] and not int4_by_idx[idx]["correct"]
        ]
        results["flipped_indices"][ds_name] = flipped

        drop = results["baselines"][f"{ds_name}_fp16"] - acc
        print(f"    Accuracy Drop: {drop:.2%}")
        print(f"    Flipped Examples: {len(flipped)}")

    # Keep 4-bit model loaded for the sweep
    print(f"\n  GPU Memory: {get_gpu_memory_mb():.0f} MB")

    # =========================================================================
    # PHASE 3: Cache FP16 Weights
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 3: Caching FP16 Weights")
    print(f"{'='*70}")

    # Temporarily unload 4-bit model to cache FP16 weights
    unload_model(model, tokenizer)

    fp16_cache = cache_all_fp16_weights(config.model_name, config.num_layers)

    # Reload 4-bit model for sweep
    model, tokenizer = load_4bit_model(config.model_name)

    # Prepare flipped examples for testing
    flipped_examples = {}
    for ds_name in all_examples:
        flipped_idx_set = set(results["flipped_indices"][ds_name])
        flipped_examples[ds_name] = [
            ex for ex in all_examples[ds_name]
            if ex["index"] in flipped_idx_set
        ]
        print(f"  {ds_name} flipped examples to test: {len(flipped_examples[ds_name])}")

    # Save Phase 2 checkpoint
    save_checkpoint(results, config.checkpoint_file)

    # =========================================================================
    # PHASE 4: Layer-by-Layer Sensitivity Sweep
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 4: Layer-by-Layer Sensitivity Sweep")
    print(f"{'='*70}")

    total_tests = config.num_layers * 2
    test_count = 0

    for layer_idx in range(config.num_layers):
        print(f"\n  Layer {layer_idx}/{config.num_layers - 1}")
        print(f"  {'-'*40}")

        # ---------------------------------------------------------------------
        # Test Attention
        # ---------------------------------------------------------------------
        test_count += 1
        print(f"    [{test_count}/{total_tests}] Testing Attention...")

        # Transplant
        original_attn = transplant_attention_layer(
            model, layer_idx, fp16_cache, config.device
        )

        # Test on flipped examples for each dataset
        for ds_name in ["gsm8k", "arc", "math"]:
            if not flipped_examples[ds_name]:
                results["layer_sensitivity"]["attention"][ds_name][str(layer_idx)] = 0.0
                continue

            res = run_inference_on_subset(
                model, tokenizer, flipped_examples[ds_name],
                config.batch_size, config.max_new_tokens
            )
            recovery_rate = calculate_recovery_rate(res)
            results["layer_sensitivity"]["attention"][ds_name][str(layer_idx)] = recovery_rate

        # Restore
        restore_attention_layer(model, layer_idx, original_attn)

        attn_rates = [
            results["layer_sensitivity"]["attention"][ds][str(layer_idx)]
            for ds in ["gsm8k", "arc", "math"]
        ]
        print(f"      Recovery: GSM8K={attn_rates[0]:.1%}, ARC={attn_rates[1]:.1%}, MATH={attn_rates[2]:.1%}")

        # ---------------------------------------------------------------------
        # Test MLP
        # ---------------------------------------------------------------------
        test_count += 1
        print(f"    [{test_count}/{total_tests}] Testing MLP...")

        # Transplant
        original_mlp = transplant_mlp_layer(
            model, layer_idx, fp16_cache, config.device
        )

        # Test on flipped examples for each dataset
        for ds_name in ["gsm8k", "arc", "math"]:
            if not flipped_examples[ds_name]:
                results["layer_sensitivity"]["mlp"][ds_name][str(layer_idx)] = 0.0
                continue

            res = run_inference_on_subset(
                model, tokenizer, flipped_examples[ds_name],
                config.batch_size, config.max_new_tokens
            )
            recovery_rate = calculate_recovery_rate(res)
            results["layer_sensitivity"]["mlp"][ds_name][str(layer_idx)] = recovery_rate

        # Restore
        restore_mlp_layer(model, layer_idx, original_mlp)

        mlp_rates = [
            results["layer_sensitivity"]["mlp"][ds][str(layer_idx)]
            for ds in ["gsm8k", "arc", "math"]
        ]
        print(f"      Recovery: GSM8K={mlp_rates[0]:.1%}, ARC={mlp_rates[1]:.1%}, MATH={mlp_rates[2]:.1%}")

        # Memory status
        print(f"      GPU Memory: {get_gpu_memory_mb():.0f} MB")

        # Periodic checkpoint
        if (layer_idx + 1) % 8 == 0:
            save_checkpoint(results, config.checkpoint_file)
            print(f"      Checkpoint saved")

    unload_model(model, tokenizer)

    # =========================================================================
    # PHASE 5: Generate Rankings
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 5: Generating Layer Rankings")
    print(f"{'='*70}")

    rankings = {"attention": {}, "mlp": {}}

    for component in ["attention", "mlp"]:
        for ds_name in ["gsm8k", "arc", "math"]:
            layer_scores = results["layer_sensitivity"][component][ds_name]

            # Sort by recovery rate (descending)
            sorted_layers = sorted(
                layer_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            rankings[component][ds_name] = [
                {"layer": int(layer), "recovery_rate": rate}
                for layer, rate in sorted_layers[:10]  # Top 10
            ]

            print(f"\n  Top 5 {component.upper()} layers for {ds_name}:")
            for i, (layer, rate) in enumerate(sorted_layers[:5]):
                print(f"    {i+1}. Layer {layer}: {rate:.1%}")

    results["rankings"] = rankings

    # =========================================================================
    # Save Final Results
    # =========================================================================
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")

    with open(config.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {config.output_file}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Critical Layers by Dataset")
    print(f"{'='*70}")

    for ds_name in ["gsm8k", "arc", "math"]:
        print(f"\n  {ds_name.upper()}:")
        print(f"    Best Attention Layer: {rankings['attention'][ds_name][0]['layer']} "
              f"({rankings['attention'][ds_name][0]['recovery_rate']:.1%})")
        print(f"    Best MLP Layer: {rankings['mlp'][ds_name][0]['layer']} "
              f"({rankings['mlp'][ds_name][0]['recovery_rate']:.1%})")

    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}")

    return results


if __name__ == "__main__":
    results = main()
