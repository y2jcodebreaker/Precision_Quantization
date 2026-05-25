"""
Optimal Mixed-Precision Model: The Practical Capstone
======================================================

Builds a surgically optimized model that patches only the 6 critical layers
identified through 11 phases of experimentation, and benchmarks it against
full FP16 and full 4-bit baselines.

Critical Layers (from layer sensitivity + cross-task interference):
- L1 MLP: MATH bookend (early feature extraction), shared substrate
- L6 MLP: ARC-optimal, but helps all tasks (superposition)
- L7 MLP: ARC-optimal, shared substrate
- L13 Attention: GSM8K arithmetic reasoning (monosemantic)
- L14 Attention: GSM8K arithmetic reasoning (monosemantic)
- L31 MLP: MATH bookend (output formation), shared substrate

Metrics:
- Accuracy on GSM8K, ARC-Challenge, MATH (500 examples each)
- GPU memory footprint (peak allocated)
- Memory overhead vs full 4-bit

Output: optimal_mixed_precision_report.json

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
    output_file: str = os.path.join("results", "optimal_mixed_precision_report.json")
    checkpoint_file: str = os.path.join("results", "optimal_mixed_precision_checkpoint.json")

    # Pre-computed baselines (skip FP16/4-bit if found)
    flip_report_file: str = "bidirectional_flip_report.json"

    num_examples_per_dataset: int = 500

    # The 6 critical layers identified through experiments
    attn_fp16_layers: Tuple[int, ...] = (13, 14)
    mlp_fp16_layers: Tuple[int, ...] = (1, 6, 7, 31)

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
    """Get GPU memory stats in MB."""
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
    """Run inference on examples and return accuracy stats."""
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


def cache_critical_fp16_weights(model_name: str, config: Config) -> Dict:
    """Cache FP16 weights for only the critical layers."""
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
    """Apply FP16 patches to the 6 critical layers of a 4-bit model."""
    print("\n  Applying surgical FP16 patches...")

    fp16_param_count = 0

    # Attention patches
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
            fp16_param_count += weight_info["in_features"] * weight_info["out_features"]
        print(f"    Patched attention L{layer_idx}")

    # MLP patches
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
            fp16_param_count += weight_info["in_features"] * weight_info["out_features"]
        print(f"    Patched MLP L{layer_idx}")

    # Calculate memory overhead
    fp16_bytes = fp16_param_count * 2  # 2 bytes per FP16 param
    nf4_bytes = fp16_param_count * 0.5  # 0.5 bytes per NF4 param
    overhead_mb = (fp16_bytes - nf4_bytes) / 1024 / 1024

    print(f"\n    FP16 parameters transplanted: {fp16_param_count:,}")
    print(f"    Memory overhead vs full 4-bit: {overhead_mb:.1f} MB")

    return {
        "fp16_param_count": fp16_param_count,
        "overhead_mb": overhead_mb,
    }


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
            return json.load(f)
    return None


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    config = Config()

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(config.output_dir)}")

    print("=" * 70)
    print("Optimal Mixed-Precision Model: The Practical Capstone")
    print("=" * 70)
    print(f"\nModel: {config.model_name}")
    print(f"Device: {config.device}")
    print(f"\nSurgical Patch Layers (6/32 = 19% of network):")
    print(f"  Attention FP16: L{list(config.attn_fp16_layers)}")
    print(f"  MLP FP16:       L{list(config.mlp_fp16_layers)}")

    torch.manual_seed(config.seed)

    # Check for checkpoint
    checkpoint = load_checkpoint(config.checkpoint_file)
    resumed_phase = checkpoint.get("completed_phase", 0) if checkpoint else 0
    # Sub-step tracking for granular resume within phases
    completed_substep = checkpoint.get("completed_substep", "") if checkpoint else ""

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

    print("  [checkpoint] Datasets loaded successfully")

    # =========================================================================
    # Try to load baselines from prior experiments
    # =========================================================================
    baselines_from_prior = False
    prior_baselines = {}

    if os.path.exists(config.flip_report_file):
        with open(config.flip_report_file, "r") as f:
            flip_report = json.load(f)
        for ds_name in ["gsm8k", "arc", "math"]:
            a = flip_report["analyses"][ds_name]
            prior_baselines[ds_name] = {
                "fp16": a["fp16_accuracy"],
                "int4": a["int4_accuracy"],
            }
        baselines_from_prior = True
        print(f"\n  Loaded prior baselines from {config.flip_report_file}")

    results = {}

    # =========================================================================
    # PHASE 1: FP16 Baseline
    # =========================================================================
    if resumed_phase < 1:
        print(f"\n{'='*70}")
        print("PHASE 1: FP16 Baseline")
        print(f"{'='*70}")

        if baselines_from_prior:
            print("  Using prior FP16 results (skipping inference)")
            results["fp16"] = {
                ds: {"accuracy": prior_baselines[ds]["fp16"],
                     "correct": int(prior_baselines[ds]["fp16"] * config.num_examples_per_dataset),
                     "total": config.num_examples_per_dataset}
                for ds in ["gsm8k", "arc", "math"]
            }
            results["fp16"]["memory_mb"] = None  # Not measured this run
            save_checkpoint({
                "completed_phase": 0,
                "completed_substep": "phase1_baselines_loaded",
                "results": results,
            }, config.checkpoint_file)
        else:
            model, tokenizer = load_fp16_model(config.model_name)
            reset_peak_memory()

            results["fp16"] = {}

            # Checkpoint after model load
            save_checkpoint({
                "completed_phase": 0,
                "completed_substep": "phase1_fp16_model_loaded",
                "results": results,
            }, config.checkpoint_file)

            for ds_name, examples in all_examples.items():
                res = run_inference(
                    model, tokenizer, examples,
                    config.batch_size, config.max_new_tokens,
                    f"FP16-{ds_name}",
                )
                results["fp16"][ds_name] = res

                # Checkpoint after each dataset
                save_checkpoint({
                    "completed_phase": 0,
                    "completed_substep": f"phase1_fp16_{ds_name}_done",
                    "results": results,
                }, config.checkpoint_file)

            results["fp16"]["memory_mb"] = get_memory_stats()
            unload_model(model, tokenizer)

        save_checkpoint({
            "completed_phase": 1,
            "completed_substep": "phase1_complete",
            "results": results,
        }, config.checkpoint_file)
        print("  [checkpoint] Phase 1 complete")
    else:
        results = checkpoint["results"]
        print(f"\n  Phase 1 loaded from checkpoint")

    # =========================================================================
    # PHASE 2: 4-bit Baseline
    # =========================================================================
    if resumed_phase < 2:
        print(f"\n{'='*70}")
        print("PHASE 2: 4-bit Baseline")
        print(f"{'='*70}")

        if baselines_from_prior:
            print("  Using prior 4-bit results (skipping inference)")
            results["int4"] = {
                ds: {"accuracy": prior_baselines[ds]["int4"],
                     "correct": int(prior_baselines[ds]["int4"] * config.num_examples_per_dataset),
                     "total": config.num_examples_per_dataset}
                for ds in ["gsm8k", "arc", "math"]
            }
            save_checkpoint({
                "completed_phase": 1,
                "completed_substep": "phase2_baselines_loaded",
                "results": results,
            }, config.checkpoint_file)

            # Still need to measure 4-bit memory — do a quick load
            print("  Measuring 4-bit memory footprint...")
            model, tokenizer = load_4bit_model(config.model_name)
            reset_peak_memory()

            save_checkpoint({
                "completed_phase": 1,
                "completed_substep": "phase2_4bit_model_loaded_for_memory",
                "results": results,
            }, config.checkpoint_file)

            # Run a single tiny inference to get realistic memory usage
            tiny_prompt = format_prompt(all_examples["arc"][0])
            inputs = tokenizer(tiny_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=5,
                             pad_token_id=tokenizer.pad_token_id)
            results["int4"]["memory_mb"] = get_memory_stats()
            print(f"    4-bit peak memory: {results['int4']['memory_mb']['peak_mb']:.0f} MB")
            unload_model(model, tokenizer)

            save_checkpoint({
                "completed_phase": 1,
                "completed_substep": "phase2_4bit_memory_measured",
                "results": results,
            }, config.checkpoint_file)
        else:
            model, tokenizer = load_4bit_model(config.model_name)
            reset_peak_memory()

            results["int4"] = {}

            # Checkpoint after model load
            save_checkpoint({
                "completed_phase": 1,
                "completed_substep": "phase2_4bit_model_loaded",
                "results": results,
            }, config.checkpoint_file)

            for ds_name, examples in all_examples.items():
                res = run_inference(
                    model, tokenizer, examples,
                    config.batch_size, config.max_new_tokens,
                    f"4bit-{ds_name}",
                )
                results["int4"][ds_name] = res

                # Checkpoint after each dataset
                save_checkpoint({
                    "completed_phase": 1,
                    "completed_substep": f"phase2_4bit_{ds_name}_done",
                    "results": results,
                }, config.checkpoint_file)

            results["int4"]["memory_mb"] = get_memory_stats()
            unload_model(model, tokenizer)

        save_checkpoint({
            "completed_phase": 2,
            "completed_substep": "phase2_complete",
            "results": results,
        }, config.checkpoint_file)
        print("  [checkpoint] Phase 2 complete")
    else:
        print(f"\n  Phase 2 loaded from checkpoint")

    # =========================================================================
    # PHASE 3: Surgical Mixed-Precision Model
    # =========================================================================
    if resumed_phase < 3:
        print(f"\n{'='*70}")
        print("PHASE 3: Surgical Mixed-Precision Model")
        print(f"{'='*70}")

        # Cache FP16 weights for critical layers
        fp16_cache = cache_critical_fp16_weights(config.model_name, config)

        save_checkpoint({
            "completed_phase": 2,
            "completed_substep": "phase3_fp16_weights_cached",
            "results": results,
        }, config.checkpoint_file)
        print("  [checkpoint] FP16 weights cached")

        # Load 4-bit model and apply patches
        model, tokenizer = load_4bit_model(config.model_name)
        reset_peak_memory()

        save_checkpoint({
            "completed_phase": 2,
            "completed_substep": "phase3_4bit_model_loaded",
            "results": results,
        }, config.checkpoint_file)
        print("  [checkpoint] 4-bit model loaded for patching")

        patch_info = apply_surgical_patch(model, fp16_cache, config)

        save_checkpoint({
            "completed_phase": 2,
            "completed_substep": "phase3_patches_applied",
            "results": results,
        }, config.checkpoint_file)
        print("  [checkpoint] Surgical patches applied")

        # Benchmark
        print(f"\n  Benchmarking surgical model...")
        results["surgical"] = {}
        results["surgical"]["patch_info"] = patch_info

        for ds_name, examples in all_examples.items():
            res = run_inference(
                model, tokenizer, examples,
                config.batch_size, config.max_new_tokens,
                f"Surgical-{ds_name}",
            )
            results["surgical"][ds_name] = res

            # Checkpoint after each dataset
            save_checkpoint({
                "completed_phase": 2,
                "completed_substep": f"phase3_surgical_{ds_name}_done",
                "results": results,
            }, config.checkpoint_file)
            print(f"  [checkpoint] Surgical {ds_name} evaluation saved")

        results["surgical"]["memory_mb"] = get_memory_stats()

        unload_model(model, tokenizer)

        save_checkpoint({
            "completed_phase": 3,
            "completed_substep": "phase3_complete",
            "results": results,
        }, config.checkpoint_file)
        print("  [checkpoint] Phase 3 complete")
    else:
        print(f"\n  Phase 3 loaded from checkpoint")

    # =========================================================================
    # RESULTS
    # =========================================================================
    print(f"\n{'='*70}")
    print("RESULTS: Model Comparison")
    print(f"{'='*70}")

    # Accuracy table
    print(f"\n  {'Model':<25} {'GSM8K':>8} {'ARC':>8} {'MATH':>8} {'Avg':>8}")
    print(f"  {'─'*61}")

    for model_key, model_label in [
        ("fp16", "Full FP16 (16-bit)"),
        ("int4", "Full 4-bit (NF4)"),
        ("surgical", "Surgical Mix (6 layers)"),
    ]:
        if model_key in results:
            gsm = results[model_key]["gsm8k"]["accuracy"]
            arc = results[model_key]["arc"]["accuracy"]
            math = results[model_key]["math"]["accuracy"]
            avg = (gsm + arc + math) / 3
            print(f"  {model_label:<25} {gsm:>7.1%} {arc:>7.1%} {math:>7.1%} {avg:>7.1%}")

    # Delta from 4-bit
    print(f"\n  Delta from Full 4-bit:")
    print(f"  {'Model':<25} {'GSM8K':>8} {'ARC':>8} {'MATH':>8} {'Avg':>8}")
    print(f"  {'─'*61}")

    for model_key, model_label in [
        ("fp16", "Full FP16"),
        ("surgical", "Surgical Mix"),
    ]:
        if model_key in results and "int4" in results:
            d_gsm = results[model_key]["gsm8k"]["accuracy"] - results["int4"]["gsm8k"]["accuracy"]
            d_arc = results[model_key]["arc"]["accuracy"] - results["int4"]["arc"]["accuracy"]
            d_math = results[model_key]["math"]["accuracy"] - results["int4"]["math"]["accuracy"]
            d_avg = (d_gsm + d_arc + d_math) / 3
            print(f"  {model_label:<25} {d_gsm:>+7.1%} {d_arc:>+7.1%} {d_math:>+7.1%} {d_avg:>+7.1%}")

    # Recovery rate (how much of 4-bit→FP16 gap does surgical close?)
    print(f"\n  Gap Recovery (% of FP16-4bit gap closed by surgical):")
    if "surgical" in results and "int4" in results and "fp16" in results:
        for ds_name in ["gsm8k", "arc", "math"]:
            fp16_acc = results["fp16"][ds_name]["accuracy"]
            int4_acc = results["int4"][ds_name]["accuracy"]
            surg_acc = results["surgical"][ds_name]["accuracy"]

            gap = fp16_acc - int4_acc
            if gap > 0:
                closed = (surg_acc - int4_acc) / gap * 100
                print(f"    {ds_name}: {closed:.1f}% of gap recovered "
                      f"(4bit={int4_acc:.1%} → surgical={surg_acc:.1%} → fp16={fp16_acc:.1%})")
            else:
                print(f"    {ds_name}: No gap (4-bit ≥ FP16)")

    # Memory table
    print(f"\n  Memory Footprint:")
    print(f"  {'Model':<25} {'Peak GPU':>12} {'vs 4-bit':>10}")
    print(f"  {'─'*50}")

    for model_key, model_label in [
        ("fp16", "Full FP16"),
        ("int4", "Full 4-bit"),
        ("surgical", "Surgical Mix"),
    ]:
        if model_key in results and results[model_key].get("memory_mb"):
            mem = results[model_key]["memory_mb"]
            peak = mem["peak_mb"]
            if "int4" in results and results["int4"].get("memory_mb"):
                int4_peak = results["int4"]["memory_mb"]["peak_mb"]
                overhead = peak - int4_peak
                print(f"  {model_label:<25} {peak:>9.0f} MB {overhead:>+8.0f} MB")
            else:
                print(f"  {model_label:<25} {peak:>9.0f} MB")

    if "surgical" in results and results["surgical"].get("patch_info"):
        pi = results["surgical"]["patch_info"]
        print(f"\n  Surgical Patch Details:")
        print(f"    FP16 parameters: {pi['fp16_param_count']:,}")
        print(f"    Theoretical overhead: {pi['overhead_mb']:.1f} MB")
        print(f"    Layers patched: 6/32 (18.75%)")

    # =========================================================================
    # Save Final Results
    # =========================================================================
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")

    output = {
        "config": {
            "model": config.model_name,
            "num_examples_per_dataset": config.num_examples_per_dataset,
            "seed": config.seed,
            "attn_fp16_layers": list(config.attn_fp16_layers),
            "mlp_fp16_layers": list(config.mlp_fp16_layers),
            "total_patched_layers": len(config.attn_fp16_layers) + len(config.mlp_fp16_layers),
            "total_layers": 32,
            "patch_percentage": (len(config.attn_fp16_layers) + len(config.mlp_fp16_layers)) / 32 * 100,
        },
        "results": results,
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
