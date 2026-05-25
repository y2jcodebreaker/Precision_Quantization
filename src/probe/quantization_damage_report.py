"""
Quantization Damage Report: FP16 vs 4-bit Comparison
=====================================================

This script identifies GSM8K questions that are correctly answered by the
FP16 model but incorrectly answered after 4-bit quantization ("flipped" cases).

These flipped cases represent the "quantization damage" - examples where
quantization specifically hurts model performance.

Authentication:
---------------
Set HF_TOKEN environment variable for model access.
"""

import json
import os
import re
import gc
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import torch
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
    """Configuration for quantization damage analysis."""
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_file: str = "quantization_damage.json"
    num_examples: int = 1000
    batch_size: int = 4  # Smaller batch for 4-bit to avoid OOM
    max_new_tokens: int = 512
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_flipped_examples_to_save: int = 5


# =============================================================================
# Data Loading
# =============================================================================

def load_gsm8k_examples(num_examples: int) -> List[Dict]:
    """Load examples from GSM8K test split."""
    print(f"Loading GSM8K dataset (first {num_examples} test examples)...")
    dataset = load_dataset("gsm8k", "main", split="test")

    examples = []
    for i, item in enumerate(dataset):
        if i >= num_examples:
            break
        examples.append({
            "index": i,
            "question": item["question"],
            "answer": item["answer"],
        })

    print(f"  Loaded {len(examples)} examples")
    return examples


# =============================================================================
# Answer Extraction
# =============================================================================

def extract_gsm8k_answer(answer_text: str) -> Optional[str]:
    """Extract the numerical answer from GSM8K ground truth format."""
    match = re.search(r"####\s*([+-]?\d[\d,]*\.?\d*)", answer_text)
    if match:
        return match.group(1).replace(",", "")
    return None


def extract_model_answer(generated_text: str) -> Optional[str]:
    """Extract the numerical answer from model's generated response."""
    # Normalize: remove markdown formatting
    text = generated_text.replace("**", "").replace("*", "")

    # Try #### pattern first
    match = re.search(r"####\s*\$?([+-]?\d[\d,]*\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")

    # LaTeX boxed
    match = re.search(r"\\boxed\{([+-]?\d[\d,]*\.?\d*)\}", text)
    if match:
        return match.group(1).replace(",", "")

    # Common answer patterns
    patterns = [
        r"(?:the\s+)?answer\s+is[:\s]*\$?([+-]?\d[\d,]*\.?\d*)",
        r"final\s+answer[:\s]*\$?([+-]?\d[\d,]*\.?\d*)",
        r"=\s*\$?([+-]?\d[\d,]*\.?\d*)\s*(?:dollars?|\.|\s|$)",
        r"(?:therefore|thus|so)[,\s]+(?:the\s+)?(?:answer\s+is\s+)?\$?([+-]?\d[\d,]*\.?\d*)",
        r"is\s+\$?([+-]?\d[\d,]*\.?\d*)\s*(?:dollars?)?\.?\s*$",
        r"total(?:\s+is|\s+of|:)\s*\$?([+-]?\d[\d,]*\.?\d*)",
        r"\$([+-]?\d[\d,]*\.?\d*)\s*(?:dollars?)?\.?\s*$",
        r"equals?\s+\$?([+-]?\d[\d,]*\.?\d*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).replace(",", "")

    # Fallback: last number
    numbers = re.findall(r"(?<!\d)([+-]?\d[\d,]*\.?\d*)(?!\d)", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def check_answer_match(model_answer: Optional[str], ground_truth: Optional[str]) -> bool:
    """Check if model answer matches ground truth."""
    if model_answer is None or ground_truth is None:
        return False
    try:
        model_num = float(model_answer)
        truth_num = float(ground_truth)
        return abs(model_num - truth_num) < 1e-6
    except ValueError:
        return model_answer.strip() == ground_truth.strip()


# =============================================================================
# Prompt Formatting
# =============================================================================

def format_prompt(question: str) -> str:
    """Format question using Llama 3.1 Instruct template."""
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


# =============================================================================
# Model Loading
# =============================================================================

def load_fp16_model(model_name: str, device: str):
    """Load model in FP16 precision."""
    print(f"\nLoading FP16 model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Left padding for generation

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    print(f"  Model loaded in FP16")
    return model, tokenizer


def load_4bit_model(model_name: str, device: str):
    """Load model in 4-bit precision using BitsAndBytes."""
    print(f"\nLoading 4-bit model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",  # NormalFloat4 is typically better
        bnb_4bit_use_double_quant=True,  # Nested quantization for memory efficiency
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()

    print(f"  Model loaded in 4-bit (NF4)")
    return model, tokenizer


def unload_model(model, tokenizer):
    """Unload model and free GPU memory."""
    print("\nUnloading model and clearing GPU memory...")
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("  Memory cleared")


# =============================================================================
# Inference
# =============================================================================

def run_inference(
    model,
    tokenizer,
    examples: List[Dict],
    batch_size: int,
    max_new_tokens: int,
    pass_name: str,
) -> List[Dict]:
    """
    Run batched inference on examples.

    Returns list of results with model answers and correctness.
    """
    print(f"\n  Running {pass_name} inference...")

    results = []
    num_batches = (len(examples) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"    {pass_name}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(examples))
        batch_examples = examples[start_idx:end_idx]

        # Format prompts
        prompts = [format_prompt(ex["question"]) for ex in batch_examples]

        # Tokenize with padding
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode and extract answers
        for i, (example, output) in enumerate(zip(batch_examples, outputs)):
            # Get only the generated part (after input)
            input_len = inputs["input_ids"][i].shape[0]
            generated_tokens = output[input_len:]
            response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Extract answers
            ground_truth = extract_gsm8k_answer(example["answer"])
            model_answer = extract_model_answer(response_text)
            is_correct = check_answer_match(model_answer, ground_truth)

            results.append({
                "index": example["index"],
                "question": example["question"],
                "ground_truth": ground_truth,
                "model_answer": model_answer,
                "correct": is_correct,
                "response": response_text,
            })

        # Clear cache periodically
        if (batch_idx + 1) % 10 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Calculate accuracy
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / len(results)
    print(f"    {pass_name} Accuracy: {accuracy:.2%} ({correct_count}/{len(results)})")

    return results


# =============================================================================
# Analysis
# =============================================================================

def analyze_results(
    fp16_results: List[Dict],
    int4_results: List[Dict],
    num_flipped_to_save: int,
) -> Dict:
    """
    Compare FP16 and 4-bit results to find flipped cases.

    Flipped = FP16 correct AND 4-bit incorrect
    """
    print("\nAnalyzing results...")

    # Build lookup by index
    fp16_by_idx = {r["index"]: r for r in fp16_results}
    int4_by_idx = {r["index"]: r for r in int4_results}

    # Calculate accuracies
    fp16_correct = sum(1 for r in fp16_results if r["correct"])
    int4_correct = sum(1 for r in int4_results if r["correct"])
    fp16_accuracy = fp16_correct / len(fp16_results)
    int4_accuracy = int4_correct / len(int4_results)

    # Find flipped cases
    flipped_indices = []
    flipped_examples = []

    for idx in sorted(fp16_by_idx.keys()):
        fp16_result = fp16_by_idx[idx]
        int4_result = int4_by_idx[idx]

        if fp16_result["correct"] and not int4_result["correct"]:
            flipped_indices.append(idx)

            # Save detailed info for first N flipped cases
            if len(flipped_examples) < num_flipped_to_save:
                flipped_examples.append({
                    "index": idx,
                    "question": fp16_result["question"],
                    "ground_truth": fp16_result["ground_truth"],
                    "fp16_answer": fp16_result["model_answer"],
                    "int4_answer": int4_result["model_answer"],
                    "fp16_response": fp16_result["response"][:500],
                    "int4_response": int4_result["response"][:500],
                })

    # Also count reverse flips (4-bit correct, FP16 incorrect) for reference
    reverse_flipped = sum(
        1 for idx in fp16_by_idx
        if not fp16_by_idx[idx]["correct"] and int4_by_idx[idx]["correct"]
    )

    # Both correct
    both_correct = sum(
        1 for idx in fp16_by_idx
        if fp16_by_idx[idx]["correct"] and int4_by_idx[idx]["correct"]
    )

    # Both incorrect
    both_incorrect = sum(
        1 for idx in fp16_by_idx
        if not fp16_by_idx[idx]["correct"] and not int4_by_idx[idx]["correct"]
    )

    print(f"\n  FP16 Accuracy:  {fp16_accuracy:.2%} ({fp16_correct}/{len(fp16_results)})")
    print(f"  4-bit Accuracy: {int4_accuracy:.2%} ({int4_correct}/{len(int4_results)})")
    print(f"  Accuracy Drop:  {fp16_accuracy - int4_accuracy:.2%}")
    print(f"\n  Breakdown:")
    print(f"    Both correct:      {both_correct}")
    print(f"    Both incorrect:    {both_incorrect}")
    print(f"    Flipped (FP16→4b): {len(flipped_indices)} (quantization damage)")
    print(f"    Reverse flip:      {reverse_flipped} (4-bit got lucky)")

    return {
        "fp16_accuracy": fp16_accuracy,
        "int4_accuracy": int4_accuracy,
        "accuracy_drop": fp16_accuracy - int4_accuracy,
        "total_examples": len(fp16_results),
        "flipped_count": len(flipped_indices),
        "flipped_indices": flipped_indices,
        "reverse_flipped_count": reverse_flipped,
        "both_correct": both_correct,
        "both_incorrect": both_incorrect,
        "flipped_examples": flipped_examples,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    config = Config()

    print("=" * 70)
    print("Quantization Damage Report: FP16 vs 4-bit")
    print("=" * 70)
    print(f"\nModel: {config.model_name}")
    print(f"Examples: {config.num_examples}")
    print(f"Device: {config.device}")

    # Set seed
    torch.manual_seed(config.seed)

    # Load data
    examples = load_gsm8k_examples(config.num_examples)

    # =========================================================================
    # Pass 1: FP16
    # =========================================================================
    print("\n" + "=" * 70)
    print("PASS 1: FP16 Inference")
    print("=" * 70)

    fp16_model, fp16_tokenizer = load_fp16_model(config.model_name, config.device)

    fp16_results = run_inference(
        model=fp16_model,
        tokenizer=fp16_tokenizer,
        examples=examples,
        batch_size=config.batch_size,
        max_new_tokens=config.max_new_tokens,
        pass_name="FP16",
    )

    # Unload FP16 model to free memory for 4-bit
    unload_model(fp16_model, fp16_tokenizer)

    # =========================================================================
    # Pass 2: 4-bit
    # =========================================================================
    print("\n" + "=" * 70)
    print("PASS 2: 4-bit Inference")
    print("=" * 70)

    int4_model, int4_tokenizer = load_4bit_model(config.model_name, config.device)

    int4_results = run_inference(
        model=int4_model,
        tokenizer=int4_tokenizer,
        examples=examples,
        batch_size=config.batch_size,
        max_new_tokens=config.max_new_tokens,
        pass_name="4-bit",
    )

    # Unload 4-bit model
    unload_model(int4_model, int4_tokenizer)

    # =========================================================================
    # Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    analysis = analyze_results(
        fp16_results=fp16_results,
        int4_results=int4_results,
        num_flipped_to_save=config.num_flipped_examples_to_save,
    )

    # =========================================================================
    # Save Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    output_data = {
        "config": {
            "model": config.model_name,
            "num_examples": config.num_examples,
            "seed": config.seed,
        },
        "fp16_accuracy": analysis["fp16_accuracy"],
        "int4_accuracy": analysis["int4_accuracy"],
        "accuracy_drop": analysis["accuracy_drop"],
        "flipped_count": analysis["flipped_count"],
        "flipped_indices": analysis["flipped_indices"],
        "reverse_flipped_count": analysis["reverse_flipped_count"],
        "both_correct": analysis["both_correct"],
        "both_incorrect": analysis["both_incorrect"],
        "flipped_examples": analysis["flipped_examples"],
    }

    with open(config.output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {config.output_file}")

    # Print flipped examples summary
    if analysis["flipped_examples"]:
        print("\n" + "-" * 70)
        print(f"SAMPLE FLIPPED CASES (first {len(analysis['flipped_examples'])})")
        print("-" * 70)

        for i, ex in enumerate(analysis["flipped_examples"]):
            print(f"\n[{i+1}] Index {ex['index']}:")
            print(f"    Question: {ex['question'][:100]}...")
            print(f"    Ground Truth: {ex['ground_truth']}")
            print(f"    FP16 Answer:  {ex['fp16_answer']} ✓")
            print(f"    4-bit Answer: {ex['int4_answer']} ✗")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

    return analysis


if __name__ == "__main__":
    analysis = main()
