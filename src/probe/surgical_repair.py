"""
Surgical Repair: Mixed-Precision Induction Patch
=================================================

This script tests the hypothesis that preserving FP16 precision in
induction head layers can recover performance lost to 4-bit quantization.

The Strategy ("Organ Transplant"):
1. Extract attention projection weights from FP16 model for target layers
2. Load 4-bit quantized model
3. Replace quantized attention projections with FP16 weights in target layers
4. Test on "flipped" examples (FP16 correct, 4-bit incorrect)
5. Measure recovery rate

Authentication:
---------------
Set HF_TOKEN environment variable for model access.
"""

import json
import os
import re
import gc
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from copy import deepcopy

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


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration for surgical repair."""
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    damage_report_file: str = "quantization_damage.json"
    induction_heads_file: str = "induction_heads_candidate.json"
    output_file: str = "surgical_report.json"

    # Target layers containing induction heads (from analysis)
    target_layers: Tuple[int, ...] = (2, 5, 8, 10, 15, 16, 20, 27)

    # Attention projections to transplant
    attn_projections: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    batch_size: int = 4
    max_new_tokens: int = 512
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_qualitative_samples: int = 5


# =============================================================================
# Answer Extraction (same as quantization_damage_report.py)
# =============================================================================

def extract_gsm8k_answer(answer_text: str) -> Optional[str]:
    """Extract the numerical answer from GSM8K ground truth format."""
    match = re.search(r"####\s*([+-]?\d[\d,]*\.?\d*)", answer_text)
    if match:
        return match.group(1).replace(",", "")
    return None


def extract_model_answer(generated_text: str) -> Optional[str]:
    """Extract the numerical answer from model's generated response."""
    text = generated_text.replace("**", "").replace("*", "")

    match = re.search(r"####\s*\$?([+-]?\d[\d,]*\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")

    match = re.search(r"\\boxed\{([+-]?\d[\d,]*\.?\d*)\}", text)
    if match:
        return match.group(1).replace(",", "")

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


def format_prompt(question: str) -> str:
    """Format question using Llama 3.1 Instruct template."""
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


# =============================================================================
# Data Loading
# =============================================================================

def load_damage_report(filepath: str) -> Dict:
    """Load the quantization damage report."""
    print(f"Loading damage report from {filepath}...")
    with open(filepath, "r") as f:
        data = json.load(f)
    print(f"  Found {data['flipped_count']} flipped examples")
    return data


def load_flipped_examples(damage_report: Dict) -> List[Dict]:
    """Extract flipped examples from damage report."""
    flipped_indices = set(damage_report["flipped_indices"])

    # We need to reload GSM8K to get the full questions
    from datasets import load_dataset
    dataset = load_dataset("gsm8k", "main", split="test")

    examples = []
    for idx in sorted(flipped_indices):
        if idx < len(dataset):
            item = dataset[int(idx)]
            examples.append({
                "index": idx,
                "question": item["question"],
                "answer": item["answer"],
            })

    print(f"  Loaded {len(examples)} flipped examples for testing")
    return examples


# =============================================================================
# Weight Extraction (FP16)
# =============================================================================

def extract_attention_weights(
    model_name: str,
    target_layers: Tuple[int, ...],
    attn_projections: Tuple[str, ...],
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Load FP16 model and extract attention projection weights for target layers.

    Returns:
        Dict mapping layer_idx -> {proj_name -> weight_tensor}
    """
    print(f"\n{'='*70}")
    print("STEP 1: Extracting FP16 Attention Weights")
    print(f"{'='*70}")
    print(f"Target layers: {target_layers}")
    print(f"Projections: {attn_projections}")

    print(f"\nLoading FP16 model (CPU offload)...")

    # Load to CPU to save GPU memory
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",  # Keep on CPU during extraction
        low_cpu_mem_usage=True,
    )

    saved_weights = {}

    for layer_idx in target_layers:
        print(f"  Extracting layer {layer_idx}...")
        layer = model.model.layers[layer_idx]
        self_attn = layer.self_attn

        saved_weights[layer_idx] = {}

        for proj_name in attn_projections:
            proj_module = getattr(self_attn, proj_name)
            # Clone and detach the weights
            saved_weights[layer_idx][proj_name] = {
                "weight": proj_module.weight.data.clone().detach(),
                "bias": proj_module.bias.data.clone().detach() if proj_module.bias is not None else None,
                "in_features": proj_module.in_features,
                "out_features": proj_module.out_features,
            }

    # Aggressive cleanup
    print("\nUnloading FP16 model...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()

    print(f"  Extracted weights for {len(saved_weights)} layers")
    return saved_weights


# =============================================================================
# Surgical Transplant
# =============================================================================

def transplant_attention_weights(
    model,
    saved_weights: Dict[int, Dict[str, torch.Tensor]],
    target_layers: Tuple[int, ...],
    attn_projections: Tuple[str, ...],
    device: str,
) -> None:
    """
    Replace quantized attention projections with FP16 weights.

    This performs in-place modification of the model.
    """
    print(f"\n{'='*70}")
    print("STEP 3: Transplanting FP16 Weights into 4-bit Model")
    print(f"{'='*70}")

    for layer_idx in target_layers:
        print(f"  Transplanting layer {layer_idx}...")
        layer = model.model.layers[layer_idx]
        self_attn = layer.self_attn

        for proj_name in attn_projections:
            weight_info = saved_weights[layer_idx][proj_name]

            # Create new FP16 Linear layer
            new_linear = nn.Linear(
                in_features=weight_info["in_features"],
                out_features=weight_info["out_features"],
                bias=weight_info["bias"] is not None,
                dtype=torch.float16,
                device=device,
            )

            # Load saved weights
            new_linear.weight.data = weight_info["weight"].to(device)
            if weight_info["bias"] is not None:
                new_linear.bias.data = weight_info["bias"].to(device)

            # Replace the quantized module with FP16 module
            setattr(self_attn, proj_name, new_linear)

        # Clear cache after each layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n  Transplant complete for {len(target_layers)} layers")

    # Verify transplant
    print("\n  Verifying transplant...")
    for layer_idx in target_layers[:2]:  # Check first 2 layers
        layer = model.model.layers[layer_idx]
        q_proj = layer.self_attn.q_proj
        print(f"    Layer {layer_idx} q_proj: {type(q_proj).__name__}, dtype={q_proj.weight.dtype}")


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
    """Run batched inference on examples."""
    print(f"\n  Running {pass_name} inference on {len(examples)} examples...")

    results = []
    num_batches = (len(examples) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"    {pass_name}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(examples))
        batch_examples = examples[start_idx:end_idx]

        prompts = [format_prompt(ex["question"]) for ex in batch_examples]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
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
            response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

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

        if (batch_idx + 1) % 5 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / len(results) if results else 0
    print(f"    {pass_name} Accuracy: {accuracy:.2%} ({correct_count}/{len(results)})")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    config = Config()

    print("=" * 70)
    print("Surgical Repair: Mixed-Precision Induction Patch")
    print("=" * 70)
    print(f"\nModel: {config.model_name}")
    print(f"Target Layers: {config.target_layers}")
    print(f"Device: {config.device}")

    torch.manual_seed(config.seed)

    # =========================================================================
    # Load Damage Report
    # =========================================================================
    damage_report = load_damage_report(config.damage_report_file)
    flipped_examples = load_flipped_examples(damage_report)

    if not flipped_examples:
        print("ERROR: No flipped examples found!")
        return

    # Store original 4-bit answers for comparison
    int4_answers = {}
    for ex in damage_report.get("flipped_examples", []):
        int4_answers[ex["index"]] = ex.get("int4_answer")

    # =========================================================================
    # Step 1: Extract FP16 Weights
    # =========================================================================
    saved_weights = extract_attention_weights(
        model_name=config.model_name,
        target_layers=config.target_layers,
        attn_projections=config.attn_projections,
    )

    # =========================================================================
    # Step 2: Load 4-bit Model
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 2: Loading 4-bit Model")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
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
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    print("  4-bit model loaded")

    # =========================================================================
    # Step 3: Transplant Weights
    # =========================================================================
    transplant_attention_weights(
        model=model,
        saved_weights=saved_weights,
        target_layers=config.target_layers,
        attn_projections=config.attn_projections,
        device=config.device,
    )

    # Free saved weights after transplant
    del saved_weights
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================================
    # Step 4: Run Inference on Flipped Examples
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 4: Testing Mixed-Precision Model (Recovery Ward)")
    print(f"{'='*70}")

    hybrid_results = run_inference(
        model=model,
        tokenizer=tokenizer,
        examples=flipped_examples,
        batch_size=config.batch_size,
        max_new_tokens=config.max_new_tokens,
        pass_name="Hybrid",
    )

    # =========================================================================
    # Step 5: Analysis
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 5: Recovery Analysis")
    print(f"{'='*70}")

    total_flipped = len(flipped_examples)
    recovered_count = sum(1 for r in hybrid_results if r["correct"])
    recovery_rate = recovered_count / total_flipped if total_flipped > 0 else 0

    recovered_indices = [r["index"] for r in hybrid_results if r["correct"]]
    still_failed_indices = [r["index"] for r in hybrid_results if not r["correct"]]

    print(f"\n  Total Flipped Examples: {total_flipped}")
    print(f"  Recovered: {recovered_count}")
    print(f"  Still Failed: {total_flipped - recovered_count}")
    print(f"  Recovery Rate: {recovery_rate:.2%}")

    # Build qualitative samples
    qualitative_samples = []
    sample_count = 0

    for r in hybrid_results:
        if sample_count >= config.num_qualitative_samples:
            break

        # Get original 4-bit answer if available
        orig_int4_answer = int4_answers.get(r["index"], "N/A")

        qualitative_samples.append({
            "index": r["index"],
            "question": r["question"][:200] + "..." if len(r["question"]) > 200 else r["question"],
            "ground_truth": r["ground_truth"],
            "failed_4bit_answer": orig_int4_answer,
            "hybrid_answer": r["model_answer"],
            "recovered": r["correct"],
            "hybrid_response": r["response"][:300] + "..." if len(r["response"]) > 300 else r["response"],
        })
        sample_count += 1

    # =========================================================================
    # Step 6: Save Report
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 6: Saving Report")
    print(f"{'='*70}")

    report = {
        "config": {
            "model": config.model_name,
            "target_layers": list(config.target_layers),
            "attn_projections": list(config.attn_projections),
        },
        "summary": {
            "total_flipped": total_flipped,
            "recovery_count": recovered_count,
            "recovery_rate": recovery_rate,
            "still_failed": total_flipped - recovered_count,
        },
        "recovered_indices": recovered_indices,
        "still_failed_indices": still_failed_indices,
        "qualitative_samples": qualitative_samples,
    }

    with open(config.output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {config.output_file}")

    # Print qualitative samples
    print(f"\n{'-'*70}")
    print(f"QUALITATIVE SAMPLES (first {len(qualitative_samples)})")
    print(f"{'-'*70}")

    for i, sample in enumerate(qualitative_samples):
        status = "✓ RECOVERED" if sample["recovered"] else "✗ STILL FAILED"
        print(f"\n[{i+1}] Index {sample['index']} - {status}")
        print(f"    Question: {sample['question'][:80]}...")
        print(f"    Ground Truth:    {sample['ground_truth']}")
        print(f"    Failed 4-bit:    {sample['failed_4bit_answer']}")
        print(f"    Hybrid Answer:   {sample['hybrid_answer']}")

    # Summary
    print(f"\n{'='*70}")
    print("SURGICAL REPAIR SUMMARY")
    print(f"{'='*70}")
    print(f"  Recovery Rate: {recovered_count}/{total_flipped} ({recovery_rate:.1%})")

    if recovery_rate > 0.5:
        print(f"\n  ✓ STRONG EVIDENCE: Mixed-precision patch recovers majority of failures")
    elif recovery_rate > 0.25:
        print(f"\n  ~ MODERATE EVIDENCE: Partial recovery suggests induction heads are involved")
    else:
        print(f"\n  ✗ WEAK EVIDENCE: Low recovery rate - may need different target layers")

    print(f"{'='*70}")
    print("Done!")
    print(f"{'='*70}")

    return report


if __name__ == "__main__":
    report = main()
