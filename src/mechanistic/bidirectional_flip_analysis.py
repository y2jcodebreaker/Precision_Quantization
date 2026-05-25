"""
Bidirectional Flip Analysis with Confidence Scoring
=====================================================

This experiment proves the "redistribution, not damage" hypothesis for
quantization effects on transformer reasoning.

Key Insight:
- Previous experiments only tracked "forward flips" (FP16 correct → 4-bit wrong)
- But if net accuracy is unchanged, there must be "reverse flips" too
- This script captures BOTH directions and measures model confidence

Analysis Dimensions:
1. Flip counts in both directions per dataset
2. Logit confidence for correct answer under FP16 vs 4-bit (ARC only — single letter)
3. Confidence distribution of forward vs reverse flipped examples
4. Example-level characterization: are forward flips low-confidence boundary cases?

Datasets:
- GSM8K: Math word problems (expected: redistribution, ~equal forward/reverse flips)
- ARC-Challenge: Science reasoning (expected: genuine damage, forward >> reverse)
- MATH: Competition math (expected: redistribution)

Output: bidirectional_flip_report.json

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
import torch.nn.functional as F
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
    """Configuration for bidirectional flip analysis."""
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_file: str = "bidirectional_flip_report.json"
    checkpoint_file: str = "bidirectional_flip_checkpoint.json"

    num_examples_per_dataset: int = 500
    batch_size: int = 4
    max_new_tokens: int = 512  # Full length for GSM8K/MATH reasoning
    max_new_tokens_arc: int = 32  # Short for ARC (just need the letter)
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
        # Last number fallback
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
            text, re.IGNORECASE
        )
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
    """Unload model and free memory."""
    del model
    if tokenizer:
        del tokenizer
    aggressive_cleanup()


# =============================================================================
# Inference with Confidence Scoring
# =============================================================================

def get_arc_confidence(
    model,
    tokenizer,
    examples: List[Dict],
    batch_size: int,
) -> Dict[int, Dict[str, float]]:
    """
    For ARC examples, compute the model's logit probability for each
    answer choice (A, B, C, D) given the prompt.

    Returns: {example_index: {"A": 0.45, "B": 0.30, "C": 0.15, "D": 0.10}}
    """
    print("    Computing ARC confidence scores...")
    confidence_map = {}

    # Get token IDs for answer labels
    # Encode each label individually to get its token ID
    label_token_ids = {}
    for label in ["A", "B", "C", "D", "1", "2", "3", "4"]:
        tokens = tokenizer.encode(label, add_special_tokens=False)
        if tokens:
            label_token_ids[label] = tokens[0]

    num_batches = (len(examples) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="      Confidence"):
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
            outputs = model(**inputs)
            # Get logits for the last token position (next token prediction)
            # For each sequence, find the last non-padding position
            for i, example in enumerate(batch_examples):
                # Find last non-pad token
                attention_mask = inputs["attention_mask"][i]
                last_pos = attention_mask.sum().item() - 1

                # Get logits at that position
                logits = outputs.logits[i, last_pos, :]

                # Extract logits for each valid label
                valid_labels = example.get("choices_labels", ["A", "B", "C", "D"])
                label_logits = {}
                for label in valid_labels:
                    if label in label_token_ids:
                        label_logits[label] = logits[label_token_ids[label]].item()

                # Convert to probabilities via softmax over just the label logits
                if label_logits:
                    label_names = list(label_logits.keys())
                    label_vals = torch.tensor([label_logits[l] for l in label_names])
                    probs = F.softmax(label_vals, dim=0)

                    confidence_map[example["index"]] = {
                        label: prob.item()
                        for label, prob in zip(label_names, probs)
                    }

        if (batch_idx + 1) % 20 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return confidence_map


def run_inference(
    model,
    tokenizer,
    examples: List[Dict],
    batch_size: int,
    max_new_tokens: int,
    pass_name: str,
) -> List[Dict]:
    """Run batched inference on examples. Returns results with answers."""
    print(f"\n  Running {pass_name} inference...")
    results = []
    num_batches = (len(examples) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"    {pass_name}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(examples))
        batch_examples = examples[start_idx:end_idx]

        # Use shorter max_new_tokens for ARC
        effective_max_tokens = max_new_tokens
        if batch_examples[0]["dataset"] == "arc":
            effective_max_tokens = 32

        prompts = [format_prompt(ex) for ex in batch_examples]

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
                "model_answer": model_answer,
                "ground_truth": ground_truth,
                "response": response[:300],  # Truncate for storage
            })

        if (batch_idx + 1) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / len(results) if results else 0
    print(f"    {pass_name} Accuracy: {accuracy:.2%} ({correct_count}/{len(results)})")

    return results


# =============================================================================
# Analysis
# =============================================================================

def analyze_bidirectional_flips(
    fp16_results: List[Dict],
    int4_results: List[Dict],
    dataset_name: str,
    fp16_confidence: Optional[Dict] = None,
    int4_confidence: Optional[Dict] = None,
) -> Dict:
    """
    Comprehensive bidirectional flip analysis for one dataset.

    Categorizes every example into one of 4 quadrants:
    - Both Correct (stable correct)
    - Both Incorrect (stable incorrect)
    - Forward Flip: FP16 correct → 4-bit wrong (quantization hurts)
    - Reverse Flip: FP16 wrong → 4-bit correct (quantization helps)
    """
    fp16_by_idx = {r["index"]: r for r in fp16_results}
    int4_by_idx = {r["index"]: r for r in int4_results}

    # Categorize each example
    both_correct = []
    both_incorrect = []
    forward_flips = []  # FP16 correct → 4-bit wrong
    reverse_flips = []  # FP16 wrong → 4-bit correct

    for idx in sorted(fp16_by_idx.keys()):
        fp16_r = fp16_by_idx[idx]
        int4_r = int4_by_idx[idx]

        entry = {
            "index": idx,
            "ground_truth": fp16_r["ground_truth"],
            "fp16_answer": fp16_r["model_answer"],
            "int4_answer": int4_r["model_answer"],
            "fp16_correct": fp16_r["correct"],
            "int4_correct": int4_r["correct"],
        }

        # Add confidence data if available (ARC)
        if fp16_confidence and idx in fp16_confidence:
            entry["fp16_confidence"] = fp16_confidence[idx]
            correct_label = fp16_r["ground_truth"]
            entry["fp16_correct_prob"] = fp16_confidence[idx].get(correct_label, 0.0)
        if int4_confidence and idx in int4_confidence:
            entry["int4_confidence"] = int4_confidence[idx]
            correct_label = int4_r["ground_truth"]
            entry["int4_correct_prob"] = int4_confidence[idx].get(correct_label, 0.0)

        if fp16_r["correct"] and int4_r["correct"]:
            both_correct.append(entry)
        elif not fp16_r["correct"] and not int4_r["correct"]:
            both_incorrect.append(entry)
        elif fp16_r["correct"] and not int4_r["correct"]:
            forward_flips.append(entry)
        else:  # not fp16 correct and int4 correct
            reverse_flips.append(entry)

    total = len(fp16_by_idx)
    fp16_acc = (len(both_correct) + len(forward_flips)) / total
    int4_acc = (len(both_correct) + len(reverse_flips)) / total

    analysis = {
        "dataset": dataset_name,
        "total_examples": total,
        "fp16_accuracy": fp16_acc,
        "int4_accuracy": int4_acc,
        "accuracy_drop": fp16_acc - int4_acc,
        "counts": {
            "both_correct": len(both_correct),
            "both_incorrect": len(both_incorrect),
            "forward_flips": len(forward_flips),
            "reverse_flips": len(reverse_flips),
        },
        "forward_flip_indices": [e["index"] for e in forward_flips],
        "reverse_flip_indices": [e["index"] for e in reverse_flips],
        # Save first 5 examples of each flip type for inspection
        "forward_flip_samples": forward_flips[:5],
        "reverse_flip_samples": reverse_flips[:5],
    }

    # Confidence analysis for ARC
    if fp16_confidence and int4_confidence:
        # For forward flips: what was FP16's confidence on the correct answer?
        fwd_fp16_conf = [
            e["fp16_correct_prob"] for e in forward_flips
            if "fp16_correct_prob" in e
        ]
        fwd_int4_conf = [
            e["int4_correct_prob"] for e in forward_flips
            if "int4_correct_prob" in e
        ]

        # For reverse flips: what was FP16's confidence on the correct answer?
        rev_fp16_conf = [
            e["fp16_correct_prob"] for e in reverse_flips
            if "fp16_correct_prob" in e
        ]
        rev_int4_conf = [
            e["int4_correct_prob"] for e in reverse_flips
            if "int4_correct_prob" in e
        ]

        # For stable correct: confidence levels
        stable_fp16_conf = [
            e["fp16_correct_prob"] for e in both_correct
            if "fp16_correct_prob" in e
        ]
        stable_int4_conf = [
            e["int4_correct_prob"] for e in both_correct
            if "int4_correct_prob" in e
        ]

        def safe_mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        def safe_median(lst):
            if not lst:
                return 0.0
            s = sorted(lst)
            n = len(s)
            if n % 2 == 0:
                return (s[n // 2 - 1] + s[n // 2]) / 2
            return s[n // 2]

        analysis["confidence_analysis"] = {
            "forward_flips": {
                "count": len(fwd_fp16_conf),
                "mean_fp16_correct_prob": safe_mean(fwd_fp16_conf),
                "median_fp16_correct_prob": safe_median(fwd_fp16_conf),
                "mean_int4_correct_prob": safe_mean(fwd_int4_conf),
                "median_int4_correct_prob": safe_median(fwd_int4_conf),
                "fp16_correct_probs": sorted(fwd_fp16_conf),
                "int4_correct_probs": sorted(fwd_int4_conf),
            },
            "reverse_flips": {
                "count": len(rev_fp16_conf),
                "mean_fp16_correct_prob": safe_mean(rev_fp16_conf),
                "median_fp16_correct_prob": safe_median(rev_fp16_conf),
                "mean_int4_correct_prob": safe_mean(rev_int4_conf),
                "median_int4_correct_prob": safe_median(rev_int4_conf),
                "fp16_correct_probs": sorted(rev_fp16_conf),
                "int4_correct_probs": sorted(rev_int4_conf),
            },
            "stable_correct": {
                "count": len(stable_fp16_conf),
                "mean_fp16_correct_prob": safe_mean(stable_fp16_conf),
                "median_fp16_correct_prob": safe_median(stable_fp16_conf),
                "mean_int4_correct_prob": safe_mean(stable_int4_conf),
                "median_int4_correct_prob": safe_median(stable_int4_conf),
            },
            "hypothesis_test": {
                "description": (
                    "If forward-flipped examples have lower FP16 confidence than "
                    "stable-correct examples, this supports the 'boundary perturbation' "
                    "hypothesis: quantization noise pushes marginal cases over the "
                    "decision boundary."
                ),
                "forward_flip_mean_confidence": safe_mean(fwd_fp16_conf),
                "stable_correct_mean_confidence": safe_mean(stable_fp16_conf),
                "confidence_gap": safe_mean(stable_fp16_conf) - safe_mean(fwd_fp16_conf),
                "boundary_perturbation_supported": (
                    safe_mean(fwd_fp16_conf) < safe_mean(stable_fp16_conf)
                    if fwd_fp16_conf and stable_fp16_conf else None
                ),
            },
        }

    return analysis


def print_analysis(analysis: Dict):
    """Pretty-print analysis results."""
    ds = analysis["dataset"].upper()
    counts = analysis["counts"]

    print(f"\n  {'='*60}")
    print(f"  {ds} — Bidirectional Flip Analysis")
    print(f"  {'='*60}")
    print(f"    Total Examples:   {analysis['total_examples']}")
    print(f"    FP16 Accuracy:    {analysis['fp16_accuracy']:.2%}")
    print(f"    4-bit Accuracy:   {analysis['int4_accuracy']:.2%}")
    print(f"    Net Drop:         {analysis['accuracy_drop']:+.2%}")
    print()
    print(f"    ┌─────────────────────────────────────┐")
    print(f"    │  Both Correct:    {counts['both_correct']:4d}  (stable)      │")
    print(f"    │  Both Incorrect:  {counts['both_incorrect']:4d}  (stable)      │")
    print(f"    │  Forward Flips:   {counts['forward_flips']:4d}  (quant hurts) │")
    print(f"    │  Reverse Flips:   {counts['reverse_flips']:4d}  (quant helps) │")
    print(f"    └─────────────────────────────────────┘")

    net_flip = counts["forward_flips"] - counts["reverse_flips"]
    if abs(net_flip) <= 3:
        verdict = "REDISTRIBUTION (balanced flips)"
    elif net_flip > 0:
        verdict = f"GENUINE DAMAGE (net {net_flip} examples lost)"
    else:
        verdict = f"QUANTIZATION BENEFIT (net {abs(net_flip)} examples gained)"

    print(f"    Verdict: {verdict}")

    # Confidence analysis
    if "confidence_analysis" in analysis:
        ca = analysis["confidence_analysis"]
        ht = ca["hypothesis_test"]
        print(f"\n    Confidence Analysis (ARC logit probabilities):")
        print(f"    ─────────────────────────────────────────────")
        print(f"    Forward flips — Mean FP16 confidence:  {ca['forward_flips']['mean_fp16_correct_prob']:.3f}")
        print(f"    Stable correct — Mean FP16 confidence: {ca['stable_correct']['mean_fp16_correct_prob']:.3f}")
        print(f"    Confidence gap:                        {ht['confidence_gap']:.3f}")
        print(f"    Boundary perturbation supported:       {ht['boundary_perturbation_supported']}")

        if ca["reverse_flips"]["count"] > 0:
            print(f"\n    Reverse flips — Mean FP16 confidence:  {ca['reverse_flips']['mean_fp16_correct_prob']:.3f}")
            print(f"    Reverse flips — Mean 4-bit confidence: {ca['reverse_flips']['mean_int4_correct_prob']:.3f}")


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(data: Dict, filepath: str):
    """Save checkpoint to disk."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Checkpoint saved to: {filepath}")


def load_checkpoint(filepath: str) -> Optional[Dict]:
    """Load checkpoint if it exists."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        print(f"  Loaded checkpoint from: {filepath}")
        return data
    return None


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    config = Config()

    print("=" * 70)
    print("Bidirectional Flip Analysis: Redistribution vs Damage")
    print("=" * 70)
    print(f"\nModel: {config.model_name}")
    print(f"Examples per dataset: {config.num_examples_per_dataset}")
    print(f"Device: {config.device}")

    torch.manual_seed(config.seed)

    # =========================================================================
    # Check for existing checkpoint
    # =========================================================================
    checkpoint = load_checkpoint(config.checkpoint_file)
    resumed_phase = 0
    fp16_results = {}
    int4_results = {}
    fp16_arc_confidence = {}
    int4_arc_confidence = {}

    if checkpoint:
        resumed_phase = checkpoint.get("completed_phase", 0)
        print(f"  Resuming from phase {resumed_phase + 1} (phase {resumed_phase} completed)")

        if resumed_phase >= 1:
            fp16_results = checkpoint["fp16_results"]
            fp16_arc_confidence = {
                int(k): v for k, v in checkpoint["fp16_arc_confidence"].items()
            }
        if resumed_phase >= 2:
            int4_results = checkpoint["int4_results"]
            int4_arc_confidence = {
                int(k): v for k, v in checkpoint["int4_arc_confidence"].items()
            }

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
    # Pass 1: FP16 Evaluation + ARC Confidence
    # =========================================================================
    if resumed_phase < 1:
        print(f"\n{'='*70}")
        print("PASS 1: FP16 Evaluation")
        print(f"{'='*70}")

        model, tokenizer = load_fp16_model(config.model_name)

        fp16_results = {}
        for ds_name, examples in all_examples.items():
            print(f"\n  Testing {ds_name}...")
            res = run_inference(
                model, tokenizer, examples,
                config.batch_size, config.max_new_tokens,
                f"FP16-{ds_name}"
            )
            fp16_results[ds_name] = res

        # ARC confidence scoring under FP16
        print(f"\n  Computing FP16 confidence scores for ARC...")
        fp16_arc_confidence = get_arc_confidence(
            model, tokenizer, all_examples["arc"], config.batch_size
        )
        print(f"    Got confidence for {len(fp16_arc_confidence)} ARC examples")

        unload_model(model, tokenizer)

        # Save checkpoint after Phase 1
        save_checkpoint({
            "completed_phase": 1,
            "fp16_results": fp16_results,
            "fp16_arc_confidence": {
                str(k): v for k, v in fp16_arc_confidence.items()
            },
        }, config.checkpoint_file)
    else:
        print(f"\n  Skipping PASS 1 (loaded from checkpoint)")

    # =========================================================================
    # Pass 2: 4-bit Evaluation + ARC Confidence
    # =========================================================================
    if resumed_phase < 2:
        print(f"\n{'='*70}")
        print("PASS 2: 4-bit Evaluation")
        print(f"{'='*70}")

        model, tokenizer = load_4bit_model(config.model_name)

        int4_results = {}
        for ds_name, examples in all_examples.items():
            print(f"\n  Testing {ds_name}...")
            res = run_inference(
                model, tokenizer, examples,
                config.batch_size, config.max_new_tokens,
                f"4bit-{ds_name}"
            )
            int4_results[ds_name] = res

        # ARC confidence scoring under 4-bit
        print(f"\n  Computing 4-bit confidence scores for ARC...")
        int4_arc_confidence = get_arc_confidence(
            model, tokenizer, all_examples["arc"], config.batch_size
        )
        print(f"    Got confidence for {len(int4_arc_confidence)} ARC examples")

        unload_model(model, tokenizer)

        # Save checkpoint after Phase 2
        save_checkpoint({
            "completed_phase": 2,
            "fp16_results": fp16_results,
            "fp16_arc_confidence": {
                str(k): v for k, v in fp16_arc_confidence.items()
            },
            "int4_results": int4_results,
            "int4_arc_confidence": {
                str(k): v for k, v in int4_arc_confidence.items()
            },
        }, config.checkpoint_file)
    else:
        print(f"\n  Skipping PASS 2 (loaded from checkpoint)")

    # =========================================================================
    # Analysis
    # =========================================================================
    print(f"\n{'='*70}")
    print("ANALYSIS: Bidirectional Flip Characterization")
    print(f"{'='*70}")

    all_analyses = {}

    for ds_name in ["gsm8k", "arc", "math"]:
        fp16_conf = fp16_arc_confidence if ds_name == "arc" else None
        int4_conf = int4_arc_confidence if ds_name == "arc" else None

        analysis = analyze_bidirectional_flips(
            fp16_results[ds_name],
            int4_results[ds_name],
            ds_name,
            fp16_confidence=fp16_conf,
            int4_confidence=int4_conf,
        )

        all_analyses[ds_name] = analysis
        print_analysis(analysis)

    # =========================================================================
    # Cross-Dataset Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("CROSS-DATASET SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Dataset':<10} {'Forward':>8} {'Reverse':>8} {'Net':>6} {'Verdict'}")
    print(f"  {'─'*60}")

    for ds_name in ["gsm8k", "arc", "math"]:
        a = all_analyses[ds_name]
        c = a["counts"]
        net = c["forward_flips"] - c["reverse_flips"]

        if abs(net) <= 3:
            verdict = "Redistribution"
        elif net > 0:
            verdict = "Genuine Damage"
        else:
            verdict = "Quant Benefits"

        print(f"  {ds_name:<10} {c['forward_flips']:>8} {c['reverse_flips']:>8} {net:>+6} {verdict}")

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
        },
        "analyses": all_analyses,
        "summary": {
            ds_name: {
                "fp16_accuracy": all_analyses[ds_name]["fp16_accuracy"],
                "int4_accuracy": all_analyses[ds_name]["int4_accuracy"],
                "forward_flips": all_analyses[ds_name]["counts"]["forward_flips"],
                "reverse_flips": all_analyses[ds_name]["counts"]["reverse_flips"],
                "net_flip": (
                    all_analyses[ds_name]["counts"]["forward_flips"]
                    - all_analyses[ds_name]["counts"]["reverse_flips"]
                ),
            }
            for ds_name in ["gsm8k", "arc", "math"]
        },
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
