"""
Baseline Comparison: NF4 vs GPTQ vs AWQ (Phase 20)
====================================================

Evaluates pre-quantized GPTQ and AWQ models from HuggingFace on the same
benchmarks (GSM8K, ARC-Challenge, MATH) using identical prompts and answer
extraction to enable fair comparison with our NF4 results.

Purpose:
  Context for the mechanistic paper — show that NF4 damage patterns are
  comparable to other 4-bit methods, so our probe is not an artifact of
  a uniquely bad quantization scheme.

Models:
  - FP16:  meta-llama/Llama-3.1-8B-Instruct
  - NF4:   meta-llama/Llama-3.1-8B-Instruct (bitsandbytes NF4)
  - GPTQ:  hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4
  - AWQ:   hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4

Output: results/baseline_comparison.json

Authentication:
---------------
Set HF_TOKEN environment variable for model access.

Dependencies:
  pip install transformers datasets bitsandbytes optimum
  Note: Do NOT install auto-gptq or autoawq — transformers uses native
  GPTQ/AWQ support via optimum. If either package is already installed,
  remove them first:  pip uninstall auto-gptq autoawq -y
"""

import json
import math
import os
import re
import gc
import hashlib
import signal
import time
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
    """Configuration for baseline comparison."""
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    gptq_model: str = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
    awq_model: str = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

    output_dir: str = "results"
    output_file: str = os.path.join("results", "baseline_comparison.json")
    checkpoint_file: str = os.path.join("results", "baseline_comparison_ckpt.json")

    # Dataset sizes (match earlier phases)
    gsm8k_examples: int = 500
    arc_examples: int = 500
    math_examples: int = 200
    winogrande_examples: int = 500
    piqa_examples: int = 500

    # WikiText-2 perplexity config
    wikitext_seq_len: int = 2048
    wikitext_num_seqs: int = 128

    batch_size: int = 64       # Quantized models (NF4, GPTQ, AWQ)
    fp16_batch_size: int = 16  # FP16 uses more memory, start lower
    ppl_batch_size: int = 4    # Full 2048-token sequences, keep small
    checkpoint_every: int = 50  # Save partial results every N examples
    max_new_tokens_mc: int = 64     # Multiple choice (ARC, WinoGrande, PIQA)
    max_new_tokens_gsm8k: int = 1024
    max_new_tokens_math: int = 1024

    def fingerprint(self) -> str:
        """Config hash for checkpoint validation."""
        key = (
            f"{self.base_model}|{self.gptq_model}|{self.awq_model}|"
            f"{self.gsm8k_examples}|{self.arc_examples}|{self.math_examples}|"
            f"{self.winogrande_examples}|{self.piqa_examples}|v2"
        )
        return hashlib.md5(key.encode()).hexdigest()[:12]


# =============================================================================
# Checkpoint Management
# =============================================================================

def save_checkpoint(path: str, data: dict) -> None:
    """Atomic checkpoint write with backup."""
    tmp_path = path + ".tmp"
    bak_path = path + ".bak"
    if os.path.exists(path):
        try:
            os.replace(path, bak_path)
        except OSError:
            pass
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def load_checkpoint(path: str, expected_fingerprint: str) -> Optional[dict]:
    """Load checkpoint with fingerprint validation."""
    for candidate in [path, path + ".bak"]:
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate) as f:
                data = json.load(f)
            if data.get("fingerprint") == expected_fingerprint:
                print(f"  Resumed from checkpoint: {candidate}")
                return data
            print(f"  Checkpoint fingerprint mismatch, ignoring: {candidate}")
        except (json.JSONDecodeError, KeyError):
            print(f"  Corrupt checkpoint, ignoring: {candidate}")
    return None


# Global for signal handler
_checkpoint_state: dict = {}
_checkpoint_path: str = ""


def _signal_handler(signum, frame):
    """Save checkpoint on interrupt."""
    if _checkpoint_state and _checkpoint_path:
        print(f"\n[Signal {signum}] Saving checkpoint before exit...")
        save_checkpoint(_checkpoint_path, _checkpoint_state)
        print("  Checkpoint saved.")
    raise SystemExit(1)


# =============================================================================
# Memory Management
# =============================================================================

def aggressive_cleanup():
    """Aggressively clear GPU memory."""
    gc.collect()
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def get_peak_gpu_memory_mb() -> float:
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


# =============================================================================
# Dataset Loading
# =============================================================================

def load_gsm8k(num_examples: int) -> List[Dict]:
    """Load GSM8K test examples."""
    print(f"  Loading GSM8K (first {num_examples} test examples)...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
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
    print(f"    Loaded {len(examples)} examples")
    return examples


def load_arc_challenge(num_examples: int) -> List[Dict]:
    """Load ARC-Challenge test examples."""
    print(f"  Loading ARC-Challenge (first {num_examples} test examples)...")
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    examples = []
    for i, item in enumerate(dataset):
        if i >= num_examples:
            break
        labels = item["choices"]["label"]
        texts = item["choices"]["text"]
        options_text = "\n".join(
            f"{label}. {text}" for label, text in zip(labels, texts)
        )
        examples.append({
            "index": i,
            "id": item["id"],
            "question": item["question"],
            "choices_labels": labels,
            "choices_texts": texts,
            "options_text": options_text,
            "answer": item["answerKey"],
            "dataset": "arc_challenge",
        })
    print(f"    Loaded {len(examples)} examples")
    return examples


def load_math(num_examples: int) -> List[Dict]:
    """Load MATH test examples."""
    print(f"  Loading MATH (first {num_examples} test examples)...")
    try:
        dataset = load_dataset("hendrycks/competition_math", split="test")
    except Exception:
        try:
            dataset = load_dataset(
                "DigitalLearningGmbH/MATH-lighteval", split="test"
            )
        except Exception:
            print("    Using train split as fallback...")
            dataset = load_dataset("hendrycks/competition_math", split="train")

    examples = []
    for i, item in enumerate(dataset):
        if i >= num_examples:
            break
        examples.append({
            "index": i,
            "problem": item["problem"],
            "solution": item["solution"],
            "level": item.get("level", "unknown"),
            "type": item.get("type", "unknown"),
            "dataset": "math",
        })
    print(f"    Loaded {len(examples)} examples")
    return examples


def load_winogrande(num_examples: int) -> List[Dict]:
    """Load WinoGrande validation examples (test has no labels)."""
    print(f"  Loading WinoGrande (first {num_examples} validation examples)...")
    dataset = load_dataset(
        "allenai/winogrande", "winogrande_xl", split="validation"
    )
    examples = []
    for item in dataset:
        if len(examples) >= num_examples:
            break
        answer = item["answer"].strip()
        if answer not in ("1", "2"):
            continue
        examples.append({
            "index": len(examples),
            "sentence": item["sentence"],
            "option1": item["option1"],
            "option2": item["option2"],
            "answer": answer,  # "1" or "2"
            "dataset": "winogrande",
        })
    print(f"    Loaded {len(examples)} examples")
    return examples


def load_piqa(num_examples: int) -> List[Dict]:
    """Load PIQA validation examples (test has no labels)."""
    print(f"  Loading PIQA (first {num_examples} validation examples)...")
    dataset = load_dataset("lighteval/piqa", split="validation")
    examples = []
    for item in dataset:
        if len(examples) >= num_examples:
            break
        label = item["label"]
        if label not in (0, 1):
            continue
        examples.append({
            "index": len(examples),
            "goal": item["goal"],
            "sol1": item["sol1"],
            "sol2": item["sol2"],
            "answer": str(label),  # "0" or "1"
            "dataset": "piqa",
        })
    print(f"    Loaded {len(examples)} examples")
    return examples


# =============================================================================
# WikiText-2 Perplexity
# =============================================================================

def load_wikitext2_sequences(seq_len: int, num_seqs: int, tokenizer) -> torch.Tensor:
    """Load WikiText-2 test set and prepare fixed-length sequences for PPL."""
    print(f"  Loading WikiText-2 (seq_len={seq_len}, num_seqs={num_seqs})...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(item["text"] for item in dataset if item["text"].strip())
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings["input_ids"][0]
    total_tokens = seq_len * num_seqs
    if input_ids.size(0) < total_tokens:
        num_seqs = input_ids.size(0) // seq_len
        print(f"    Adjusted to {num_seqs} sequences (not enough tokens)")
    sequences = input_ids[: seq_len * num_seqs].view(num_seqs, seq_len)
    print(f"    Prepared {num_seqs} sequences of {seq_len} tokens")
    return sequences


def compute_perplexity(
    model, sequences: torch.Tensor, batch_size: int
) -> float:
    """Compute perplexity on pre-tokenized sequences.

    HuggingFace shifts labels internally, so the loss is averaged over
    seq_len-1 prediction steps per sequence (not seq_len).
    """
    total_nll = 0.0
    total_tokens = 0
    num_seqs = sequences.size(0)
    seq_len = sequences.size(1)
    effective_bs = batch_size
    idx = 0

    pbar = tqdm(total=num_seqs, desc="  PPL")
    while idx < num_seqs:
        bs = min(effective_bs, num_seqs - idx)
        batch = sequences[idx : idx + bs].to(next(model.parameters()).device)
        try:
            with torch.no_grad():
                outputs = model(batch, labels=batch)
            # loss is mean NLL; HF shifts internally so it's over seq_len-1 tokens
            num_tokens = bs * (seq_len - 1)
            total_nll += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
            pbar.update(bs)
            idx += bs
        except torch.cuda.OutOfMemoryError:
            aggressive_cleanup()
            effective_bs = max(1, effective_bs // 2)
            print(f"\n    OOM in PPL — reducing batch to {effective_bs}")
            # retry same idx

    pbar.close()
    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    return round(ppl, 4)


# =============================================================================
# Prompt Formatting (identical to earlier phases)
# =============================================================================

def format_gsm8k_prompt(example: Dict, tokenizer) -> str:
    """Format GSM8K prompt using chat template."""
    messages = [{"role": "user", "content": (
        "Solve this math problem step by step. "
        "End with your final numerical answer after '####'.\n\n"
        f"{example['question']}"
    )}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def format_arc_prompt(example: Dict, tokenizer) -> str:
    """Format ARC-Challenge prompt using chat template."""
    labels_str = ", ".join(example["choices_labels"])
    messages = [{"role": "user", "content": (
        "Answer the following science question by selecting the correct option.\n\n"
        f"Question: {example['question']}\n\n"
        f"Options:\n{example['options_text']}\n\n"
        f"Respond with just the letter of the correct answer ({labels_str})."
    )}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def format_math_prompt(example: Dict, tokenizer) -> str:
    """Format MATH prompt using chat template."""
    messages = [{"role": "user", "content": (
        "Solve this mathematics problem. Show your work and "
        "put your final answer in \\boxed{}.\n\n"
        f"{example['problem']}"
    )}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def format_winogrande_prompt(example: Dict, tokenizer) -> str:
    """Format WinoGrande prompt using chat template."""
    messages = [{"role": "user", "content": (
        "Complete the sentence by choosing the correct option.\n\n"
        f"Sentence: {example['sentence']}\n\n"
        f"A. {example['option1']}\n"
        f"B. {example['option2']}\n\n"
        "Respond with just the letter (A or B)."
    )}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def format_piqa_prompt(example: Dict, tokenizer) -> str:
    """Format PIQA prompt using chat template."""
    messages = [{"role": "user", "content": (
        "Which solution best achieves the goal?\n\n"
        f"Goal: {example['goal']}\n\n"
        f"A. {example['sol1']}\n"
        f"B. {example['sol2']}\n\n"
        "Respond with just the letter (A or B)."
    )}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# =============================================================================
# Answer Extraction (identical to earlier phases)
# =============================================================================

def extract_gsm8k_ground_truth(example: Dict) -> str:
    """Extract numerical answer from GSM8K ground truth."""
    match = re.search(r"####\s*([+-]?\d[\d,]*\.?\d*)", example["answer"])
    if match:
        return match.group(1).replace(",", "")
    return ""


def extract_gsm8k_answer(response: str) -> Optional[str]:
    """Extract numerical answer from model response."""
    text = response.replace("**", "").replace("*", "")

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
        r"(?:therefore|thus|so)[,\s]+\$?([+-]?\d[\d,]*\.?\d*)",
        r"total(?:\s+is|\s+of|:)\s*\$?([+-]?\d[\d,]*\.?\d*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).replace(",", "")

    numbers = re.findall(r"(?<!\d)([+-]?\d[\d,]*\.?\d*)(?!\d)", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def check_gsm8k_match(model_answer: Optional[str], ground_truth: str) -> bool:
    """Check GSM8K answer match."""
    if model_answer is None or not ground_truth:
        return False
    try:
        return abs(float(model_answer) - float(ground_truth)) < 1e-6
    except ValueError:
        return model_answer.strip() == ground_truth.strip()


def extract_arc_answer(response: str, valid_labels: List[str]) -> Optional[str]:
    """Extract multiple-choice answer letter from model response."""
    text = response.strip()
    valid_set = {label.upper() for label in valid_labels}
    valid_pattern = "".join(valid_labels)

    if text and text[0].upper() in valid_set:
        return text[0].upper()

    patterns = [
        rf"(?:the\s+)?answer\s+is[:\s]*([{valid_pattern}])\b",
        rf"\b([{valid_pattern}])\s+is\s+(?:the\s+)?(?:correct|right)",
        rf"correct\s+answer\s+is[:\s]*([{valid_pattern}])\b",
        rf"(?:I\s+)?(?:would\s+)?(?:choose|select|pick)\s+([{valid_pattern}])\b",
        rf"(?:option|choice)\s+([{valid_pattern}])\b",
        rf"^([{valid_pattern}])\s*[\.:\)]",
        rf"\*\*([{valid_pattern}])\*\*",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    for label in valid_labels:
        if re.search(rf"\b{label}\b", text, re.IGNORECASE):
            return label.upper()
    return None


def extract_math_ground_truth(example: Dict) -> str:
    """Extract answer from MATH solution."""
    match = re.search(r"\\boxed\{([^}]+)\}", example["solution"])
    if match:
        return _normalize_math(match.group(1))
    return ""


def extract_math_answer(response: str) -> Optional[str]:
    """Extract answer from MATH model response."""
    match = re.search(r"\\boxed\{([^}]+)\}", response)
    if match:
        return _normalize_math(match.group(1))

    patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*([^\n\.]+)",
        r"=\s*([^\n]+?)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return _normalize_math(match.group(1))
    return None


def _normalize_math(answer: str) -> str:
    """Normalize MATH answer for comparison."""
    answer = answer.strip()
    answer = re.sub(r"\\(?:text|mathrm|mathbf)\{([^}]+)\}", r"\1", answer)
    answer = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)", "", answer)
    answer = re.sub(r"\\,|\\;|\\:|\\!", "", answer)
    answer = answer.replace("\\$", "$").replace("\\%", "%")
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer


def check_math_match(model_answer: Optional[str], ground_truth: str) -> bool:
    """Check MATH answer match."""
    if model_answer is None or not ground_truth:
        return False
    model_norm = _normalize_math(model_answer)
    truth_norm = _normalize_math(ground_truth)
    if model_norm == truth_norm:
        return True
    try:
        m = float(re.sub(r"[^\d.\-+]", "", model_norm))
        t = float(re.sub(r"[^\d.\-+]", "", truth_norm))
        return abs(m - t) < 1e-6
    except (ValueError, TypeError):
        pass
    return model_norm.lower() == truth_norm.lower()


def extract_ab_answer(response: str, valid_labels: List[str]) -> Optional[str]:
    """Extract A/B answer from model response (for WinoGrande, PIQA)."""
    text = response.strip()
    valid_set = {label.upper() for label in valid_labels}
    valid_pattern = "".join(valid_labels)

    if text and text[0].upper() in valid_set:
        return text[0].upper()

    patterns = [
        rf"(?:the\s+)?answer\s+is[:\s]*([{valid_pattern}])\b",
        rf"\b([{valid_pattern}])\s+is\s+(?:the\s+)?(?:correct|right|better)",
        rf"(?:I\s+)?(?:would\s+)?(?:choose|select|pick)\s+([{valid_pattern}])\b",
        rf"(?:option|choice|solution)\s+([{valid_pattern}])\b",
        rf"^([{valid_pattern}])\s*[\.:\)]",
        rf"\*\*([{valid_pattern}])\*\*",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    for label in valid_labels:
        if re.search(rf"\b{label}\b", text, re.IGNORECASE):
            return label.upper()
    return None


def extract_winogrande_gt(example: Dict) -> str:
    """Convert WinoGrande answer ('1'/'2') to 'A'/'B'."""
    return "A" if example["answer"] == "1" else "B"


def extract_piqa_gt(example: Dict) -> str:
    """Convert PIQA answer ('0'/'1') to 'A'/'B'."""
    return "A" if example["answer"] == "0" else "B"


def check_ab_match(pred: Optional[str], gt: str) -> bool:
    """Check A/B answer match."""
    if pred is None:
        return False
    return pred.upper() == gt.upper()


# =============================================================================
# Batched Evaluation (OOM-safe)
# =============================================================================

def _run_one_batch(
    model, tokenizer, prompts: List[str], max_new_tokens: int
) -> List[str]:
    """Run inference on a single batch, return generated texts."""
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=2048,
    ).to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    responses = []
    for output in outputs:
        generated = output[input_len:]
        responses.append(tokenizer.decode(generated, skip_special_tokens=True))
    return responses


def evaluate_dataset(
    model,
    tokenizer,
    examples: List[Dict],
    dataset_name: str,
    format_fn,
    extract_gt_fn,
    extract_answer_fn,
    check_match_fn,
    max_new_tokens: int,
    batch_size: int,
    partial_results: Optional[List[Dict]] = None,
    save_partial_fn=None,
    checkpoint_every: int = 50,
) -> Dict:
    """Evaluate a model on a dataset with OOM-safe batching and per-batch checkpointing.

    Args:
        partial_results: Existing results to resume from (from checkpoint).
        save_partial_fn: Callable(results, next_idx) — called every checkpoint_every examples.
        checkpoint_every: Save partial checkpoint every this many completed examples.
    """
    results = list(partial_results) if partial_results else []
    start_idx = len(results)  # Resume from where we left off
    idx = start_idx
    effective_bs = batch_size
    last_ckpt_count = start_idx

    pbar = tqdm(total=len(examples), initial=start_idx, desc=f"  {dataset_name}")

    while idx < len(examples):
        batch_end = min(idx + effective_bs, len(examples))
        batch = examples[idx:batch_end]
        prompts = [format_fn(ex, tokenizer) for ex in batch]

        try:
            responses = _run_one_batch(model, tokenizer, prompts, max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            aggressive_cleanup()
            effective_bs = max(1, effective_bs // 2)
            print(f"\n    OOM — reducing batch size to {effective_bs}")
            continue  # Retry same idx

        for ex, resp in zip(batch, responses):
            gt = extract_gt_fn(ex)
            pred = extract_answer_fn(resp)
            correct = check_match_fn(pred, gt)
            results.append({
                "index": ex["index"],
                "correct": correct,
                "predicted": pred,
                "ground_truth": gt,
            })
        pbar.update(len(batch))
        idx = batch_end

        # Per-batch checkpoint: save every checkpoint_every examples
        if save_partial_fn and (len(results) - last_ckpt_count) >= checkpoint_every:
            save_partial_fn(results, idx)
            last_ckpt_count = len(results)

    pbar.close()

    # Final checkpoint save
    if save_partial_fn:
        save_partial_fn(results, len(examples))

    correct_count = sum(r["correct"] for r in results)
    accuracy = correct_count / len(results) if results else 0.0
    return {
        "accuracy": round(accuracy * 100, 2),
        "correct": correct_count,
        "total": len(results),
        "results": results,
    }


# =============================================================================
# Model Loading
# =============================================================================

def load_model_fp16(model_name: str) -> Tuple:
    """Load model in FP16."""
    print(f"  Loading FP16: {model_name}")
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


def load_model_nf4(model_name: str) -> Tuple:
    """Load model with bitsandbytes NF4 quantization."""
    print(f"  Loading NF4: {model_name}")
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


def _check_gptq_env() -> None:
    """Warn if a conflicting auto_gptq install is detected."""
    try:
        import auto_gptq  # noqa: F401
        import importlib.metadata as meta
        ver = meta.version("auto-gptq")
        print(f"  WARNING: auto-gptq {ver} is installed and may conflict with "
              f"transformers native GPTQ support.\n"
              f"  If loading fails, run:  pip uninstall auto-gptq -y && pip install optimum")
    except Exception:
        pass


def _check_awq_env() -> None:
    """Warn if a conflicting autoawq install is detected."""
    try:
        import awq  # noqa: F401
        import importlib.metadata as meta
        ver = meta.version("autoawq")
        print(f"  WARNING: autoawq {ver} is installed and may conflict with "
              f"transformers native AWQ support.\n"
              f"  If loading fails, run:  pip uninstall autoawq -y && pip install optimum")
    except Exception:
        pass


def _patch_gptq_quantize_config(model_name: str) -> None:
    """Patch cached quantize_config.json for gptqmodel 6.x compatibility.

    hugging-quants GPTQ models have 'is_marlin_format' which was removed in
    gptqmodel 6.x. Replace it with the explicit 'format' field.
    """
    try:
        from huggingface_hub import hf_hub_download
        qconfig_path = hf_hub_download(model_name, "quantize_config.json")
        with open(qconfig_path) as f:
            qconfig = json.load(f)
        if "is_marlin_format" in qconfig:
            is_marlin = qconfig.pop("is_marlin_format")
            qconfig.setdefault("format", "marlin" if is_marlin else "gptq")
            with open(qconfig_path, "w") as f:
                json.dump(qconfig, f, indent=2)
            print(f"  Patched quantize_config.json: "
                  f"is_marlin_format={is_marlin} -> format={qconfig['format']}")
    except Exception as e:
        print(f"  Warning: Could not patch quantize_config.json: {e}")


def load_model_gptq(model_name: str) -> Tuple:
    """Load pre-quantized GPTQ model via gptqmodel with Triton backend.

    Patches the cached quantize_config.json to fix the deprecated
    'is_marlin_format' field before calling from_quantized.
    """
    print(f"  Loading GPTQ: {model_name}")
    _patch_gptq_quantize_config(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    from gptqmodel import GPTQModel, BACKEND  # requires: pip install gptqmodel
    model = GPTQModel.from_quantized(
        model_name,
        backend=BACKEND.TRITON,
        device="cuda:0",
    )
    model.eval()
    return model, tokenizer


def load_model_awq(model_name: str) -> Tuple:
    """Load pre-quantized AWQ model via autoawq's own API.

    Calls AutoAWQForCausalLM.from_quantized directly to bypass gptqmodel,
    which intercepts AutoModelForCausalLM for AWQ models but requires
    a compiled gptqmodel_exllamav2_awq_kernels that is not available here.
    Requires: pip install autoawq
    """
    print(f"  Loading AWQ: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    from awq import AutoAWQForCausalLM  # requires: pip install autoawq
    model = AutoAWQForCausalLM.from_quantized(
        model_name,
        fuse_layers=True,  # enables kernel fusion for ~2x speedup
    )
    model.eval()
    return model, tokenizer


# =============================================================================
# Main Pipeline
# =============================================================================

MODEL_CONFIGS = [
    ("fp16", "FP16 (baseline)", "load_fp16"),
    ("nf4", "NF4 (bitsandbytes)", "load_nf4"),
    ("gptq", "GPTQ-INT4", "load_gptq"),
    ("awq", "AWQ-INT4", "load_awq"),
]

def _extract_arc_answer(resp: str) -> Optional[str]:
    """ARC answer extractor with fixed ABCD labels (closure over valid set)."""
    return extract_arc_answer(resp, ["A", "B", "C", "D"])


def _extract_ab_answer(resp: str) -> Optional[str]:
    """WinoGrande / PIQA answer extractor with fixed AB labels."""
    return extract_ab_answer(resp, ("A", "B"))


DATASET_CONFIGS = [
    ("gsm8k", "GSM8K", "gsm8k_examples", "max_new_tokens_gsm8k",
     load_gsm8k, format_gsm8k_prompt, extract_gsm8k_ground_truth,
     extract_gsm8k_answer, check_gsm8k_match),
    ("arc_challenge", "ARC-Challenge", "arc_examples", "max_new_tokens_mc",
     load_arc_challenge, format_arc_prompt,
     lambda ex: ex["answer"], _extract_arc_answer, check_ab_match),
    ("math", "MATH", "math_examples", "max_new_tokens_math",
     load_math, format_math_prompt, extract_math_ground_truth,
     extract_math_answer, check_math_match),
    ("winogrande", "WinoGrande", "winogrande_examples", "max_new_tokens_mc",
     load_winogrande, format_winogrande_prompt, extract_winogrande_gt,
     _extract_ab_answer, check_ab_match),
    ("piqa", "PIQA", "piqa_examples", "max_new_tokens_mc",
     load_piqa, format_piqa_prompt, extract_piqa_gt,
     _extract_ab_answer, check_ab_match),
]


def run_comparison():
    """Run full baseline comparison pipeline."""
    global _checkpoint_state, _checkpoint_path

    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)
    _checkpoint_path = config.checkpoint_file

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    print("=" * 70)
    print("PHASE 20: Baseline Comparison — NF4 vs GPTQ vs AWQ")
    print("=" * 70)

    # Load checkpoint
    ckpt = load_checkpoint(config.checkpoint_file, config.fingerprint())
    if ckpt is None:
        ckpt = {
            "fingerprint": config.fingerprint(),
            "model_results": {},
            "completed_conditions": [],
        }

    # Load all datasets once
    print("\n[Step 1] Loading datasets...")
    datasets = {}
    for ds_key, ds_name, num_key, _, load_fn, *_ in DATASET_CONFIGS:
        datasets[ds_key] = load_fn(getattr(config, num_key))

    # WikiText-2 sequences will be tokenized per-model (tokenizer may differ)
    wikitext_raw = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="test"
    )
    wikitext_text = "\n\n".join(
        item["text"] for item in wikitext_raw if item["text"].strip()
    )
    print(f"  WikiText-2 text loaded ({len(wikitext_text):,} chars)")

    # Evaluate each model
    print("\n[Step 2] Evaluating models...")

    loader_map = {
        "load_fp16": lambda: load_model_fp16(config.base_model),
        "load_nf4": lambda: load_model_nf4(config.base_model),
        "load_gptq": lambda: load_model_gptq(config.gptq_model),
        "load_awq": lambda: load_model_awq(config.awq_model),
    }

    for model_key, model_name, loader_key in MODEL_CONFIGS:
        ppl_key = f"{model_key}__wikitext2_ppl"
        ds_done = all(
            f"{model_key}__{ds_key}" in ckpt["completed_conditions"]
            for ds_key, *_ in DATASET_CONFIGS
        )
        ppl_done = ppl_key in ckpt["completed_conditions"]
        if ds_done and ppl_done:
            print(f"\n  [{model_name}] All tasks complete (from checkpoint)")
            continue

        print(f"\n{'─' * 60}")
        print(f"  Loading {model_name}...")
        print(f"{'─' * 60}")

        try:
            model, tokenizer = loader_map[loader_key]()
        except Exception as e:
            print(f"  ERROR loading {model_name}: {e}")
            print(f"  Skipping {model_name}.")
            ckpt["model_results"].setdefault(model_key, {})["load_error"] = str(e)
            _checkpoint_state = ckpt
            save_checkpoint(config.checkpoint_file, ckpt)
            continue

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        mem_loaded = get_gpu_memory_mb()
        print(f"  GPU memory after load: {mem_loaded:.0f} MB")

        ckpt["model_results"].setdefault(model_key, {})
        ckpt["model_results"][model_key]["model_name"] = model_name
        ckpt["model_results"][model_key]["gpu_memory_mb"] = round(mem_loaded, 1)

        bs = (config.fp16_batch_size if model_key == "fp16"
              else config.batch_size)

        for ds_key, ds_name, _, max_tok_key, _, fmt_fn, gt_fn, ans_fn, match_fn in DATASET_CONFIGS:
            condition_key = f"{model_key}__{ds_key}"
            partial_key = f"{condition_key}__partial"

            if condition_key in ckpt["completed_conditions"]:
                print(f"  [{ds_name}] Already complete (from checkpoint)")
                continue

            # Resume from partial results if available
            partial_results = ckpt.get("partial", {}).get(partial_key)
            if partial_results:
                print(f"  [{ds_name}] Resuming from {len(partial_results)} saved examples")

            print(f"\n  Evaluating {model_name} on {ds_name}...")
            max_tok = getattr(config, max_tok_key)

            def _save_partial(results: list, next_idx: int,
                              _ckpt=ckpt, _key=partial_key) -> None:
                _ckpt.setdefault("partial", {})[_key] = results
                global _checkpoint_state
                _checkpoint_state = _ckpt
                save_checkpoint(config.checkpoint_file, _ckpt)

            result = evaluate_dataset(
                model=model,
                tokenizer=tokenizer,
                examples=datasets[ds_key],
                dataset_name=ds_name,
                format_fn=fmt_fn,
                extract_gt_fn=gt_fn,
                extract_answer_fn=ans_fn,
                check_match_fn=match_fn,
                max_new_tokens=max_tok,
                batch_size=bs,
                partial_results=partial_results,
                save_partial_fn=_save_partial,
                checkpoint_every=config.checkpoint_every,
            )

            # Mark complete; drop partial state
            ckpt["model_results"][model_key][ds_key] = {
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "total": result["total"],
            }
            ckpt["completed_conditions"].append(condition_key)
            ckpt.get("partial", {}).pop(partial_key, None)
            print(f"    {ds_name}: {result['accuracy']:.1f}% "
                  f"({result['correct']}/{result['total']})")

            _checkpoint_state = ckpt
            save_checkpoint(config.checkpoint_file, ckpt)

        # WikiText-2 perplexity
        if ppl_key not in ckpt["completed_conditions"]:
            print(f"\n  Computing WikiText-2 perplexity for {model_name}...")
            encodings = tokenizer(wikitext_text, return_tensors="pt")
            input_ids = encodings["input_ids"][0]
            seq_len = config.wikitext_seq_len
            num_seqs = config.wikitext_num_seqs
            if input_ids.size(0) < seq_len * num_seqs:
                num_seqs = input_ids.size(0) // seq_len
            sequences = input_ids[: seq_len * num_seqs].view(num_seqs, seq_len)
            ppl = compute_perplexity(model, sequences, config.ppl_batch_size)
            ckpt["model_results"][model_key]["wikitext2_ppl"] = ppl
            ckpt["completed_conditions"].append(ppl_key)
            print(f"    WikiText-2 PPL: {ppl:.2f}")
            _checkpoint_state = ckpt
            save_checkpoint(config.checkpoint_file, ckpt)

        peak_mem = get_peak_gpu_memory_mb()
        ckpt["model_results"][model_key]["peak_gpu_memory_mb"] = round(peak_mem, 1)

        # Free model before loading next
        del model, tokenizer
        aggressive_cleanup()
        print(f"  Peak GPU memory: {peak_mem:.0f} MB")

    # =========================================================================
    # Step 3: Build comparison table
    # =========================================================================
    print("\n[Step 3] Building comparison table...")
    header_ds = ["GSM8K", "ARC", "MATH", "WinoG", "PIQA", "PPL↓", "GPU MB"]
    print(f"\n{'Model':<22} " + " ".join(f"{h:>7}" for h in header_ds))
    print("─" * 80)

    def _fmt(val, is_ppl=False):
        if isinstance(val, (int, float)):
            return f"{val:.1f}" if not is_ppl else f"{val:.2f}"
        return "—"

    for model_key, model_name, _ in MODEL_CONFIGS:
        mdata = ckpt["model_results"].get(model_key, {})
        if "load_error" in mdata:
            print(f"{model_name:<22} " + "  ERROR" * len(header_ds))
            continue
        vals = [
            _fmt(mdata.get("gsm8k", {}).get("accuracy")),
            _fmt(mdata.get("arc_challenge", {}).get("accuracy")),
            _fmt(mdata.get("math", {}).get("accuracy")),
            _fmt(mdata.get("winogrande", {}).get("accuracy")),
            _fmt(mdata.get("piqa", {}).get("accuracy")),
            _fmt(mdata.get("wikitext2_ppl"), is_ppl=True),
            _fmt(mdata.get("gpu_memory_mb")),
        ]
        print(f"{model_name:<22} " + " ".join(f"{v:>7}" for v in vals))

    # =========================================================================
    # Step 4: Compute gaps relative to FP16
    # =========================================================================
    fp16_data = ckpt["model_results"].get("fp16", {})
    gap_table = {}

    for model_key, model_name, _ in MODEL_CONFIGS:
        if model_key == "fp16":
            continue
        mdata = ckpt["model_results"].get(model_key, {})
        if "load_error" in mdata:
            continue
        gaps = {}
        for ds_key, ds_name, *_ in DATASET_CONFIGS:
            fp16_acc = fp16_data.get(ds_key, {}).get("accuracy")
            quant_acc = mdata.get(ds_key, {}).get("accuracy")
            if fp16_acc is not None and quant_acc is not None:
                gaps[ds_key] = round(quant_acc - fp16_acc, 2)
        # PPL gap (higher PPL = worse, so positive gap = degradation)
        fp16_ppl = fp16_data.get("wikitext2_ppl")
        quant_ppl = mdata.get("wikitext2_ppl")
        if fp16_ppl is not None and quant_ppl is not None:
            gaps["wikitext2_ppl"] = round(quant_ppl - fp16_ppl, 2)
        gap_table[model_key] = {
            "model_name": model_name,
            "gaps": gaps,
        }

    gap_headers = ["GSM8K", "ARC", "MATH", "WinoG", "PIQA", "PPL↑"]
    gap_keys = ["gsm8k", "arc_challenge", "math", "winogrande", "piqa", "wikitext2_ppl"]
    print(f"\n{'Model':<22} " + " ".join(f"{h:>8}" for h in gap_headers))
    print("─" * 80)
    for model_key, info in gap_table.items():
        g = info["gaps"]
        parts = []
        for gk in gap_keys:
            v = g.get(gk)
            if v is not None:
                parts.append(f"{v:>+8.1f}")
            else:
                parts.append(f"{'—':>8}")
        print(f"{info['model_name']:<22} " + " ".join(parts))

    # =========================================================================
    # Step 5: Save final report
    # =========================================================================
    report = {
        "experiment": "Phase 20: Baseline Comparison — NF4 vs GPTQ vs AWQ",
        "purpose": (
            "Compare quantization damage across methods to show our NF4 probe "
            "is not an artifact of a uniquely bad quantization scheme"
        ),
        "base_model": config.base_model,
        "models_tested": {
            "fp16": config.base_model,
            "nf4": f"{config.base_model} (bitsandbytes NF4, double quant)",
            "gptq": config.gptq_model,
            "awq": config.awq_model,
        },
        "datasets": {
            "gsm8k": config.gsm8k_examples,
            "arc_challenge": config.arc_examples,
            "math": config.math_examples,
            "winogrande": config.winogrande_examples,
            "piqa": config.piqa_examples,
            "wikitext2_ppl": f"{config.wikitext_num_seqs} seqs x {config.wikitext_seq_len} tokens",
        },
        "results": ckpt["model_results"],
        "gap_vs_fp16": gap_table,
        "completed": True,
    }

    with open(config.output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Report saved: {config.output_file}")

    # Clean up checkpoint
    if os.path.exists(config.checkpoint_file):
        os.remove(config.checkpoint_file)
        print("  Checkpoint cleaned up (run complete).")

    print("=" * 70)


if __name__ == "__main__":
    run_comparison()
