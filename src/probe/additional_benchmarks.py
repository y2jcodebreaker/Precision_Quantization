"""
Additional Benchmarks: HellaSwag + MMLU Layer Sensitivity (Phase 19)
=====================================================================

HIGH PRIORITY fix for EMNLP reviewer concern W4.

Extends the layer sensitivity profiling to two non-math benchmarks:
  - HellaSwag: Commonsense reasoning (sentence completion)
  - MMLU (selected subsets): Factual knowledge recall

Tests whether the attention/MLP task-selectivity pattern holds beyond
math-adjacent tasks. If attention L13-14 is also critical for commonsense
and factual tasks, the "arithmetic-specialized" claim weakens. If MLP L6-7
is universally important across ALL task types, the "shared computation"
interpretation is strengthened.

Uses the same methodology as Phase 9: one-layer-at-a-time FP16 restoration
on the 12 most informative layers (6 critical + 6 controls) rather than
the full 64-layer sweep.

Dataset schemas (verified):
  HellaSwag (Rowan/hellaswag, split=validation):
    - ctx: str (full context, may have trailing space if ctx_b is empty)
    - endings: List[str] of length 4
    - label: str "0"/"1"/"2"/"3" (convert to int)
    - activity_label: str (topic category, used for context)
    - No empty labels in validation split (10042 examples)

  MMLU (cais/mmlu, <subject>, split=test):
    - question: str
    - choices: List[str] of length 4
    - answer: int 0-3
    - subject: str (already in item, no need to pass separately)
    Subject sizes: abstract_algebra=100, anatomy=135, college_chemistry=100,
      high_school_us_history=204, philosophy=311, professional_law=1534

Output: results/additional_benchmarks_report.json
Checkpoint: results/additional_benchmarks_checkpoint.json

Authentication:
---------------
Set HF_TOKEN environment variable for model access.
"""

import json
import os
import re
import gc
import signal
import tempfile
import hashlib
import time
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, asdict

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
    """Configuration for additional benchmark evaluation."""
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir: str = "results"
    output_file: str = os.path.join("results", "additional_benchmarks_report.json")
    checkpoint_file: str = os.path.join("results", "additional_benchmarks_checkpoint.json")

    num_examples: int = 500
    num_layers: int = 32

    # Layers to test: 6 critical (from Phase 9) + 6 controls
    test_attn_layers: Tuple[int, ...] = (13, 14, 0, 10, 20, 25)
    test_mlp_layers: Tuple[int, ...] = (1, 6, 7, 31, 3, 15)

    attn_projections: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    mlp_projections: Tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")

    # MMLU subjects — diverse sample across STEM, humanities, social sciences
    # Per-subject cap: num_examples // len(mmlu_subjects) = ~83 each
    mmlu_subjects: Tuple[str, ...] = (
        "abstract_algebra",        # 100 available — STEM math
        "anatomy",                 # 135 available — STEM biology
        "college_chemistry",       # 100 available — STEM science
        "high_school_us_history",  # 204 available — Humanities
        "philosophy",              # 311 available — Humanities/reasoning
        "professional_law",        # 1534 available — Professional
    )

    batch_size: int = 64          # For 4-bit model (~5.6 GB); safe on A100/H100
    fp16_batch_size: int = 16     # Conservative for FP16 model (~16 GB baseline pass)
    max_new_tokens: int = 10   # Only need "A", "B", "C", or "D" + brief explanation
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def fingerprint(self) -> str:
        """Hash of key config params to detect resume mismatches."""
        key = (
            self.model_name,
            self.num_examples,
            self.num_layers,
            self.test_attn_layers,
            self.test_mlp_layers,
            self.mmlu_subjects,
        )
        return hashlib.md5(str(key).encode()).hexdigest()[:8]


# =============================================================================
# Checkpoint I/O  (atomic writes + backup)
# =============================================================================

# Global reference so signal handler can flush before exit
_checkpoint: Dict = {}
_checkpoint_file: str = ""


def save_checkpoint(checkpoint: Dict, path: str) -> None:
    """Atomically save checkpoint: write to tmp file, then rename."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    # Keep one backup of the previous checkpoint
    backup_path = path + ".bak"
    if os.path.exists(path):
        try:
            os.replace(path, backup_path)
        except OSError:
            pass

    # Write new checkpoint atomically
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    os.replace(tmp_path, path)


def load_checkpoint(path: str, config: Config) -> Dict:
    """Load and validate checkpoint. Falls back to backup if corrupted."""
    for candidate in [path, path + ".bak"]:
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate) as f:
                ckpt = json.load(f)
            # Validate fingerprint
            stored_fp = ckpt.get("config_fingerprint", "")
            current_fp = config.fingerprint()
            if stored_fp and stored_fp != current_fp:
                print(
                    f"  WARNING: Checkpoint fingerprint mismatch "
                    f"(stored={stored_fp}, current={current_fp}). "
                    f"Config may have changed — ignoring checkpoint."
                )
                return {}
            print(f"  Loaded checkpoint from: {candidate}")
            return ckpt
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  WARNING: Could not load {candidate}: {e}")
    return {}


def _signal_handler(sig, frame) -> None:
    """Save checkpoint on SIGINT/SIGTERM before exiting."""
    if _checkpoint and _checkpoint_file:
        print("\n\n[SIGNAL] Saving checkpoint before exit...")
        save_checkpoint(_checkpoint, _checkpoint_file)
        print(f"[SIGNAL] Checkpoint saved to {_checkpoint_file}")
    raise SystemExit(0)


# =============================================================================
# Memory Management
# =============================================================================

def aggressive_cleanup() -> None:
    """Aggressively clear GPU and CPU memory."""
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_mb() -> float:
    """Get current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


# =============================================================================
# Dataset Loading
# =============================================================================

def load_hellaswag(num_examples: int) -> List[Dict]:
    """Load HellaSwag examples (4-way sentence completion).

    Schema (Rowan/hellaswag, split=validation):
      - ctx: full context string (ctx_a + ctx_b); strip trailing whitespace
      - endings: List[str] of 4 candidate completions
      - label: str "0"/"1"/"2"/"3"
      - activity_label: topic category string
    """
    print(f"  Loading HellaSwag ({num_examples} examples)...")
    dataset = load_dataset("Rowan/hellaswag", split="validation")

    examples = []
    for item in dataset:
        if len(examples) >= num_examples:
            break

        # label is a string "0"–"3"; skip any unexpected values
        raw_label = item["label"].strip()
        if raw_label not in ("0", "1", "2", "3"):
            continue

        ctx = item["ctx"].strip()
        if not ctx:
            continue  # skip rare empty context

        endings = item["endings"]
        if len(endings) != 4:
            continue

        label = int(raw_label)
        options_text = "\n".join(
            f"{chr(65 + j)}. {ending}" for j, ending in enumerate(endings)
        )

        examples.append({
            "index": len(examples),
            "context": ctx,
            "activity_label": item.get("activity_label", ""),
            "endings": endings,
            "options_text": options_text,
            "answer": chr(65 + label),  # "A"/"B"/"C"/"D"
            "dataset": "hellaswag",
        })

    print(f"  Loaded {len(examples)} HellaSwag examples")
    return examples


def load_mmlu(subjects: Tuple[str, ...], num_examples: int) -> List[Dict]:
    """Load MMLU examples from selected subjects, capped uniformly per subject.

    Schema (cais/mmlu, <subject>, split=test):
      - question: str
      - choices: List[str] of length 4
      - answer: int 0-3
      - subject: str (already in each item)
    """
    print(f"  Loading MMLU ({len(subjects)} subjects, target {num_examples} examples)...")

    # Cap per subject to distribute evenly; never exceed what's available
    per_subject_cap = num_examples // len(subjects)

    examples = []
    for subject in subjects:
        loaded = 0
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test")
        except Exception:
            try:
                dataset = load_dataset("hendrycks/test", subject, split="test")
            except Exception:
                print(f"    WARNING: Could not load MMLU subject '{subject}', skipping")
                continue

        for item in dataset:
            if loaded >= per_subject_cap:
                break

            choices = item["choices"]
            answer_idx = item["answer"]  # int 0-3

            if len(choices) != 4:
                continue
            if not (0 <= answer_idx <= 3):
                continue

            options_text = "\n".join(
                f"{chr(65 + j)}. {choice}" for j, choice in enumerate(choices)
            )

            # subject field is present in the item itself
            examples.append({
                "index": len(examples),
                "question": item["question"],
                "choices": choices,
                "options_text": options_text,
                "answer": chr(65 + answer_idx),  # "A"/"B"/"C"/"D"
                "subject": item["subject"],
                "dataset": "mmlu",
            })
            loaded += 1

        print(f"    {subject}: loaded {loaded}/{len(dataset)} examples "
              f"(cap={per_subject_cap})")

    print(f"  Total MMLU examples: {len(examples)}")
    return examples


# =============================================================================
# Prompt Formatting
# =============================================================================

def format_prompt(example: Dict) -> str:
    """Format example into Llama-3.1-8B-Instruct chat prompt."""
    ds = example["dataset"]

    if ds == "hellaswag":
        topic = example.get("activity_label", "")
        topic_prefix = f"[Topic: {topic}] " if topic else ""
        user_msg = (
            "Pick the most plausible continuation. "
            "Answer with just the letter (A, B, C, or D).\n\n"
            f"{topic_prefix}{example['context']}\n\n{example['options_text']}"
        )
    elif ds == "mmlu":
        subject_display = example["subject"].replace("_", " ").title()
        user_msg = (
            f"Answer this {subject_display} question. "
            "Answer with just the letter (A, B, C, or D).\n\n"
            f"{example['question']}\n\n{example['options_text']}"
        )
    else:
        raise ValueError(f"Unknown dataset: {ds}")

    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


# =============================================================================
# Answer Extraction
# =============================================================================

def extract_mc_answer(text: str) -> str:
    """Extract single letter A/B/C/D from model response."""
    text_clean = text.strip()
    # Prefer answer at start of response
    for pat in [
        r"^([A-D])\b",
        r"(?:answer|correct)\s*(?:is|:)\s*\(?([A-D])\)?",
        r"\b([A-D])\s*[\.\)]",
    ]:
        m = re.search(pat, text_clean, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    # Fallback: first A/B/C/D character in response
    for c in text_clean:
        if c.upper() in "ABCD":
            return c.upper()
    return ""


# =============================================================================
# Evaluation
# =============================================================================

def _run_one_batch(
    model: Any,
    tokenizer: Any,
    batch: List[Dict],
    config: Config,
) -> List[Dict]:
    """Run generation on a single batch; returns per-example results."""
    prompts = [format_prompt(ex) for ex in batch]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(config.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    results = []
    for j, ex in enumerate(batch):
        output_ids = outputs[j][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        pred = extract_mc_answer(response)
        correct = pred.upper() == ex["answer"].upper()
        results.append({
            "index": ex["index"],
            "dataset": ex["dataset"],
            "predicted": pred,
            "gold": ex["answer"],
            "correct": correct,
        })
    return results


def evaluate_batch(
    model: Any,
    tokenizer: Any,
    examples: List[Dict],
    config: Config,
    desc: str = "eval",
    batch_size: Optional[int] = None,
) -> List[Dict]:
    """Evaluate model on all examples with OOM-safe adaptive batch sizing.

    If a batch causes CUDA OOM, batch size is halved and the batch is retried.
    The reduced size persists for the remainder of this call (conservative).
    """
    effective_bs = batch_size if batch_size is not None else config.batch_size
    results = []

    pbar = tqdm(
        total=len(examples),
        desc=f"    {desc}",
        leave=False,
        unit="ex",
    )

    idx = 0
    while idx < len(examples):
        batch = examples[idx : idx + effective_bs]
        try:
            batch_results = _run_one_batch(model, tokenizer, batch, config)
            results.extend(batch_results)
            idx += len(batch)
            pbar.update(len(batch))

        except torch.cuda.OutOfMemoryError:
            aggressive_cleanup()
            new_bs = effective_bs // 2
            if new_bs < 1:
                raise RuntimeError(
                    f"CUDA OOM even at batch_size=1 for {desc}. "
                    f"Check GPU memory."
                )
            print(
                f"\n    [OOM] batch_size {effective_bs} → {new_bs} for {desc}"
            )
            effective_bs = new_bs
            # Do NOT advance idx — retry the same batch at smaller size

    pbar.close()
    return results


def accuracy_from_results(results: List[Dict]) -> float:
    """Compute accuracy from evaluate_batch output."""
    if not results:
        return 0.0
    return sum(r["correct"] for r in results) / len(results)


# =============================================================================
# Layer Patching (FP16 cache → restore single layer)
# =============================================================================

def cache_fp16_weights(model_name: str, config: Config) -> Dict:
    """Load FP16 model on CPU and cache all projection weights."""
    print("  Caching FP16 weights to CPU...")
    fp16_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cpu",
    )

    cache: Dict = {}
    for layer_idx in range(config.num_layers):
        layer = fp16_model.model.layers[layer_idx]
        cache[layer_idx] = {"attention": {}, "mlp": {}}

        for proj in config.attn_projections:
            module = getattr(layer.self_attn, proj)
            cache[layer_idx]["attention"][proj] = {
                "weight": module.weight.data.clone(),
                "bias": module.bias.data.clone() if module.bias is not None else None,
                "in_features": module.in_features,
                "out_features": module.out_features,
            }
        for proj in config.mlp_projections:
            module = getattr(layer.mlp, proj)
            cache[layer_idx]["mlp"][proj] = {
                "weight": module.weight.data.clone(),
                "bias": module.bias.data.clone() if module.bias is not None else None,
                "in_features": module.in_features,
                "out_features": module.out_features,
            }

    del fp16_model
    aggressive_cleanup()
    print("  FP16 weights cached")
    return cache


def compute_fp16_baselines(
    model_name: str,
    tokenizer: Any,
    datasets_map: Dict[str, List[Dict]],
    config: Config,
) -> Dict:
    """Load FP16 model briefly to compute FP16 baselines, then delete."""
    print("  Loading FP16 model for baselines...")
    fp16_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
    )
    fp16_model.eval()

    baselines_fp16: Dict = {}
    for ds_name, ds_examples in datasets_map.items():
        print(f"  Evaluating FP16 on {ds_name} (batch_size={config.fp16_batch_size})...")
        results = evaluate_batch(fp16_model, tokenizer, ds_examples, config,
                                 desc=f"FP16 {ds_name}",
                                 batch_size=config.fp16_batch_size)
        acc = accuracy_from_results(results)
        baselines_fp16[f"{ds_name}_fp16"] = {
            "accuracy": round(acc, 4),
            "n": len(results),
        }
        print(f"    {ds_name} FP16 accuracy: {acc:.1%}")

    del fp16_model
    aggressive_cleanup()
    return baselines_fp16


def patch_layer(
    model: Any,
    layer_idx: int,
    component: str,
    fp16_cache: Dict,
    config: Config,
) -> Dict[str, nn.Module]:
    """Replace quantized projections with FP16 weights. Returns saved originals."""
    layer = model.model.layers[layer_idx]
    parent = layer.self_attn if component == "attention" else layer.mlp
    projs = config.attn_projections if component == "attention" else config.mlp_projections

    orig_modules: Dict[str, nn.Module] = {}
    for proj_name in projs:
        orig_module = getattr(parent, proj_name)
        orig_modules[proj_name] = orig_module
        cached = fp16_cache[layer_idx][component][proj_name]

        # Detect target device from the original module
        if hasattr(orig_module, "weight"):
            target_device = orig_module.weight.device
        else:
            target_device = torch.device(config.device)

        new_module = nn.Linear(
            cached["in_features"],
            cached["out_features"],
            bias=cached["bias"] is not None,
        ).to(target_device).half()
        new_module.weight.data = cached["weight"].to(target_device)
        if cached["bias"] is not None:
            new_module.bias.data = cached["bias"].to(target_device)

        setattr(parent, proj_name, new_module)

    return orig_modules


def restore_layer(
    model: Any,
    layer_idx: int,
    component: str,
    orig_modules: Dict[str, nn.Module],
    config: Config,
) -> None:
    """Restore original quantized modules after a sweep condition."""
    layer = model.model.layers[layer_idx]
    parent = layer.self_attn if component == "attention" else layer.mlp

    for proj_name, orig_module in orig_modules.items():
        current = getattr(parent, proj_name)
        setattr(parent, proj_name, orig_module)
        del current

    aggressive_cleanup()


# =============================================================================
# Main Pipeline
# =============================================================================

def run_experiment() -> None:
    """Run HellaSwag + MMLU layer sensitivity sweep with checkpointing."""
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)

    torch.manual_seed(config.seed)

    # Register signal handlers for graceful checkpoint saves
    global _checkpoint, _checkpoint_file
    _checkpoint_file = config.checkpoint_file
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Load checkpoint
    checkpoint = load_checkpoint(config.checkpoint_file, config)
    checkpoint["config_fingerprint"] = config.fingerprint()
    _checkpoint = checkpoint

    print("=" * 70)
    print("PHASE 19: Additional Benchmarks (HellaSwag + MMLU)")
    print(f"Config fingerprint: {config.fingerprint()}")
    print("=" * 70)

    # =========================================================================
    # Step 1: Load datasets
    # =========================================================================
    print("\n[Step 1] Loading datasets...")
    hellaswag_data = load_hellaswag(config.num_examples)
    mmlu_data = load_mmlu(config.mmlu_subjects, config.num_examples)

    datasets_map = {
        "hellaswag": hellaswag_data,
        "mmlu": mmlu_data,
    }

    # =========================================================================
    # Step 2: Cache FP16 weights (CPU)
    # =========================================================================
    print("\n[Step 2] Caching FP16 weights...")
    fp16_cache = cache_fp16_weights(config.model_name, config)

    # =========================================================================
    # Step 3: Tokenizer
    # =========================================================================
    print("\n[Step 3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # =========================================================================
    # Step 4: FP16 baselines (load FP16 model → evaluate → delete)
    # =========================================================================
    baselines: Dict = checkpoint.get("baselines", {})
    fp16_baselines_done = all(
        f"{ds}_fp16" in baselines for ds in datasets_map
    )

    if not fp16_baselines_done:
        print("\n[Step 4a] Computing FP16 baselines...")
        fp16_results = compute_fp16_baselines(
            config.model_name, tokenizer, datasets_map, config
        )
        baselines.update(fp16_results)
        checkpoint["baselines"] = baselines
        save_checkpoint(checkpoint, config.checkpoint_file)
        print(f"  FP16 baselines saved.")
    else:
        print("\n[Step 4a] FP16 baselines: loaded from checkpoint.")

    # =========================================================================
    # Step 5: Load 4-bit quantized model
    # =========================================================================
    print("\n[Step 5] Loading 4-bit quantized model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    print(f"  GPU memory: {get_gpu_memory_mb():.0f} MB")

    # =========================================================================
    # Step 6: 4-bit baselines
    # =========================================================================
    fourbit_baselines_done = all(
        f"{ds}_4bit" in baselines for ds in datasets_map
    )

    if not fourbit_baselines_done:
        print("\n[Step 6] Computing 4-bit baselines...")
        for ds_name, ds_examples in datasets_map.items():
            if f"{ds_name}_4bit" in baselines:
                print(f"  {ds_name} 4-bit: loaded from checkpoint")
                continue
            print(f"  Evaluating 4-bit on {ds_name} (batch_size={config.batch_size})...")
            results = evaluate_batch(model, tokenizer, ds_examples, config,
                                     desc=f"4bit {ds_name}",
                                     batch_size=config.batch_size)
            acc = accuracy_from_results(results)
            baselines[f"{ds_name}_4bit"] = {
                "accuracy": round(acc, 4),
                "n": len(results),
            }
            print(f"    {ds_name} 4-bit accuracy: {acc:.1%}")
            checkpoint["baselines"] = baselines
            save_checkpoint(checkpoint, config.checkpoint_file)
    else:
        print("\n[Step 6] 4-bit baselines: loaded from checkpoint.")

    # Print baseline summary
    print("\n  Baseline summary:")
    for ds_name in datasets_map:
        fp16_acc = baselines.get(f"{ds_name}_fp16", {}).get("accuracy", "N/A")
        bit4_acc = baselines.get(f"{ds_name}_4bit", {}).get("accuracy", "N/A")
        print(f"    {ds_name}: FP16={fp16_acc:.1%}  4-bit={bit4_acc:.1%}  "
              f"gap={(fp16_acc - bit4_acc)*100:+.1f}pp"
              if isinstance(fp16_acc, float) and isinstance(bit4_acc, float)
              else f"    {ds_name}: fp16={fp16_acc} 4bit={bit4_acc}")

    # =========================================================================
    # Step 7: Layer sensitivity sweep
    # =========================================================================
    print("\n[Step 7] Running layer sensitivity sweep...")

    test_conditions = (
        [(li, "attention") for li in config.test_attn_layers]
        + [(li, "mlp") for li in config.test_mlp_layers]
    )
    total_conditions = len(test_conditions)

    sweep_results: Dict = checkpoint.get("sweep", {})

    done_count = sum(1 for (li, comp) in test_conditions
                     if f"L{li}_{comp}" in sweep_results)
    print(f"  Progress: {done_count}/{total_conditions} conditions already done")

    t_start = time.time()

    for test_idx, (layer_idx, component) in enumerate(test_conditions):
        key = f"L{layer_idx}_{component}"

        if key in sweep_results:
            print(f"  [{test_idx+1}/{total_conditions}] {key} — SKIP (cached)")
            continue

        print(f"\n  [{test_idx+1}/{total_conditions}] Patching {key}  "
              f"(GPU: {get_gpu_memory_mb():.0f} MB)")

        orig_modules = patch_layer(model, layer_idx, component, fp16_cache, config)

        layer_result: Dict = {
            "layer": layer_idx,
            "component": component,
            "datasets": {},
        }

        # Evaluate each dataset independently so partial results survive a crash
        for ds_name, ds_examples in datasets_map.items():
            # Fine-grained checkpoint: save partial layer results per dataset
            partial_key = f"{key}__partial"
            partial = checkpoint.get(partial_key, {})

            if ds_name in partial:
                print(f"    {ds_name}: loaded partial result from checkpoint")
                layer_result["datasets"][ds_name] = partial[ds_name]
                continue

            eval_results = evaluate_batch(
                model, tokenizer, ds_examples, config,
                desc=f"{key}/{ds_name}",
                batch_size=config.batch_size,
            )
            acc = accuracy_from_results(eval_results)

            baseline_acc = baselines[f"{ds_name}_4bit"]["accuracy"]
            fp16_acc = baselines.get(f"{ds_name}_fp16", {}).get("accuracy", None)

            improvement_vs_4bit = acc - baseline_acc
            improvement_vs_fp16 = (acc - fp16_acc) if fp16_acc is not None else None

            ds_result = {
                "accuracy": round(acc, 4),
                "improvement_pp_vs_4bit": round(improvement_vs_4bit * 100, 2),
                "improvement_pp_vs_fp16": (
                    round(improvement_vs_fp16 * 100, 2)
                    if improvement_vs_fp16 is not None else None
                ),
            }
            layer_result["datasets"][ds_name] = ds_result

            # Save partial result immediately
            partial[ds_name] = ds_result
            checkpoint[partial_key] = partial
            save_checkpoint(checkpoint, config.checkpoint_file)

            print(f"    {ds_name}: {acc:.1%} "
                  f"({improvement_vs_4bit*100:+.2f}pp vs 4-bit)")

        # Restore quantized modules
        restore_layer(model, layer_idx, component, orig_modules, config)

        # Promote partial results to sweep_results, clean up partial key
        sweep_results[key] = layer_result
        checkpoint["sweep"] = sweep_results
        if f"{key}__partial" in checkpoint:
            del checkpoint[f"{key}__partial"]
        save_checkpoint(checkpoint, config.checkpoint_file)

        # ETA estimate
        elapsed = time.time() - t_start
        done = test_idx + 1 - done_count
        remaining = total_conditions - (test_idx + 1)
        if done > 0 and remaining > 0:
            eta_s = elapsed / done * remaining
            print(f"    ETA: {eta_s/60:.1f} min for {remaining} remaining conditions")

    # =========================================================================
    # Step 8: Analysis
    # =========================================================================
    print("\n[Step 8] Analyzing results...")

    analysis: Dict = {
        "hellaswag_top_layers": [],
        "mmlu_top_layers": [],
        "critical_layer_comparison": {},
    }

    for ds_name in ["hellaswag", "mmlu"]:
        ranked = []
        for key, data in sweep_results.items():
            ds_data = data.get("datasets", {}).get(ds_name)
            if ds_data:
                ranked.append({
                    "layer": key,
                    "improvement_pp": ds_data["improvement_pp_vs_4bit"],
                })
        ranked.sort(key=lambda x: -x["improvement_pp"])
        analysis[f"{ds_name}_top_layers"] = ranked[:6]

        print(f"\n  Top layers for {ds_name}:")
        for i, entry in enumerate(ranked[:6]):
            print(f"    {i+1}. {entry['layer']}: {entry['improvement_pp']:+.2f}pp")

    # Compare critical vs control layers across all datasets
    groups = [
        ([f"L{li}_attention" for li in (13, 14)], "Attention L13-14 (arithmetic-specialized)"),
        ([f"L{li}_mlp" for li in (6, 7)],         "MLP L6-7 (shared computation)"),
        ([f"L{li}_attention" for li in (0, 10)],   "Attention L0,L10 (controls)"),
        ([f"L{li}_mlp" for li in (3, 15)],         "MLP L3,L15 (controls)"),
    ]

    for keys, label in groups:
        avg: Dict = {}
        for ds_name in ["hellaswag", "mmlu"]:
            vals = [
                sweep_results[k]["datasets"][ds_name]["improvement_pp_vs_4bit"]
                for k in keys
                if k in sweep_results and ds_name in sweep_results[k].get("datasets", {})
            ]
            avg[ds_name] = round(sum(vals) / len(vals), 2) if vals else None

        analysis["critical_layer_comparison"][label] = avg
        print(f"\n  {label}:")
        for ds, imp in avg.items():
            print(f"    {ds}: {imp:+.2f}pp" if imp is not None else f"    {ds}: N/A")

    # =========================================================================
    # Step 9: Save final report
    # =========================================================================
    report = {
        "experiment": "Phase 19: Additional Benchmarks (HellaSwag + MMLU)",
        "purpose": "Test attention/MLP task-selectivity beyond math-adjacent benchmarks",
        "model": config.model_name,
        "num_examples_per_dataset": config.num_examples,
        "mmlu_subjects": list(config.mmlu_subjects),
        "baselines": baselines,
        "sweep_results": sweep_results,
        "analysis": analysis,
        "completed": True,
    }

    with open(config.output_file, "w") as f:
        json.dump(report, f, indent=2)

    # Mark checkpoint as complete
    checkpoint["completed"] = True
    save_checkpoint(checkpoint, config.checkpoint_file)

    print(f"\n✓ Report saved: {config.output_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
