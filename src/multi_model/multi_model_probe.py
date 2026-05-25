"""
Multi-Model Restoration Probe (Phase 23)
==========================================

Runs the NF4 restoration probe on three additional models to extend
cross-model generalization beyond Llama-3.1-8B + Mistral-7B.

Models:
  1. Qwen/Qwen2.5-Math-7B-Instruct     — math-specialized; tests MATH null
  2. deepseek-ai/DeepSeek-R1-Distill-Llama-8B — reasoning-distilled; same arch
  3. google/gemma-2-9b-it               — alternating local/global attention

Scientific questions:
  - Qwen-Math:  When MATH is no longer capability-limited (~85% FP16),
                 does it become precision-sensitive? Do L13-14 attention
                 extend their task-selectivity to MATH?
  - DeepSeek:   Same Llama architecture but reasoning-distilled. Do the
                 same layers emerge, or does training redistribute load?
  - Gemma-2:    Different attention design (local+global alternating).
                 Does task-selectivity still appear at relative depth ~40%?

Pipeline per model:
  1. FP16 baseline → identify per-task accuracy
  2. NF4 baseline → identify flipped examples
  3. Cache FP16 weights
  4. 64-layer (or N-layer) restoration sweep
  5. Rankings + cross-model comparison

Output: results/multi_model_probe.json

Dependencies:
  pip install transformers datasets torch bitsandbytes

Authentication:
  Set HF_TOKEN environment variable for model access.
"""

import argparse
import json
import os
import re
import gc
import random
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from datasets import load_dataset


# =============================================================================
# Model Specifications
# =============================================================================

@dataclass
class ModelSpec:
    """Specification for a single model to probe."""
    name: str               # display name
    hf_id: str              # HuggingFace model ID
    num_layers: int          # number of transformer layers
    attn_projections: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    mlp_projections: Tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")
    # Layer accessor path from the HF model object
    layer_path: str = "model.layers"
    # Max new tokens override (R1 models emit long CoT, need more)
    max_new_tokens: int = 256
    # Extra kwargs for AutoModelForCausalLM.from_pretrained
    extra_load_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Optional system prompt injected into the chat template.
    # When set: GSM8K user prompt omits the "####" format instruction
    # (model is expected to use \boxed{} driven by the system prompt).
    system_prompt: Optional[str] = None
    # Notes
    notes: str = ""


MODEL_SPECS = {
    "qwen_math": ModelSpec(
        name="Qwen2.5-Math-7B",
        hf_id="Qwen/Qwen2.5-Math-7B-Instruct",
        num_layers=28,
        # Required system prompt — without it the model ignores its math-solving mode.
        system_prompt=(
            "Please reason step by step, and put your final answer within \\boxed{}."
        ),
        # Qwen2.5-Math uses verbose step-by-step CoT: typical GSM8K solution
        # is 400-800 tokens; 256 truncates before \boxed{} is ever written.
        max_new_tokens=1024,
        notes="Math-specialized; FP16 MATH ~85%; tests capability-limitation hypothesis",
    ),
    "deepseek_r1": ModelSpec(
        name="DeepSeek-R1-Distill-Llama-8B",
        hf_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        num_layers=32,
        max_new_tokens=1024,  # R1 emits <think>...</think> CoT before answer
        notes="Llama architecture, reasoning-distilled; tests training vs architecture",
    ),
    "gemma2": ModelSpec(
        name="Gemma-2-9B-it",
        hf_id="google/gemma-2-9b-it",
        num_layers=42,
        mlp_projections=("gate_proj", "up_proj", "down_proj"),
        extra_load_kwargs={"attn_implementation": "eager"},  # Required for Gemma-2
        notes="Alternating local(4096)/global attention; different arch family",
    ),
}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration for multi-model restoration probe."""
    output_dir: str = "results"
    output_file: str = os.path.join("results", "multi_model_probe.json")
    checkpoint_file: str = os.path.join("results", "multi_model_probe_ckpt.json")

    num_examples: int = 500
    batch_size: int = 4   # Match Phase 9 layer_sensitivity_profiler.py
    max_new_tokens: int = 256  # Default; overridden by ModelSpec.max_new_tokens
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Which models to probe (keys from MODEL_SPECS)
    models_to_run: Tuple[str, ...] = ("qwen_math", "deepseek_r1", "gemma2")


# =============================================================================
# Memory Management
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def aggressive_cleanup():
    """Clear GPU and CPU memory."""
    gc.collect()
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_mb() -> float:
    """Current GPU memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


# =============================================================================
# Layer Accessor
# =============================================================================

def get_layers(model, layer_path: str = "model.layers") -> list:
    """Navigate to transformer layers using dot-separated path."""
    obj = model
    for attr in layer_path.split("."):
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
        else:
            raise AttributeError(
                f"Model has no attribute '{attr}' in path '{layer_path}'. "
                f"Available: {[a for a in dir(obj) if not a.startswith('_')]}"
            )
    return obj


# =============================================================================
# Dataset Loading
# =============================================================================

def load_gsm8k(n: int) -> List[Dict]:
    """Load GSM8K test examples."""
    print(f"  Loading GSM8K ({n} examples)...")
    dataset = load_dataset("gsm8k", "main", split="test")
    return [
        {"index": i, "question": item["question"],
         "answer": item["answer"], "dataset": "gsm8k"}
        for i, item in enumerate(dataset) if i < n
    ]


def load_arc_challenge(n: int) -> List[Dict]:
    """Load ARC-Challenge test examples."""
    print(f"  Loading ARC-Challenge ({n} examples)...")
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    examples = []
    for i, item in enumerate(dataset):
        if i >= n:
            break
        labels = item["choices"]["label"]
        texts = item["choices"]["text"]
        examples.append({
            "index": i,
            "question": item["question"],
            "choices_labels": labels,
            "choices_texts": texts,
            "options_text": "\n".join(f"{l}. {t}" for l, t in zip(labels, texts)),
            "answer": item["answerKey"],
            "dataset": "arc",
        })
    return examples


def load_math(n: int) -> List[Dict]:
    """Load MATH competition examples."""
    print(f"  Loading MATH ({n} examples)...")
    for source, split in [
        ("hendrycks/competition_math", "test"),
        ("DigitalLearningGmbH/MATH-lighteval", "test"),
        ("hendrycks/competition_math", "train"),
        ("DigitalLearningGmbH/MATH-lighteval", "train"),
    ]:
        try:
            dataset = load_dataset(source, split=split)
            print(f"    Loaded from {source} ({split})")
            return [
                {"index": i, "problem": item["problem"],
                 "solution": item["solution"], "dataset": "math"}
                for i, item in enumerate(dataset) if i < n
            ]
        except Exception:
            continue
    raise RuntimeError("Could not load MATH dataset from any source")


# =============================================================================
# Prompt Formatting (uses tokenizer.apply_chat_template)
# =============================================================================

def build_user_content(example: Dict, has_system_prompt: bool = False) -> str:
    """Build the user message content from an example.

    Args:
        example: Dataset example dict.
        has_system_prompt: When True the system prompt already instructs the model
            on output format (e.g. \\boxed{}), so omit format instructions here.
    """
    ds = example["dataset"]
    if ds == "gsm8k":
        if has_system_prompt:
            # System prompt handles format (e.g. Qwen2.5-Math uses \boxed{})
            return f"Solve this math problem.\n\n{example['question']}"
        return (
            "Solve this math problem. End with your numerical answer "
            "after '####'.\n\n"
            f"{example['question']}"
        )
    elif ds == "arc":
        return (
            "Answer this science question by selecting the correct option.\n\n"
            f"Question: {example['question']}\n\n"
            f"Options:\n{example['options_text']}\n\n"
            f"Respond with just the letter ({', '.join(example['choices_labels'])})."
        )
    elif ds == "math":
        return (
            "Solve this problem. Put your final answer in \\boxed{}.\n\n"
            f"{example['problem']}"
        )
    return ""


def format_prompt_with_template(
    example: Dict, tokenizer, spec: Optional["ModelSpec"] = None
) -> str:
    """Format prompt using the tokenizer's chat template.

    Works across Llama, Qwen, DeepSeek, and Gemma tokenizers.
    Injects a system message when spec.system_prompt is set.
    Falls back to user-only messages if chat template not available.
    """
    has_sys = spec is not None and spec.system_prompt is not None
    content = build_user_content(example, has_system_prompt=has_sys)
    messages = []
    if has_sys:
        messages.append({"role": "system", "content": spec.system_prompt})
    messages.append({"role": "user", "content": content})

    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        # Some models don't support apply_chat_template
        # Fall back to raw prompt (prepend system text if present)
        prefix = (spec.system_prompt + "\n\n") if has_sys else ""
        return prefix + content + "\n\nAnswer:"


# =============================================================================
# Answer Extraction & Matching
# =============================================================================

def extract_answer(response: str, example: Dict) -> Optional[str]:
    """Extract answer from model response.

    Handles DeepSeek-R1 <think>...</think> blocks by stripping them
    and extracting only from the final answer portion.
    """
    # Strip DeepSeek-R1 thinking blocks if present
    text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    text = text.replace("**", "").replace("*", "")
    ds = example["dataset"]

    if ds == "gsm8k":
        # \boxed{} style first (Qwen2.5-Math and similar models)
        m = re.search(r"\\boxed\{([^}]+)\}", text)
        if m:
            val = m.group(1).strip().replace(",", "")
            try:
                float(val)
                return val
            except ValueError:
                pass
        # Standard #### format (Llama, Mistral, etc.)
        m = re.search(r"####\s*\$?([+-]?\d[\d,]*\.?\d*)", text)
        if m:
            return m.group(1).replace(",", "")
        for p in [
            r"(?:the\s+)?answer\s+is[:\s]*\$?([+-]?\d[\d,]*\.?\d*)",
            r"=\s*\$?([+-]?\d[\d,]*\.?\d*)\s*$",
        ]:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return m.group(1).replace(",", "")
        numbers = re.findall(r"(?<!\d)([+-]?\d[\d,]*\.?\d*)(?!\d)", text)
        return numbers[-1].replace(",", "") if numbers else None

    elif ds == "arc":
        valid = example.get("choices_labels", ["A", "B", "C", "D"])
        pat = ''.join(valid)
        if text and text[0].upper() in valid:
            return text[0].upper()
        for p in [
            rf"(?:the\s+)?answer\s+is[:\s]*([{pat}])\b",
            rf"\b([{pat}])\s+is\s+(?:the\s+)?(?:correct|right)",
            rf"^([{pat}])\s*[\.:\)]",
        ]:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return m.group(1).upper()
        for label in valid:
            if re.search(rf"\b{label}\b", text, re.IGNORECASE):
                return label.upper()

    elif ds == "math":
        m = re.search(r"\\boxed\{([^}]+)\}", text)
        if m:
            return m.group(1).strip()
        m = re.search(
            r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*(.+?)(?:\.|$)",
            text, re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()

    return None


def extract_ground_truth(example: Dict) -> str:
    """Extract ground truth from example."""
    ds = example["dataset"]
    if ds == "gsm8k":
        m = re.search(r"####\s*([+-]?\d[\d,]*\.?\d*)", example["answer"])
        return m.group(1).replace(",", "") if m else ""
    elif ds == "arc":
        return example["answer"]
    elif ds == "math":
        m = re.search(r"\\boxed\{([^}]+)\}", example["solution"])
        return m.group(1).strip() if m else ""
    return ""


def check_match(pred: Optional[str], gt: str, dataset: str) -> bool:
    """Check answer match."""
    if pred is None or not gt:
        return False
    if dataset == "gsm8k":
        try:
            return abs(float(pred) - float(gt)) < 1e-6
        except ValueError:
            return False
    elif dataset == "arc":
        return pred.upper() == gt.upper()
    elif dataset == "math":
        return pred.strip().lower() == gt.strip().lower()
    return pred == gt


# =============================================================================
# Model Loading
# =============================================================================

def _setup_tokenizer(hf_id: str):
    """Load tokenizer with safe pad token assignment."""
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            # Avoid using eos_token as pad (causes early stopping).
            # Add a dedicated pad token instead.
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            print(f"    Added <pad> token (model had no unk_token)")
    tokenizer.padding_side = "left"
    return tokenizer


def load_fp16_model(hf_id: str, spec: Optional['ModelSpec'] = None):
    """Load model in FP16."""
    print(f"\n  Loading FP16: {hf_id}")
    tokenizer = _setup_tokenizer(hf_id)
    extra = spec.extra_load_kwargs if spec else {}
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
        **extra,
    )
    model.eval()
    return model, tokenizer


def load_nf4_model(hf_id: str, spec: Optional['ModelSpec'] = None):
    """Load model in NF4 4-bit."""
    print(f"\n  Loading NF4: {hf_id}")
    tokenizer = _setup_tokenizer(hf_id)
    extra = spec.extra_load_kwargs if spec else {}
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
        **extra,
    )
    model.eval()
    return model, tokenizer


def unload_model(model, tokenizer=None):
    """Free model memory."""
    del model
    if tokenizer:
        del tokenizer
    aggressive_cleanup()


# =============================================================================
# FP16 Weight Cache
# =============================================================================

def cache_fp16_weights(
    hf_id: str, num_layers: int, spec: ModelSpec,
) -> Dict:
    """Cache all FP16 attention + MLP weights to CPU."""
    print(f"\n  Caching FP16 weights for {spec.name} ({num_layers} layers)...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, dtype=torch.bfloat16,
        device_map="cpu", low_cpu_mem_usage=True,
        trust_remote_code=True,
        **spec.extra_load_kwargs,
    )

    layers = get_layers(model, spec.layer_path)
    cache = {"attention": {}, "mlp": {}}

    for idx in tqdm(range(num_layers), desc="    Caching"):
        layer = layers[idx]

        cache["attention"][idx] = {}
        for proj in spec.attn_projections:
            p = getattr(layer.self_attn, proj)
            cache["attention"][idx][proj] = {
                "weight": p.weight.data.clone().cpu(),
                "bias": p.bias.data.clone().cpu() if p.bias is not None else None,
                "in_features": p.in_features,
                "out_features": p.out_features,
            }

        cache["mlp"][idx] = {}
        for proj in spec.mlp_projections:
            p = getattr(layer.mlp, proj)
            cache["mlp"][idx][proj] = {
                "weight": p.weight.data.clone().cpu(),
                "bias": p.bias.data.clone().cpu() if p.bias is not None else None,
                "in_features": p.in_features,
                "out_features": p.out_features,
            }

    del model
    aggressive_cleanup()
    print(f"    Cached {num_layers} layers")
    return cache


# =============================================================================
# Transplant Operations
# =============================================================================

def transplant_component(
    model, layer_idx: int, component: str,
    fp16_cache: Dict, spec: ModelSpec, device: str,
) -> Dict[str, Any]:
    """Replace quantized sub-layer with FP16 weights. Returns originals."""
    layers = get_layers(model, spec.layer_path)
    layer = layers[layer_idx]

    if component == "attention":
        parent = layer.self_attn
        projections = spec.attn_projections
    else:
        parent = layer.mlp
        projections = spec.mlp_projections

    originals = {}
    for proj_name in projections:
        originals[proj_name] = getattr(parent, proj_name)

        info = fp16_cache[component][layer_idx][proj_name]
        new_linear = nn.Linear(
            info["in_features"], info["out_features"],
            bias=info["bias"] is not None,
            dtype=torch.bfloat16, device=device,
        )
        new_linear.weight.data = info["weight"].to(device=device, dtype=torch.bfloat16)
        if info["bias"] is not None:
            new_linear.bias.data = info["bias"].to(device=device, dtype=torch.bfloat16)

        setattr(parent, proj_name, new_linear)

    return originals


def restore_component(
    model, layer_idx: int, component: str,
    originals: Dict[str, Any], spec: ModelSpec,
):
    """Restore original quantized modules."""
    layers = get_layers(model, spec.layer_path)
    layer = layers[layer_idx]
    parent = layer.self_attn if component == "attention" else layer.mlp

    for proj_name, orig in originals.items():
        current = getattr(parent, proj_name)
        del current
        setattr(parent, proj_name, orig)

    aggressive_cleanup()


# =============================================================================
# Inference
# =============================================================================

def run_inference(
    model, tokenizer, examples: List[Dict],
    batch_size: int, max_new_tokens: int,
    spec: Optional["ModelSpec"] = None,
) -> List[Dict]:
    """Batched inference with chat template formatting."""
    if not examples:
        return []

    results = []
    for batch_start in range(0, len(examples), batch_size):
        batch = examples[batch_start:batch_start + batch_size]
        prompts = [
            format_prompt_with_template(ex, tokenizer, spec)
            for ex in batch
        ]

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=1024,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for i, (example, output) in enumerate(zip(batch, outputs)):
            input_len = inputs["input_ids"][i].shape[0]
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            gt = extract_ground_truth(example)
            pred = extract_answer(response, example)
            results.append({
                "index": example["index"],
                "dataset": example["dataset"],
                "correct": check_match(pred, gt, example["dataset"]),
            })

    return results


# =============================================================================
# Checkpoint
# =============================================================================

def save_checkpoint(data: Dict, path: str):
    """Atomic checkpoint save."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> Optional[Dict]:
    """Load checkpoint if exists."""
    abs_path = os.path.abspath(path)
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            models_done = list(data.get("models", {}).keys())
            print(f"  Checkpoint found: {abs_path}")
            print(f"    Models in checkpoint: {models_done}")
            for mk, mv in data.get("models", {}).items():
                status = mv.get("status", "unknown")
                has_baselines = "fp16_baselines" in mv and "flipped_indices" in mv
                has_sweep = "layer_sensitivity" in mv
                print(f"    [{mk}] status={status}  baselines={has_baselines}  sweep={has_sweep}")
            return data
        except (json.JSONDecodeError, KeyError):
            print(f"  Corrupt checkpoint at {abs_path}, starting fresh")
    else:
        print(f"  No checkpoint found at: {abs_path}")
    return None


# =============================================================================
# Single-Model Probe
# =============================================================================

def run_probe_for_model(
    model_key: str,
    spec: ModelSpec,
    all_examples: Dict[str, List[Dict]],
    config: Config,
    checkpoint: Optional[Dict] = None,
) -> Dict:
    """Run the full restoration probe for one model.

    Returns dict with baselines, flipped, sensitivity, rankings.
    """
    print(f"\n{'='*70}")
    print(f"PROBE: {spec.name}")
    print(f"  HF ID:   {spec.hf_id}")
    print(f"  Layers:  {spec.num_layers}")
    print(f"  Notes:   {spec.notes}")
    print(f"{'='*70}")

    # ---- Checkpoint recovery ----
    _ckpt_existing = {}
    if checkpoint and model_key in checkpoint.get("models", {}):
        _ckpt_existing = checkpoint["models"][model_key]
        if _ckpt_existing.get("status") == "complete":
            print(f"  Already complete in checkpoint, skipping")
            return _ckpt_existing

    # ---- FP16 + NF4 baselines (skip if already in checkpoint) ----
    if "fp16_baselines" in _ckpt_existing and "flipped_indices" in _ckpt_existing:
        print(f"\n  Baselines restored from checkpoint (skipping re-evaluation)...")
        fp16_baselines = _ckpt_existing["fp16_baselines"]
        nf4_baselines = _ckpt_existing["nf4_baselines"]
        flipped_indices = _ckpt_existing["flipped_indices"]
        for ds_name in ["gsm8k", "arc", "math"]:
            drop = fp16_baselines.get(ds_name, 0) - nf4_baselines.get(ds_name, 0)
            n = len(flipped_indices.get(ds_name, []))
            print(f"    {ds_name}: FP16={fp16_baselines.get(ds_name, 0):.2%}  NF4={nf4_baselines.get(ds_name, 0):.2%}  Drop={drop:.2%}  Flipped={n}")
    else:
        print(f"\n  FP16 Baseline...")
        model, tokenizer = load_fp16_model(spec.hf_id, spec)
        fp16_results = {}
        fp16_baselines = {}

        for ds_name, examples in all_examples.items():
            print(f"    {ds_name}...")
            res = run_inference(
                model, tokenizer, examples,
                config.batch_size, spec.max_new_tokens, spec,
            )
            fp16_results[ds_name] = res
            acc = sum(r["correct"] for r in res) / len(res) if res else 0
            fp16_baselines[ds_name] = acc
            print(f"      FP16: {acc:.2%}")

        unload_model(model, tokenizer)

        # ---- NF4 baseline & flipped identification ----
        print(f"\n  NF4 Baseline...")
        model, tokenizer = load_nf4_model(spec.hf_id, spec)
        nf4_results = {}
        nf4_baselines = {}
        flipped_indices = {}

        for ds_name, examples in all_examples.items():
            print(f"    {ds_name}...")
            res = run_inference(
                model, tokenizer, examples,
                config.batch_size, spec.max_new_tokens, spec,
            )
            nf4_results[ds_name] = res
            acc = sum(r["correct"] for r in res) / len(res) if res else 0
            nf4_baselines[ds_name] = acc

            fp16_by_idx = {r["index"]: r for r in fp16_results[ds_name]}
            nf4_by_idx = {r["index"]: r for r in res}
            flipped = [
                idx for idx in fp16_by_idx
                if fp16_by_idx[idx]["correct"] and not nf4_by_idx[idx]["correct"]
            ]
            flipped_indices[ds_name] = flipped

            drop = fp16_baselines[ds_name] - acc
            print(f"      NF4: {acc:.2%}  Drop: {drop:.2%}  Flipped: {len(flipped)}")

        unload_model(model, tokenizer)

        # Save baseline checkpoint immediately — if sweep crashes before layer 4,
        # resume will restore baselines from here instead of recomputing them.
        _baseline_ckpt = {
            "fp16_baselines": fp16_baselines,
            "nf4_baselines": nf4_baselines,
            "flipped_indices": flipped_indices,
            "flipped_counts": {ds: len(v) for ds, v in flipped_indices.items()},
            "status": "baselines_complete",
        }
        if checkpoint is None:
            checkpoint = {"models": {}}
        checkpoint["models"][model_key] = _baseline_ckpt
        save_checkpoint(checkpoint, config.checkpoint_file)
        print(f"  Baseline checkpoint saved")

    # ---- Flipped examples subset ----
    flipped_examples = {}
    for ds_name in all_examples:
        idx_set = set(flipped_indices[ds_name])
        flipped_examples[ds_name] = [
            ex for ex in all_examples[ds_name] if ex["index"] in idx_set
        ]

    # ---- Cache FP16 weights ----
    print(f"\n  Caching FP16 weights...")
    fp16_cache = cache_fp16_weights(spec.hf_id, spec.num_layers, spec)

    # ---- Load NF4 for sweep ----
    model, tokenizer = load_nf4_model(spec.hf_id, spec)

    # ---- Restoration sweep ----
    print(f"\n  Starting {spec.num_layers * 2}-test sweep...")
    sensitivity = {
        "attention": {ds: {} for ds in ["gsm8k", "arc", "math"]},
        "mlp": {ds: {} for ds in ["gsm8k", "arc", "math"]},
    }

    # Resume support
    start_layer = 0
    if checkpoint and model_key in checkpoint.get("models", {}):
        existing = checkpoint["models"][model_key]
        if "layer_sensitivity" in existing:
            sensitivity = existing["layer_sensitivity"]
            for l in range(spec.num_layers - 1, -1, -1):
                if str(l) in sensitivity["mlp"]["gsm8k"]:
                    start_layer = l + 1
                    break
            if start_layer > 0:
                print(f"    Resuming from layer {start_layer}")

    total_tests = spec.num_layers * 2
    test_count = start_layer * 2

    for layer_idx in range(start_layer, spec.num_layers):
        print(f"\n  Layer {layer_idx}/{spec.num_layers - 1}")

        for component in ["attention", "mlp"]:
            test_count += 1
            print(f"    [{test_count}/{total_tests}] {component.upper()}...")

            originals = transplant_component(
                model, layer_idx, component, fp16_cache, spec, config.device,
            )

            for ds_name in ["gsm8k", "arc", "math"]:
                if not flipped_examples[ds_name]:
                    sensitivity[component][ds_name][str(layer_idx)] = 0.0
                    continue
                res = run_inference(
                    model, tokenizer, flipped_examples[ds_name],
                    config.batch_size, spec.max_new_tokens, spec,
                )
                rate = sum(r["correct"] for r in res) / len(res)
                sensitivity[component][ds_name][str(layer_idx)] = rate

            restore_component(model, layer_idx, component, originals, spec)

            rates = [
                sensitivity[component][ds][str(layer_idx)]
                for ds in ["gsm8k", "arc", "math"]
            ]
            print(f"      GSM8K={rates[0]:.1%} ARC={rates[1]:.1%} MATH={rates[2]:.1%}")

        print(f"      GPU: {get_gpu_mb():.0f} MB")

        # Periodic checkpoint
        if (layer_idx + 1) % 4 == 0:
            partial = {
                "fp16_baselines": fp16_baselines,
                "nf4_baselines": nf4_baselines,
                "flipped_indices": flipped_indices,
                "flipped_counts": {ds: len(v) for ds, v in flipped_indices.items()},
                "layer_sensitivity": sensitivity,
                "status": "in_progress",
                "last_completed_layer": layer_idx,
            }
            if checkpoint is None:
                checkpoint = {"models": {}}
            checkpoint["models"][model_key] = partial
            save_checkpoint(checkpoint, config.checkpoint_file)
            print(f"      Checkpoint saved (layer {layer_idx})")

    unload_model(model, tokenizer)

    # ---- Rankings ----
    rankings = {"attention": {}, "mlp": {}}
    for comp in ["attention", "mlp"]:
        for ds in ["gsm8k", "arc", "math"]:
            sorted_layers = sorted(
                sensitivity[comp][ds].items(),
                key=lambda x: x[1], reverse=True,
            )
            rankings[comp][ds] = [
                {"layer": int(l), "recovery_rate": r}
                for l, r in sorted_layers[:10]
            ]

    result = {
        "model_name": spec.name,
        "hf_id": spec.hf_id,
        "num_layers": spec.num_layers,
        "notes": spec.notes,
        "fp16_baselines": fp16_baselines,
        "nf4_baselines": nf4_baselines,
        "flipped_counts": {ds: len(v) for ds, v in flipped_indices.items()},
        "layer_sensitivity": sensitivity,
        "rankings": rankings,
        "status": "complete",
    }

    # Print summary
    print(f"\n  === {spec.name} Top Layers ===")
    for ds in ["gsm8k", "arc", "math"]:
        attn_top = rankings["attention"][ds][0] if rankings["attention"][ds] else {"layer": -1, "recovery_rate": 0}
        mlp_top = rankings["mlp"][ds][0] if rankings["mlp"][ds] else {"layer": -1, "recovery_rate": 0}
        print(f"    {ds}: Attn L{attn_top['layer']} ({attn_top['recovery_rate']:.1%}) | "
              f"MLP L{mlp_top['layer']} ({mlp_top['recovery_rate']:.1%})")

    return result


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 23: Multi-Model Restoration Probe"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=list(MODEL_SPECS.keys()),
        metavar="MODEL_KEY",
        help=(
            "Which models to run (default: all three). "
            f"Choices: {list(MODEL_SPECS.keys())}. "
            "Example: --models qwen_math"
        ),
    )
    parser.add_argument(
        "--rerun", nargs="+", default=None,
        choices=list(MODEL_SPECS.keys()),
        metavar="MODEL_KEY",
        help=(
            "Force re-run these models even if marked complete in checkpoint "
            "(clears their cached state). Example: --rerun qwen_math"
        ),
    )
    args = parser.parse_args()

    config = Config()

    # Override models_to_run if --models specified
    if args.models:
        config.models_to_run = tuple(args.models)

    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 70)
    print("Phase 23: Multi-Model Restoration Probe")
    print("=" * 70)
    print(f"  Models: {config.models_to_run}")
    print(f"  Examples: {config.num_examples} per dataset")
    print(f"  Device: {config.device}")

    set_seed(config.seed)
    checkpoint = load_checkpoint(config.checkpoint_file)

    # --rerun: remove specified models from checkpoint so they re-run fresh
    if args.rerun:
        if checkpoint is None:
            checkpoint = {"models": {}}
        for mk in args.rerun:
            if mk in checkpoint.get("models", {}):
                del checkpoint["models"][mk]
                print(f"  Cleared checkpoint for '{mk}' (--rerun flag)")
            else:
                print(f"  No checkpoint entry for '{mk}' (will run fresh)")

    # ---- Load datasets (shared) ----
    print(f"\n{'='*70}")
    print("Loading Datasets")
    print(f"{'='*70}")

    all_examples = {
        "gsm8k": load_gsm8k(config.num_examples),
        "arc": load_arc_challenge(config.num_examples),
        "math": load_math(config.num_examples),
    }

    # ---- Run probes ----
    all_results = {
        "experiment": "Phase 23: Multi-Model Restoration Probe",
        "num_examples": config.num_examples,
        "models": {},
    }

    # When running a subset (--models), preserve other models' results from the
    # existing output file so the final JSON stays complete.
    if args.models and os.path.exists(config.output_file):
        try:
            with open(config.output_file) as f:
                existing = json.load(f)
            all_results["models"] = existing.get("models", {})
            kept = [k for k in all_results["models"] if k not in config.models_to_run]
            if kept:
                print(f"  Preserved existing results for: {kept}")
        except (json.JSONDecodeError, KeyError):
            print(f"  Could not load existing results — starting fresh")

    for model_key in config.models_to_run:
        if model_key not in MODEL_SPECS:
            print(f"\n  WARNING: Unknown model '{model_key}', skipping")
            continue

        spec = MODEL_SPECS[model_key]
        result = run_probe_for_model(
            model_key, spec, all_examples, config, checkpoint,
        )
        all_results["models"][model_key] = result

        # Save after each model
        save_checkpoint(all_results, config.checkpoint_file)

    # ====================================================================
    # Cross-model comparison
    # ====================================================================
    print(f"\n{'='*70}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'='*70}")

    # Print baselines side by side
    print("\n  FP16 / NF4 Baselines:")
    print(f"  {'Model':<30s} {'GSM8K':>10s} {'ARC':>10s} {'MATH':>10s}")
    for model_key in config.models_to_run:
        if model_key not in all_results["models"]:
            continue
        r = all_results["models"][model_key]
        fp = r["fp16_baselines"]
        nf = r["nf4_baselines"]
        print(f"  {r['model_name']:<30s} "
              f"{fp['gsm8k']:.1%}/{nf['gsm8k']:.1%}  "
              f"{fp['arc']:.1%}/{nf['arc']:.1%}  "
              f"{fp['math']:.1%}/{nf['math']:.1%}")

    # Print top attention layers across models
    print("\n  Top Attention Layer per Dataset:")
    for ds in ["gsm8k", "arc", "math"]:
        print(f"\n    {ds.upper()}:")
        for model_key in config.models_to_run:
            if model_key not in all_results["models"]:
                continue
            r = all_results["models"][model_key]
            rank = r["rankings"]["attention"][ds]
            top = rank[0] if rank else {"layer": -1, "recovery_rate": 0}
            n_layers = r["num_layers"]
            rel_depth = top["layer"] / n_layers if n_layers > 0 else 0
            print(f"      {r['model_name']:<30s} L{top['layer']:2d} "
                  f"({top['recovery_rate']:.1%}) "
                  f"[depth={rel_depth:.0%}]")

    # Print top MLP layers across models
    print("\n  Top MLP Layer per Dataset:")
    for ds in ["gsm8k", "arc", "math"]:
        print(f"\n    {ds.upper()}:")
        for model_key in config.models_to_run:
            if model_key not in all_results["models"]:
                continue
            r = all_results["models"][model_key]
            rank = r["rankings"]["mlp"][ds]
            top = rank[0] if rank else {"layer": -1, "recovery_rate": 0}
            n_layers = r["num_layers"]
            rel_depth = top["layer"] / n_layers if n_layers > 0 else 0
            print(f"      {r['model_name']:<30s} L{top['layer']:2d} "
                  f"({top['recovery_rate']:.1%}) "
                  f"[depth={rel_depth:.0%}]")

    # ====================================================================
    # Save final
    # ====================================================================
    with open(config.output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {config.output_file}")

    if os.path.exists(config.checkpoint_file):
        os.remove(config.checkpoint_file)
        print("Checkpoint removed (run complete)")

    print(f"\n{'='*70}")
    print("Phase 23 Complete!")
    print(f"{'='*70}")

    return all_results


if __name__ == "__main__":
    results = main()
