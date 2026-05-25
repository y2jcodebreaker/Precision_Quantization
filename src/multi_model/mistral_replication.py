"""
Mistral-7B Replication: Layer Sensitivity + Cross-Task Interference + Surgical Capstone
========================================================================================

Replicates the three core experiments from the Llama-3.1-8B study on
Mistral-7B-Instruct-v0.3 to establish cross-model generalization.

Pipeline:
  Phase A — FP16 and 4-bit baselines on all 3 datasets (500 examples each)
  Phase B — 64-test layer sensitivity sweep (every layer × attention/MLP)
  Phase C — Auto-select top layers; run cross-task interference matrix
  Phase D — Surgical capstone: patch selected layers, evaluate all datasets
  Phase E — 3-seed statistical validation (McNemar's test)

Key difference from Llama scripts:
  - Uses tokenizer.apply_chat_template() for Mistral's [INST] format
  - GQA: k_proj/v_proj are smaller (8 KV heads × 128 dim = 1024, vs 4096 for Llama)
    → transplant code handles this automatically via in_features/out_features
  - Layer selection is data-driven from Phase B, not hardcoded

Output: results/mistral_replication_report.json

Authentication:
---------------
Set HF_TOKEN environment variable for model access.
"""

import json
import os
import re
import gc
import time
import random
import math
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

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
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    output_dir: str = "results"
    output_file: str = os.path.join("results", "mistral_replication_report.json")
    checkpoint_file: str = os.path.join("results", "mistral_replication_checkpoint.json")

    num_examples: int = 500        # per dataset for baselines
    sweep_seeds: Tuple[int, ...] = (42, 52, 62)
    num_layers: int = 32

    # Layer selection: how many top attention and MLP layers to use
    top_attn_layers: int = 2
    top_mlp_layers: int = 4

    attn_projections: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    mlp_projections: Tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")

    batch_size: int = 64
    max_new_tokens: int = 512
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Memory Management
# =============================================================================

def aggressive_cleanup():
    gc.collect(); gc.collect(); gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# Dataset Loading
# =============================================================================

def load_gsm8k(n: int, seed: int = 42) -> List[Dict]:
    print(f"  Loading GSM8K (n={n}, seed={seed})...")
    dataset = list(load_dataset("gsm8k", "main", split="test"))
    rng = random.Random(seed)
    rng.shuffle(dataset)
    return [{"index": i, "question": item["question"], "answer": item["answer"],
             "dataset": "gsm8k"} for i, item in enumerate(dataset[:n])]


def load_arc_challenge(n: int, seed: int = 42) -> List[Dict]:
    print(f"  Loading ARC-Challenge (n={n}, seed={seed})...")
    dataset = list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"))
    rng = random.Random(seed)
    rng.shuffle(dataset)
    examples = []
    for i, item in enumerate(dataset[:n]):
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


def load_math(n: int, seed: int = 42) -> List[Dict]:
    print(f"  Loading MATH (n={n}, seed={seed})...")
    dataset = None
    for source, split in [
        ("hendrycks/competition_math", "test"),
        ("DigitalLearningGmbH/MATH-lighteval", "test"),
        ("hendrycks/competition_math", "train"),
        ("DigitalLearningGmbH/MATH-lighteval", "train"),
    ]:
        try:
            dataset = list(load_dataset(source, split=split))
            print(f"    Loaded from {source} ({split})")
            break
        except Exception:
            continue
    if dataset is None:
        raise RuntimeError("Could not load MATH dataset")
    rng = random.Random(seed)
    rng.shuffle(dataset)
    return [{"index": i, "problem": item["problem"], "solution": item["solution"],
             "dataset": "math"} for i, item in enumerate(dataset[:n])]


# =============================================================================
# Prompt Formatting (Mistral chat template)
# =============================================================================

def format_prompt(example: Dict, tokenizer) -> str:
    ds = example["dataset"]
    if ds == "gsm8k":
        content = (f"Solve this math problem. End with your numerical answer after '####'.\n\n"
                   f"{example['question']}")
    elif ds == "arc":
        content = (f"Answer this science question by selecting the correct option.\n\n"
                   f"Question: {example['question']}\n\nOptions:\n{example['options_text']}\n\n"
                   f"Respond with just the letter ({', '.join(example['choices_labels'])}).")
    else:  # math
        content = (f"Solve this problem. Put your final answer in \\boxed{{}}.\n\n"
                   f"{example['problem']}")
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# =============================================================================
# Answer Extraction
# =============================================================================

def extract_answer(response: str, example: Dict) -> Optional[str]:
    text = response.replace("**", "").replace("*", "")
    ds = example["dataset"]

    if ds == "gsm8k":
        m = re.search(r"####\s*\$?([+-]?\d[\d,]*\.?\d*)", text)
        if m: return m.group(1).replace(",", "")
        for p in [
            r"answer\s+is[:\s]*\$?([+-]?\d[\d,]*\.?\d*)",
            r"final\s+answer[:\s]*\$?([+-]?\d[\d,]*\.?\d*)",
            r"=\s*\$?([+-]?\d[\d,]*\.?\d*)\s*(?:\.|$)",
        ]:
            m = re.search(p, text, re.IGNORECASE)
            if m: return m.group(1).replace(",", "")
        nums = re.findall(r"(?<!\d)([+-]?\d[\d,]*\.?\d*)(?!\d)", text)
        return nums[-1].replace(",", "") if nums else None

    elif ds == "arc":
        valid = example.get("choices_labels", ["A", "B", "C", "D"])
        vp = "".join(valid)
        if text and text[0].upper() in valid:
            return text[0].upper()
        for p in [rf"answer\s+is[:\s]*([{vp}])\b", rf"\b([{vp}])\s+is\s+(?:correct|right)",
                  rf"^([{vp}])\s*[\.:\)]"]:
            m = re.search(p, text, re.IGNORECASE)
            if m: return m.group(1).upper()
        for l in valid:
            if re.search(rf"\b{l}\b", text, re.IGNORECASE):
                return l.upper()

    else:  # math
        m = re.search(r"\\boxed\{([^}]+)\}", text)
        if m: return m.group(1).strip()
        m = re.search(r"answer\s+is[:\s]*(.+?)(?:\.|$)", text, re.IGNORECASE)
        if m: return m.group(1).strip()

    return None


def extract_ground_truth(example: Dict) -> str:
    ds = example["dataset"]
    if ds == "gsm8k":
        m = re.search(r"####\s*([+-]?\d[\d,]*\.?\d*)", example["answer"])
        return m.group(1).replace(",", "") if m else ""
    elif ds == "arc":
        return example["answer"]
    else:
        m = re.search(r"\\boxed\{([^}]+)\}", example["solution"])
        return m.group(1).strip() if m else ""


def check_match(pred: Optional[str], truth: str, ds: str) -> bool:
    if pred is None or not truth:
        return False
    if ds == "gsm8k":
        try: return abs(float(pred) - float(truth)) < 1e-6
        except ValueError: return False
    elif ds == "arc":
        return pred.upper() == truth.upper()
    return pred.strip().lower() == truth.strip().lower()


# =============================================================================
# Inference
# =============================================================================

def run_inference(model, tokenizer, examples: List[Dict], config: Config,
                  label: str, max_new_tokens: Optional[int] = None) -> Dict:
    if not examples:
        return {"accuracy": 0.0, "correct": 0, "total": 0,
                "per_example_correct": [], "wall_time_s": 0.0}

    mnt = max_new_tokens or config.max_new_tokens
    ds = examples[0]["dataset"]
    if ds == "arc":
        mnt = 32

    correct, total = 0, len(examples)
    per_example_correct, total_tokens = [], 0
    num_batches = (total + config.batch_size - 1) // config.batch_size
    t0 = time.time()

    for b in tqdm(range(num_batches), desc=f"    {label}", leave=False):
        batch = examples[b * config.batch_size: (b + 1) * config.batch_size]
        prompts = [format_prompt(ex, tokenizer) for ex in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=mnt, temperature=0.0,
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for i, (ex, out) in enumerate(zip(batch, outputs)):
            in_len = inputs["input_ids"][i].shape[0]
            gen = out[in_len:]
            total_tokens += len(gen)
            resp = tokenizer.decode(gen, skip_special_tokens=True)
            pred = extract_answer(resp, ex)
            truth = extract_ground_truth(ex)
            ok = check_match(pred, truth, ex["dataset"])
            if ok: correct += 1
            per_example_correct.append(1 if ok else 0)

        if (b + 1) % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    wall = time.time() - t0
    acc = correct / total
    thr = total_tokens / wall if wall > 0 else 0
    print(f"      {label}: {acc:.2%} ({correct}/{total}) | {thr:.0f} tok/s")
    return {"accuracy": round(acc, 4), "correct": correct, "total": total,
            "per_example_correct": per_example_correct,
            "wall_time_s": round(wall, 2), "throughput_tok_per_s": round(thr, 1)}


# =============================================================================
# Model Loading
# =============================================================================

def _configure_tokenizer(tok):
    """Set pad token for Mistral (prefers unk over eos to avoid early-stop edge cases)."""
    if tok.pad_token is None:
        if tok.unk_token is not None:
            tok.pad_token = tok.unk_token
        else:
            tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_fp16_model(model_name: str):
    print(f"\n  Loading FP16 model: {model_name}")
    tok = _configure_tokenizer(AutoTokenizer.from_pretrained(model_name))
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    return model, tok


def load_4bit_model(model_name: str):
    print(f"\n  Loading 4-bit model: {model_name}")
    tok = _configure_tokenizer(AutoTokenizer.from_pretrained(model_name))
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto")
    model.eval()
    return model, tok


def unload_model(model, tok=None):
    del model
    if tok: del tok
    aggressive_cleanup()


def cache_all_fp16_weights(model_name: str, config: Config) -> Dict:
    """Cache FP16 weights for ALL 32 layers on CPU (for the sweep)."""
    print("\n  Caching all FP16 weights (CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
    cache = {}
    for layer_idx in range(config.num_layers):
        layer = model.model.layers[layer_idx]
        cache[layer_idx] = {"attention": {}, "mlp": {}}
        for proj in config.attn_projections:
            p = getattr(layer.self_attn, proj)
            cache[layer_idx]["attention"][proj] = {
                "weight": p.weight.data.clone().cpu(),
                "bias": p.bias.data.clone().cpu() if p.bias is not None else None,
                "in_features": p.in_features, "out_features": p.out_features,
            }
        for proj in config.mlp_projections:
            p = getattr(layer.mlp, proj)
            cache[layer_idx]["mlp"][proj] = {
                "weight": p.weight.data.clone().cpu(),
                "bias": p.bias.data.clone().cpu() if p.bias is not None else None,
                "in_features": p.in_features, "out_features": p.out_features,
            }
    del model
    aggressive_cleanup()
    print(f"  Cached {config.num_layers} layers.")
    return cache


def cache_selected_fp16_weights(model_name: str, attn_layers: List[int],
                                mlp_layers: List[int], config: Config) -> Dict:
    """Cache FP16 weights for selected layers only (for capstone)."""
    print(f"\n  Caching FP16 weights for selected layers...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
    cache = {"attention": {}, "mlp": {}}
    for li in attn_layers:
        layer = model.model.layers[li]
        cache["attention"][li] = {}
        for proj in config.attn_projections:
            p = getattr(layer.self_attn, proj)
            cache["attention"][li][proj] = {
                "weight": p.weight.data.clone().cpu(),
                "bias": p.bias.data.clone().cpu() if p.bias is not None else None,
                "in_features": p.in_features, "out_features": p.out_features,
            }
        print(f"    Cached attention L{li}")
    for li in mlp_layers:
        layer = model.model.layers[li]
        cache["mlp"][li] = {}
        for proj in config.mlp_projections:
            p = getattr(layer.mlp, proj)
            cache["mlp"][li][proj] = {
                "weight": p.weight.data.clone().cpu(),
                "bias": p.bias.data.clone().cpu() if p.bias is not None else None,
                "in_features": p.in_features, "out_features": p.out_features,
            }
        print(f"    Cached MLP L{li}")
    del model
    aggressive_cleanup()
    return cache


def patch_layer(model, layer_idx: int, component: str, weight_cache: Dict, config: Config):
    """Transplant FP16 weights into one layer (attention or mlp)."""
    layer = model.model.layers[layer_idx]
    projs = config.attn_projections if component == "attention" else config.mlp_projections
    parent = layer.self_attn if component == "attention" else layer.mlp
    w_dict = weight_cache[layer_idx][component]
    # Detect device from first existing proj to handle multi-GPU device_map correctly
    first_proj = getattr(parent, projs[0])
    target_device = (first_proj.weight.device if hasattr(first_proj, "weight")
                     else torch.device(config.device))
    for proj in projs:
        w = w_dict[proj]
        new_lin = nn.Linear(w["in_features"], w["out_features"],
                            bias=w["bias"] is not None, dtype=torch.float16)
        new_lin.weight.data = w["weight"].to(target_device)
        if w["bias"] is not None:
            new_lin.bias.data = w["bias"].to(target_device)
        new_lin = new_lin.to(target_device)
        setattr(parent, proj, new_lin)


def apply_surgical_patch(model, selected_cache: Dict, config: Config):
    """Apply FP16 patches to selected layers."""
    print("\n  Applying surgical patches...")
    for li, layer_cache in selected_cache["attention"].items():
        layer = model.model.layers[li]
        target_device = getattr(layer.self_attn, config.attn_projections[0]).weight.device
        for proj in config.attn_projections:
            w = layer_cache[proj]
            new_lin = nn.Linear(w["in_features"], w["out_features"],
                                bias=w["bias"] is not None, dtype=torch.float16)
            new_lin.weight.data = w["weight"].to(target_device)
            if w["bias"] is not None:
                new_lin.bias.data = w["bias"].to(target_device)
            setattr(layer.self_attn, proj, new_lin.to(target_device))
        print(f"    Patched attention L{li}")
    for li, layer_cache in selected_cache["mlp"].items():
        layer = model.model.layers[li]
        target_device = getattr(layer.mlp, config.mlp_projections[0]).weight.device
        for proj in config.mlp_projections:
            w = layer_cache[proj]
            new_lin = nn.Linear(w["in_features"], w["out_features"],
                                bias=w["bias"] is not None, dtype=torch.float16)
            new_lin.weight.data = w["weight"].to(target_device)
            if w["bias"] is not None:
                new_lin.bias.data = w["bias"].to(target_device)
            setattr(layer.mlp, proj, new_lin.to(target_device))
        print(f"    Patched MLP L{li}")


# =============================================================================
# Statistical Tests
# =============================================================================

def mcnemar_test(a: List[int], b: List[int]) -> Dict:
    assert len(a) == len(b)
    bc = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    cc = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    if bc + cc == 0:
        return {"b_count": 0, "c_count": 0, "chi2": 0.0, "p_value": 1.0,
                "significant_at_0.05": False, "note": "no discordant pairs"}
    chi2 = (abs(bc - cc) - 1) ** 2 / (bc + cc)
    if HAS_SCIPY:
        p = float(1 - scipy_stats.chi2.cdf(chi2, df=1))
    else:
        x = chi2 / 2.0
        p = 1.0 - math.erf(math.sqrt(x)) if x > 0 else 1.0
    return {"b_count": bc, "c_count": cc, "chi2": round(chi2, 4),
            "p_value": round(p, 6), "significant_at_0.05": p < 0.05}


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(data: Dict, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    print(f"  [ckpt] {path}")


def load_checkpoint(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except (json.JSONDecodeError, ValueError):
        print(f"  WARNING: corrupt checkpoint, ignoring: {path}")
        return None


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 70)
    print("Mistral-7B Replication Experiment")
    print("=" * 70)
    print(f"Model:  {config.model_name}")
    print(f"Device: {config.device}")

    ckpt = load_checkpoint(config.checkpoint_file)
    state = ckpt if ckpt else {}

    datasets = ["gsm8k", "arc", "math"]

    # =========================================================================
    # Phase A: FP16 and 4-bit baselines
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE A: Baselines (FP16 and 4-bit)")
    print(f"{'='*70}")

    if "baselines" not in state:
        state["baselines"] = {}

    examples = {
        "gsm8k": load_gsm8k(config.num_examples, config.seed),
        "arc":   load_arc_challenge(config.num_examples, config.seed),
        "math":  load_math(config.num_examples, config.seed),
    }

    for condition in ["fp16", "int4"]:
        if condition in state["baselines"] and all(
            ds in state["baselines"][condition] for ds in datasets
        ):
            print(f"  {condition} baselines loaded from checkpoint.")
            continue

        if condition == "fp16":
            model, tok = load_fp16_model(config.model_name)
        else:
            model, tok = load_4bit_model(config.model_name)

        state["baselines"].setdefault(condition, {})
        for ds in datasets:
            if ds in state["baselines"][condition]:
                print(f"  {condition}/{ds} loaded from checkpoint.")
                continue
            result = run_inference(model, tok, examples[ds], config, f"{condition}/{ds}")
            state["baselines"][condition][ds] = result
            save_checkpoint(state, config.checkpoint_file)

        unload_model(model, tok)
        save_checkpoint(state, config.checkpoint_file)

    # Identify flipped examples per dataset
    flipped = {}
    for ds in datasets:
        fp16_vec = state["baselines"]["fp16"][ds]["per_example_correct"]
        int4_vec = state["baselines"]["int4"][ds]["per_example_correct"]
        flipped_idx = [i for i, (a, b) in enumerate(zip(fp16_vec, int4_vec)) if a == 1 and b == 0]
        flipped[ds] = [examples[ds][i] for i in flipped_idx]
        print(f"  {ds}: {len(flipped[ds])} flipped examples (FP16 correct, 4-bit wrong)")

    state["flipped_counts"] = {ds: len(flipped[ds]) for ds in datasets}
    save_checkpoint(state, config.checkpoint_file)

    # =========================================================================
    # Phase B: Layer sensitivity sweep
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE B: Layer Sensitivity Sweep (64 tests)")
    print(f"{'='*70}")

    if "sweep" not in state:
        state["sweep"] = {}

    sweep_done = all(
        f"attn_{li}" in state["sweep"] and f"mlp_{li}" in state["sweep"]
        for li in range(config.num_layers)
    )

    if not sweep_done:
        full_cache = cache_all_fp16_weights(config.model_name, config)
        model, tok = load_4bit_model(config.model_name)

        for li in range(config.num_layers):
            for component in ["attention", "mlp"]:
                key = f"{component}_{li}"
                if key in state["sweep"]:
                    continue

                # Save original quantized modules before patching so we can restore them
                layer = model.model.layers[li]
                parent = layer.self_attn if component == "attention" else layer.mlp
                projs = (config.attn_projections if component == "attention"
                         else config.mlp_projections)
                orig_modules = {proj: getattr(parent, proj) for proj in projs}

                patch_layer(model, li, component, full_cache, config)

                layer_result = {}
                for ds in datasets:
                    if not flipped[ds]:
                        layer_result[ds] = {"recovery_rate": 0.0, "recovered": 0,
                                            "total_flipped": 0}
                        continue
                    res = run_inference(model, tok, flipped[ds], config,
                                        f"L{li} {component} / {ds}", max_new_tokens=256)
                    recovered = res["correct"]
                    layer_result[ds] = {
                        "recovery_rate": round(recovered / len(flipped[ds]), 4),
                        "recovered": recovered,
                        "total_flipped": len(flipped[ds]),
                    }

                # Restore original quantized modules — keep model clean for next test
                for proj in projs:
                    setattr(parent, proj, orig_modules[proj])
                del orig_modules

                state["sweep"][key] = layer_result
                save_checkpoint(state, config.checkpoint_file)

        unload_model(model, tok)
        del full_cache
        aggressive_cleanup()
    else:
        print("  Sweep loaded from checkpoint.")

    # Rank layers per dataset
    state["layer_rankings"] = {}
    for ds in datasets:
        attn_scores = {li: state["sweep"][f"attention_{li}"][ds]["recovery_rate"]
                       for li in range(config.num_layers)}
        mlp_scores  = {li: state["sweep"][f"mlp_{li}"][ds]["recovery_rate"]
                       for li in range(config.num_layers)}
        state["layer_rankings"][ds] = {
            "attention_top5": sorted(attn_scores, key=attn_scores.get, reverse=True)[:5],
            "mlp_top5":       sorted(mlp_scores,  key=mlp_scores.get,  reverse=True)[:5],
            "attention_scores": attn_scores,
            "mlp_scores":       mlp_scores,
        }
        top_a = state["layer_rankings"][ds]["attention_top5"]
        top_m = state["layer_rankings"][ds]["mlp_top5"]
        print(f"  {ds} top attention: {top_a[:3]} | top MLP: {top_m[:3]}")

    save_checkpoint(state, config.checkpoint_file)

    # Auto-select layers for surgical patch
    # Top-2 attention by GSM8K recovery + top-4 MLP by ARC recovery (mirrors Llama strategy)
    gsm8k_attn_scores = state["layer_rankings"]["gsm8k"]["attention_scores"]
    arc_mlp_scores    = state["layer_rankings"]["arc"]["mlp_scores"]

    selected_attn = sorted(gsm8k_attn_scores, key=gsm8k_attn_scores.get, reverse=True)[:config.top_attn_layers]
    selected_mlp  = sorted(arc_mlp_scores,    key=arc_mlp_scores.get,    reverse=True)[:config.top_mlp_layers]

    state["selected_layers"] = {"attention": selected_attn, "mlp": selected_mlp}
    print(f"\n  Selected attention layers (GSM8K-optimal): {selected_attn}")
    print(f"  Selected MLP layers (ARC-optimal):         {selected_mlp}")
    save_checkpoint(state, config.checkpoint_file)

    # =========================================================================
    # Phase C: Cross-task interference matrix
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE C: Cross-Task Interference Matrix")
    print(f"{'='*70}")

    selected_cache = cache_selected_fp16_weights(
        config.model_name, selected_attn, selected_mlp, config)

    # GSM8K-optimal: just top-2 attention layers
    gsm8k_opt_cache = {
        "attention": {li: selected_cache["attention"][li] for li in selected_attn},
        "mlp": {},
    }
    # ARC-optimal: just top-4 MLP layers
    arc_opt_cache = {
        "attention": {},
        "mlp": {li: selected_cache["mlp"][li] for li in selected_mlp},
    }
    # All combined
    combined_cache = selected_cache

    interference_conditions = [
        ("int4_baseline",    None),
        ("gsm8k_opt_attn",   gsm8k_opt_cache),
        ("arc_opt_mlp",      arc_opt_cache),
        ("all_combined",     combined_cache),
    ]

    if "interference" not in state:
        state["interference"] = {}

    for cond_name, patch_cache in interference_conditions:
        if cond_name in state["interference"]:
            print(f"  {cond_name} loaded from checkpoint.")
            continue

        model, tok = load_4bit_model(config.model_name)
        if patch_cache is not None:
            apply_surgical_patch(model, patch_cache, config)

        state["interference"][cond_name] = {}
        for ds in datasets:
            # Evaluate on flipped examples for recovery rate
            if flipped[ds]:
                res = run_inference(model, tok, flipped[ds], config,
                                    f"{cond_name}/{ds} (flipped)")
                rec = res["correct"] / len(flipped[ds])
            else:
                rec = 0.0
            # Also evaluate on full examples for accuracy delta
            full_res = run_inference(model, tok, examples[ds], config,
                                     f"{cond_name}/{ds} (full)")
            state["interference"][cond_name][ds] = {
                "recovery_rate": round(rec, 4),
                "full_accuracy": full_res["accuracy"],
                "full_per_example": full_res["per_example_correct"],
            }

        unload_model(model, tok)
        save_checkpoint(state, config.checkpoint_file)

    # Print interference table
    print(f"\n  {'Condition':<22} {'GSM8K rec':>10} {'ARC rec':>9} {'MATH rec':>9}")
    print("  " + "-" * 52)
    for cond_name, _ in interference_conditions:
        r = state["interference"][cond_name]
        print(f"  {cond_name:<22} {r['gsm8k']['recovery_rate']:>10.1%} "
              f"{r['arc']['recovery_rate']:>9.1%} {r['math']['recovery_rate']:>9.1%}")

    # =========================================================================
    # Phase D: Surgical capstone
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE D: Surgical Capstone (full dataset evaluation)")
    print(f"{'='*70}")

    if "capstone" not in state:
        state["capstone"] = {}

    for condition in ["fp16", "int4", "surgical"]:
        if condition in state["capstone"] and all(
            ds in state["capstone"][condition] for ds in datasets
        ):
            print(f"  {condition} capstone loaded from checkpoint.")
            continue

        if condition == "fp16":
            model, tok = load_fp16_model(config.model_name)
        elif condition == "int4":
            model, tok = load_4bit_model(config.model_name)
        else:
            model, tok = load_4bit_model(config.model_name)
            apply_surgical_patch(model, combined_cache, config)

        state["capstone"].setdefault(condition, {})
        for ds in datasets:
            if ds in state["capstone"][condition]:
                continue
            res = run_inference(model, tok, examples[ds], config,
                                f"capstone/{condition}/{ds}")
            state["capstone"][condition][ds] = {
                "accuracy": res["accuracy"],
                "correct": res["correct"],
                "total": res["total"],
                "per_example_correct": res["per_example_correct"],
            }
            save_checkpoint(state, config.checkpoint_file)

        unload_model(model, tok)
        save_checkpoint(state, config.checkpoint_file)

    # Capstone summary
    print(f"\n  {'Model':<12} {'GSM8K':>8} {'ARC':>8} {'MATH':>8} {'Avg':>8}")
    print("  " + "-" * 48)
    for condition in ["fp16", "int4", "surgical"]:
        accs = [state["capstone"][condition][ds]["accuracy"] for ds in datasets]
        print(f"  {condition:<12} {accs[0]:>8.2%} {accs[1]:>8.2%} {accs[2]:>8.2%} "
              f"{sum(accs)/len(accs):>8.2%}")

    # Gap recovery
    for ds in datasets:
        fp16_a = state["capstone"]["fp16"][ds]["accuracy"]
        int4_a = state["capstone"]["int4"][ds]["accuracy"]
        surg_a = state["capstone"]["surgical"][ds]["accuracy"]
        gap = fp16_a - int4_a
        recovery = (surg_a - int4_a) / gap * 100 if abs(gap) > 0.005 else None
        print(f"  {ds} gap recovery: {recovery:.1f}%" if recovery is not None
              else f"  {ds} gap recovery: N/A (gap < 0.5pp)")

    # =========================================================================
    # Phase E: 3-seed statistical validation
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE E: 3-Seed Statistical Validation")
    print(f"{'='*70}")

    if "seed_validation" not in state:
        state["seed_validation"] = {}

    for seed in config.sweep_seeds:
        seed_key = f"seed_{seed}"
        if seed_key in state["seed_validation"] and all(
            cond in state["seed_validation"][seed_key]
            for cond in ["fp16", "int4", "surgical"]
        ):
            print(f"  Seed {seed} loaded from checkpoint.")
            continue

        seed_examples = {
            "gsm8k": load_gsm8k(config.num_examples, seed),
            "arc":   load_arc_challenge(config.num_examples, seed),
            "math":  load_math(config.num_examples, seed),
        }

        state["seed_validation"].setdefault(seed_key, {})
        for condition in ["fp16", "int4", "surgical"]:
            if condition in state["seed_validation"][seed_key]:
                continue
            if condition == "fp16":
                model, tok = load_fp16_model(config.model_name)
            elif condition == "int4":
                model, tok = load_4bit_model(config.model_name)
            else:
                model, tok = load_4bit_model(config.model_name)
                apply_surgical_patch(model, combined_cache, config)

            state["seed_validation"][seed_key][condition] = {}
            for ds in datasets:
                res = run_inference(model, tok, seed_examples[ds], config,
                                    f"seed{seed}/{condition}/{ds}")
                state["seed_validation"][seed_key][condition][ds] = {
                    "accuracy": res["accuracy"],
                    "per_example_correct": res["per_example_correct"],
                }
            unload_model(model, tok)
            save_checkpoint(state, config.checkpoint_file)

    # Aggregate and run McNemar's
    state["statistics"] = {}
    for ds in datasets:
        all_fp16, all_int4, all_surg = [], [], []
        for seed in config.sweep_seeds:
            sk = f"seed_{seed}"
            all_fp16 += state["seed_validation"][sk]["fp16"][ds]["per_example_correct"]
            all_int4 += state["seed_validation"][sk]["int4"][ds]["per_example_correct"]
            all_surg += state["seed_validation"][sk]["surgical"][ds]["per_example_correct"]

        state["statistics"][ds] = {
            "fp16_mean": round(sum(all_fp16) / len(all_fp16), 4),
            "int4_mean": round(sum(all_int4) / len(all_int4), 4),
            "surgical_mean": round(sum(all_surg) / len(all_surg), 4),
            "fp16_vs_int4":    mcnemar_test(all_fp16, all_int4),
            "fp16_vs_surgical": mcnemar_test(all_fp16, all_surg),
            "int4_vs_surgical": mcnemar_test(all_int4, all_surg),
        }

    print(f"\n  McNemar's test results (3 seeds pooled):")
    print(f"  {'Dataset':<8} {'FP16 vs 4-bit':>16} {'Surg vs FP16':>16} {'Surg vs 4-bit':>16}")
    print("  " + "-" * 60)
    for ds in datasets:
        s = state["statistics"][ds]
        def fmt(t): return f"p={t['p_value']:.3f} {'✓' if t['significant_at_0.05'] else 'ns'}"
        print(f"  {ds:<8} {fmt(s['fp16_vs_int4']):>16} "
              f"{fmt(s['fp16_vs_surgical']):>16} {fmt(s['int4_vs_surgical']):>16}")

    # =========================================================================
    # Save final report
    # =========================================================================
    def strip_vec(obj):
        if isinstance(obj, dict):
            return {k: strip_vec(v) for k, v in obj.items() if k != "per_example_correct"}
        if isinstance(obj, list):
            return [strip_vec(i) for i in obj]
        return obj

    report = {
        "config": {
            "model": config.model_name,
            "num_examples": config.num_examples,
            "seed": config.seed,
            "selected_attn_layers": selected_attn,
            "selected_mlp_layers": selected_mlp,
        },
        "baselines": strip_vec(state["baselines"]),
        "flipped_counts": state["flipped_counts"],
        "layer_rankings": state["layer_rankings"],
        "selected_layers": state["selected_layers"],
        "interference": strip_vec(state["interference"]),
        "capstone": strip_vec(state["capstone"]),
        "statistics": state["statistics"],
    }

    tmp = config.output_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, config.output_file)
    print(f"\n  Report saved: {config.output_file}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
