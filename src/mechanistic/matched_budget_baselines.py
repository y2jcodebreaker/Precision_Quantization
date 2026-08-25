"""
Rebuttal baselines: RAND-6 and EARLIEST-6-MLP vs REST-6 (matched 6/32 FP16 budget).

Addresses ShEY-W ("Missing obvious baselines: random-6, earliest-6 MLP layers
predicted by Curse of Depth ... if random early layers match REST-6, novelty is
reduced") and the ShEY/tAz5 requests for confidence intervals.

Self-contained: evaluates FP16, NF4, REST-6, EARLIEST6_MLP and RAND-6 x N ALL in
one environment on the same examples, so recovery is computed within-run. No reuse
of frozen arrays, so it is robust to datasets/bitsandbytes/transformers version
drift on the runtime box.

Recovery uses the CORRECT (clean) definition:
    recovery = |FP16-right AND NF4-wrong AND cond-correct| / |FP16-right AND NF4-wrong|
(The original Phase-24 stats path used a numerator that dropped the FP16-right
 term; this script is the corrected reference and is the single source for the
 rebuttal baselines table.)

Conditions (all keep exactly 6/32 sub-layers at FP16; budget is matched at
SUB-LAYER granularity, not parameter count -- MLP sub-layers are ~4x larger than
attention sub-layers on Llama-3.1-8B, so EARLIEST6_MLP holds more FP16 weight than
REST-6. Per-condition FP16 parameter counts are reported in the output.)
  - REST-6         : L13a, L14a, L1m, L6m, L7m, L31m   (dissociation-guided)
  - EARLIEST6_MLP  : mlp L0..L5                          (Curse-of-Depth naive pick)
  - RAND-6 x N     : N random 6-sub-layer draws (seeds 101..)

Output: results/llama/matched_budget_baselines.json

Usage
-----
  pip install transformers bitsandbytes datasets scipy statsmodels torch numpy
  python src/mechanistic/matched_budget_baselines.py                 # default 5 random draws
  python src/mechanistic/matched_budget_baselines.py --n-random 3
  python src/mechanistic/matched_budget_baselines.py --stats-only    # recompute recovery+CI+McNemar from
                                               #   an existing output JSON (no GPU)
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from hessian_vs_restoration import (
    Config,
    RESTORATION_LAYERS,
    set_seed,
    aggressive_cleanup,
    mcnemar_p,
    load_fp16,
    load_nf4,
    cache_fp16_weights,
    transplant_layer_set,
    restore_layer_set,
    assert_nf4_clean,
    run_inference,
    load_gsm8k,
    load_arc_challenge,
    load_math,
    logger,
)

DATASETS = ("gsm8k", "arc", "math")
B_BOOT = 10000


# ---------------------------------------------------------------------------
# Clean recovery + bootstrap CI
# ---------------------------------------------------------------------------
def clean_recovery(fp_c, nf_c, cond_c) -> Tuple[float, int, int]:
    fp = np.asarray(fp_c, bool)
    nf = np.asarray(nf_c, bool)
    cd = np.asarray(cond_c, bool)
    flip = fp & ~nf
    nflip = int(flip.sum())
    rec = int((flip & cd).sum())
    return (rec / nflip if nflip else 0.0), rec, nflip


def bootstrap_recovery_ci(fp_c, nf_c, cond_c, rng, B=B_BOOT):
    fp = np.asarray(fp_c, bool)
    nf = np.asarray(nf_c, bool)
    cd = np.asarray(cond_c, bool)
    n = len(fp)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        flip = fp[idx] & ~nf[idx]
        if flip.sum() == 0:
            continue
        vals.append((flip & cd[idx]).sum() / flip.sum())
    if not vals:
        return (0.0, 0.0)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def condition_fp16_params(layer_set: List[Tuple[str, int]], fp16_cache: Dict) -> int:
    """Total FP16 weight parameters held by a layer set."""
    total = 0
    for comp, idx in layer_set:
        for _proj, info in fp16_cache[comp][idx].items():
            total += int(info["in_features"]) * int(info["out_features"])
    return total


# ---------------------------------------------------------------------------
# Layer-set definitions
# ---------------------------------------------------------------------------
def earliest6_mlp() -> List[Tuple[str, int]]:
    return [("mlp", i) for i in range(6)]


def random6(seed: int, num_layers: int) -> List[Tuple[str, int]]:
    """Draw 6 distinct (component, layer) sub-layers from the 64."""
    rng = np.random.default_rng(seed)
    pool = [("attention", i) for i in range(num_layers)] + \
           [("mlp", i) for i in range(num_layers)]
    idx = rng.choice(len(pool), size=6, replace=False)
    return [pool[i] for i in idx]


# ---------------------------------------------------------------------------
# Stats summariser (used by both full run and --stats-only)
# ---------------------------------------------------------------------------
def summarise(fp16: Dict[str, List[int]], nf4: Dict[str, List[int]],
              cond_arrays: Dict[str, Dict[str, List[int]]], rng) -> Dict:
    """cond_arrays: {cond_name: {ds: per_example_correct}}, must include 'REST-6'.
    Computes clean recovery + bootstrap CI + McNemar vs NF4/FP16/REST-6 per ds."""
    out = {}
    for ds in DATASETS:
        if ds not in fp16:
            continue
        fp_c, nf_c = fp16[ds], nf4[ds]
        rest_c = cond_arrays["REST-6"][ds]
        ds_out = {"fp16_acc": round(float(np.mean(fp_c)), 4),
                  "nf4_acc": round(float(np.mean(nf_c)), 4),
                  "flipped_count": None, "conditions": {}}
        for name, per in cond_arrays.items():
            cond_c = per[ds]
            rec, nrec, nflip = clean_recovery(fp_c, nf_c, cond_c)
            ci = bootstrap_recovery_ci(fp_c, nf_c, cond_c, rng)
            ds_out["flipped_count"] = nflip
            ds_out["conditions"][name] = {
                "acc": round(float(np.mean(cond_c)), 4),
                "recovery": round(rec, 4),
                "recovery_ci95": [round(ci[0], 4), round(ci[1], 4)],
                "recovered_count": nrec,
                "mcnemar_vs_nf4_p": round(mcnemar_p(cond_c, nf_c), 4),
                "mcnemar_vs_fp16_p": round(mcnemar_p(cond_c, fp_c), 4),
                "mcnemar_vs_rest6_p": round(mcnemar_p(cond_c, rest_c), 4),
            }
        out[ds] = ds_out
    return out


def print_table(summary: Dict):
    for ds in DATASETS:
        if ds not in summary:
            continue
        blk = summary[ds]
        print(f"\n[{ds.upper()}]  FP16={blk['fp16_acc']*100:.1f}%  "
              f"NF4={blk['nf4_acc']*100:.1f}%  flipped={blk['flipped_count']}")
        print(f"  {'condition':16s} {'acc':>6s} {'recovery':>9s} {'95% CI':>15s}"
              f" {'vsNF4':>7s} {'vsFP16':>7s} {'vsREST6':>8s}")
        for name, c in blk["conditions"].items():
            lo, hi = c["recovery_ci95"]
            print(f"  {name:16s} {c['acc']*100:5.1f}% {c['recovery']*100:8.1f}%"
                  f"  [{lo*100:4.1f},{hi*100:5.1f}]"
                  f" {c['mcnemar_vs_nf4_p']:7.4f} {c['mcnemar_vs_fp16_p']:7.4f}"
                  f" {c['mcnemar_vs_rest6_p']:8.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Rebuttal baselines: RAND-6 / EARLIEST-6-MLP")
    ap.add_argument("--n-random", type=int, default=5,
                    help="number of random 6-sub-layer draws (default 5)")
    ap.add_argument("--stats-only", action="store_true",
                    help="recompute recovery+CI+McNemar from an existing output JSON")
    args = ap.parse_args()

    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_path = os.path.join("results","llama","matched_budget_baselines.json")
    ckpt_path = os.path.join(cfg.output_dir, "matched_budget_baselines_ckpt.json")
    rng = np.random.default_rng(42)  # project-standard seed; must match bootstrap_ci_tost.py

    # ---- checkpoint helpers: persist baselines + each finished condition so a
    #      crash/preemption resumes instead of restarting the whole ~4h run ----
    def load_ckpt():
        if os.path.exists(ckpt_path):
            with open(ckpt_path) as f:
                st = json.load(f)
            logger.info("Resuming from checkpoint: fp16=%s nf4=%s conditions_done=%s",
                        bool(st.get("fp16")), bool(st.get("nf4")),
                        list(st.get("condition_arrays", {}).keys()))
            return st
        return {"fp16": None, "nf4": None, "condition_arrays": {}}

    def save_ckpt(st):
        tmp = ckpt_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(st, f)
        os.replace(tmp, ckpt_path)  # atomic: never leaves a half-written checkpoint

    # ---- stats-only: recompute from stored arrays, no GPU ----
    if args.stats_only:
        if not os.path.exists(out_path):
            raise FileNotFoundError(f"{out_path} not found; run a full pass first")
        with open(out_path) as f:
            prev = json.load(f)
        fp16 = prev["baselines"]["fp16"]
        nf4 = prev["baselines"]["nf4"]
        cond_arrays = {n: v["per_example_correct"]
                       for n, v in prev["condition_arrays"].items()}
        prev["summary"] = summarise(fp16, nf4, cond_arrays, rng)
        with open(out_path, "w") as f:
            json.dump(prev, f, indent=2)
        print_table(prev["summary"])
        return

    # ---- define conditions (REST-6 first so it's evaluated like any other) ----
    layer_sets: Dict[str, List[Tuple[str, int]]] = {
        "REST-6": list(RESTORATION_LAYERS),
        "EARLIEST6_MLP": earliest6_mlp(),
    }
    for k in range(args.n_random):
        layer_sets[f"RAND6_s{101 + k}"] = random6(101 + k, cfg.num_layers)
    for name, ls in layer_sets.items():
        logger.info("Condition %s -> %s", name, ls)

    set_seed(cfg.seed)
    state = load_ckpt()
    condition_arrays: Dict[str, Dict] = dict(state["condition_arrays"])

    everything_done = (state["fp16"] and state["nf4"] and
                       all(n in condition_arrays for n in layer_sets))

    if everything_done:
        logger.info("Checkpoint already has all baselines + conditions; skipping GPU.")
        fp16 = state["fp16"]
        nf4 = state["nf4"]
    else:
        logger.info("Loading datasets...")
        all_examples = {
            "gsm8k": load_gsm8k(cfg.n_examples),
            "arc": load_arc_challenge(cfg.n_examples),
            "math": load_math(cfg.n_examples),
        }

        # ---- FP16 baseline (own model load; freed before NF4) ----
        if state["fp16"]:
            logger.info("FP16 baseline from checkpoint.")
            fp16 = state["fp16"]
        else:
            logger.info("Evaluating FP16 baseline...")
            fp16_model, tokenizer = load_fp16(cfg.model_name)
            fp16 = {}
            for ds, examples in all_examples.items():
                res = run_inference(fp16_model, tokenizer, examples,
                                    cfg.batch_size, cfg.max_new_tokens)
                fp16[ds] = [int(r["correct"]) for r in res]
                logger.info("  FP16 %s acc=%.1f%%", ds, np.mean(fp16[ds]) * 100)
            del fp16_model
            aggressive_cleanup()
            state["fp16"] = fp16
            save_ckpt(state)

        # ---- FP16 weight cache + NF4 model (only if NF4 or any condition remains) ----
        remaining = [n for n in layer_sets if n not in condition_arrays]
        need_gpu = (state["nf4"] is None) or bool(remaining)
        nf4_model = None
        if need_gpu:
            fp16_cache = cache_fp16_weights(cfg.model_name, cfg.num_layers)
            nf4_model, tokenizer = load_nf4(cfg.model_name)

        # ---- NF4 baseline ----
        if state["nf4"]:
            logger.info("NF4 baseline from checkpoint.")
            nf4 = state["nf4"]
        else:
            logger.info("Evaluating NF4 baseline...")
            nf4 = {}
            for ds, examples in all_examples.items():
                res = run_inference(nf4_model, tokenizer, examples,
                                    cfg.batch_size, cfg.max_new_tokens)
                nf4[ds] = [int(r["correct"]) for r in res]
                logger.info("  NF4 %s acc=%.1f%%", ds, np.mean(nf4[ds]) * 100)
            state["nf4"] = nf4
            save_ckpt(state)

        # ---- each surgical condition (checkpoint saved after each) ----
        for name, ls in layer_sets.items():
            if name in condition_arrays:
                logger.info("Condition %s from checkpoint; skipping.", name)
                continue
            logger.info("Evaluating %s ...", name)
            originals = transplant_layer_set(nf4_model, fp16_cache, ls, cfg.device)
            per = {}
            for ds, examples in all_examples.items():
                res = run_inference(nf4_model, tokenizer, examples,
                                    cfg.batch_size, cfg.max_new_tokens)
                per[ds] = [int(r["correct"]) for r in res]
                logger.info("  %s %s acc=%.1f%%", name, ds, np.mean(per[ds]) * 100)
            restore_layer_set(nf4_model, originals)
            assert_nf4_clean(nf4_model, cfg.num_layers)
            condition_arrays[name] = {
                "layers": [{"component": c, "layer": i} for c, i in ls],
                "fp16_params": condition_fp16_params(ls, fp16_cache),
                "per_example_correct": per,
            }
            state["condition_arrays"][name] = condition_arrays[name]
            save_ckpt(state)

        if nf4_model is not None:
            del nf4_model
            aggressive_cleanup()

    # ---- FP16 parameter budget per condition (from stored counts) ----
    rest_params = condition_arrays["REST-6"]["fp16_params"]
    logger.info("FP16 parameter budget (6 sub-layers each):")
    for name, arr in condition_arrays.items():
        logger.info("  %-16s %.1fM  (%.2fx REST-6)",
                    name, arr["fp16_params"] / 1e6, arr["fp16_params"] / rest_params)

    # ---- summarise + save ----
    cond_pe = {n: v["per_example_correct"] for n, v in condition_arrays.items()}
    summary = summarise(fp16, nf4, cond_pe, rng)

    output = {
        "experiment": "Rebuttal baselines: RAND-6 / EARLIEST-6-MLP vs REST-6",
        "model": cfg.model_name,
        "n_examples": cfg.n_examples,
        "seed": cfg.seed,
        "note": "Self-contained run; recovery uses the CORRECTED clean definition. "
                "Single source for the rebuttal baselines table.",
        "baselines": {"fp16": fp16, "nf4": nf4},
        "condition_arrays": condition_arrays,
        "summary": summary,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    dl = os.path.join("results","llama","matched_budget_baselines.json")
    try:
        with open(dl, "w") as f:
            json.dump(output, f, indent=2)
    except OSError:
        pass
    logger.info("Saved: %s", out_path)
    print_table(summary)


if __name__ == "__main__":
    main()
