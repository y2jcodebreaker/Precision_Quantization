# Figure-by-figure reproducibility map

Every numbered figure, table, and appendix in the paper, with the exact command(s) that produce it.

Assumes you are at the repo root and have followed the install instructions in `README.md`.

---

## Figures (body)

| Figure | What it shows | Producing script | Input JSON | Output |
|---|---|---|---|---|
| **Fig 0** (teaser) | Three-panel: Problem / Diagnostic / Outcome | `src/plots/generate_teaser.py` | none (procedural) | `figures/fig0_teaser.{png,pdf}` |
| **Fig 1** (ρ scatter) | Hessian rank vs functional rank, ρ = −0.506 | `src/plots/generate_rho_scatter.py` *and* `src/plots/generate_plots.py` | `results/llama/sensitivity_ranking_comparison.json` | `figures/plot13_restoration_vs_hessian.png`, `figures/rho_scatter.{png,pdf}` |
| **Fig 2** (per-layer lines) | Per-layer recovery on GSM8K / ARC / MATH | `src/plots/generate_plots.py` | `results/llama/layer_sensitivity_heatmap.json` | `figures/plot5_layer_sensitivity_lines.png` |
| **Fig 3** (cross-task matrix) | 5×3 patch-by-task interference matrix | `src/plots/generate_plots.py` | `results/llama/cross_task_interference_report.json` | `figures/plot7_cross_task_interference.png` |
| **Fig 4** (multi-seed) | 5-seed McNemar significance panels | `src/plots/generate_plots.py` | `results/llama/multi_seed_significance_report.json` | `figures/plot10_multi_seed_significance.png` |
| **Fig 5** (HESS vs REST) | HESS-6 vs REST-6 recovery comparison | `src/plots/generate_plots.py` | `results/llama/hessian_vs_restoration.json` | `figures/plot16_hessian_vs_restoration.png` |

## Tables (body)

| Table | What it shows | Producing script | Input JSON |
|---|---|---|---|
| **Tab 1** | FP16 / NF4 / GPTQ baselines across 6 benchmarks | `src/mechanistic/baseline_comparison.py` | writes `results/llama/baseline_comparison.json` |
| **Tab 2** | Cross-model replication (LLaMA + Mistral + DeepSeek) | `src/multi_model/mistral_replication.py` + `src/multi_model/multi_model_probe.py` | writes to `results/mistral/` and `results/multi_model/` |
| **Tab 3** | HESS-6 vs REST-6 head-to-head (the key result) | `src/mechanistic/hessian_vs_restoration.py` | writes `results/llama/hessian_vs_restoration.json` |

## Appendices

| Appendix | Topic | Producing script | Input JSON |
|---|---|---|---|
| **A** | Multi-seed McNemar matrix | `src/probe/multi_seed_significance.py` → plot via `src/plots/generate_plots.py` | `multi_seed_significance_report.json` |
| **B** | Full 64-sub-layer heatmap | `src/mechanistic/layer_sensitivity_profiler.py` → plot via `generate_plots.py` | `layer_sensitivity_heatmap.json` |
| **C** | Surgical 6/32 capstone + patch-count ablation | `src/probe/optimal_mixed_precision.py` + `src/probe/patch_count_ablation.py` | `optimal_mixed_precision_report.json`, `patch_count_ablation_report.json` |
| **D** | Cross-model comparison (LLaMA vs Mistral) | `src/multi_model/mistral_replication.py` → plot via `generate_plots.py` | `results/mistral/mistral_replication_report.json` |
| **E** | MATH difficulty stratification (levels 1–5) | `src/probe/math_difficulty_stratification.py` → plot via `generate_plots.py` | `math_difficulty_stratification_report.json` |
| **F** | Bidirectional flip analysis | `src/mechanistic/bidirectional_flip_analysis.py` → plot via `generate_plots.py` | `bidirectional_flip_report.json` |
| **G** | GPTQ cross-method replication | `src/mechanistic/sensitivity_ranking_comparison.py --quantizer gptq` → `generate_plots.py` | `multi_quant_probe_ckpt.json` |
| **H** | Extended multi-model probe (DeepSeek/Qwen/Gemma) | `src/multi_model/multi_model_probe.py` → `generate_plots.py` | `multi_model_probe.json` |
| **I** | Mistral Hessian-vs-restoration replication | `src/mechanistic/mistral_hessian_replication.py` | `results/mistral/mistral_hessian_replication.json` |

---

## Statistical recomputation (no GPU)

All per-example correctness arrays are stored in the shipped JSON. You can recompute the McNemar tests without re-running inference:

```bash
pip install statsmodels

# HESS-6 vs REST-6 McNemar (the headline result)
python src/mechanistic/hessian_vs_restoration.py --stats-only \
       --input results/llama/hessian_vs_restoration.json

# Spearman rho on Hessian-vs-restoration ranks
python src/mechanistic/sensitivity_ranking_comparison.py --stats-only \
       --input results/llama/sensitivity_ranking_comparison.json
```

---

## Expected runtimes (single A100 80GB)

| Block | Approx wall-clock |
|---|---|
| Damage baseline + surgical transplant | < 1 hour |
| Full 64-sub-layer sweep (Fig 2 / App B) | ~6 hours |
| Cross-task interference (5 patches × 3 datasets) | ~2 hours |
| Multi-seed significance (5 seeds × full pipeline) | ~3 hours |
| Sensitivity ranking + Hessian computation | ~2 hours |
| HESS-6 vs REST-6 head-to-head | ~1 hour |
| Mistral full replication (everything) | ~10 hours |
| Multi-model probe (DeepSeek + Qwen + Gemma) | ~6 hours |

All experiments are **inference-only**; no model training is performed.
