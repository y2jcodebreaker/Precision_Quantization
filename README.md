# The Misallocation of Precision in Quantized Language Models — Code & Data

Reproducibility repository for the paper
**"The Misallocation of Precision in Quantized Language Models"**
(anonymous submission).

This repo contains:
- the 21 Python scripts used to produce every result, table, figure, and appendix in the paper;
- 18 pre-computed JSON result files (so a reviewer can regenerate all figures **without a GPU**);
- 16 pre-generated PNG figures (so a reviewer can visually verify the paper's claims with `git clone` alone).

---

## TL;DR for reviewers

```bash
# 1. clone / download this repo
# 2. install plotting deps only (no GPU needed)
pip install matplotlib numpy scipy

# 3. regenerate every paper figure from shipped JSON
python src/plots/generate_plots.py --results-dir results

# 4. inspect figures/
open figures/plot13_restoration_vs_hessian.png   # main rho = -0.506 scatter
open figures/plot16_hessian_vs_restoration.png   # HESS-6 vs REST-6 (the core finding)
```

If you want to re-run the underlying experiments on a GPU, see [Full reproduction](#full-reproduction-from-scratch) below.

---

## Note on the recovery metric (corrected 2026-08-25)

Recovery is the fraction of *forward-flipped* examples a condition answers correctly:

    recovery = |FP16 right AND NF4 wrong AND condition right|
             / |FP16 right AND NF4 wrong|

An earlier version of `src/mechanistic/hessian_vs_restoration.py` omitted the
"FP16 right" term from the numerator, which counted items FP16 also failed and
inflated recovery. The per-example correctness arrays shipped in
`results/llama/hessian_vs_restoration.json` are the ground truth and were never
affected; accuracies and McNemar p-values are computed from them and are
unchanged. The reducer and the plotting code now share one definition, and the
derived `*_recovery_rate` fields in that JSON were recomputed from the arrays.

## Repository layout

```
.
├── README.md                              <- this file
├── LICENSE                                <- MIT
├── requirements.txt                       <- pinned dependencies
├── REPRODUCIBILITY.md                     <- figure-by-figure command map
│
├── src/
│   ├── probe/                             <- the restoration probe (Sections 3-4)
│   │   ├── quantization_damage_report.py     baseline NF4 damage
│   │   ├── surgical_repair.py                FP16-into-NF4 transplant mechanism
│   │   ├── optimal_mixed_precision.py        the 6/32 surgical model (Sec 4.4, App C)
│   │   ├── multi_seed_significance.py        5-seed McNemar validation (Sec 4.4, App A)
│   │   ├── math_difficulty_stratification.py MATH null analysis (Sec 4.5, App E)
│   │   ├── additional_benchmarks.py          HellaSwag + MMLU contrast (Sec 4.5)
│   │   └── patch_count_ablation.py           2/4/6/8-layer ablation
│   │
│   ├── mechanistic/                       <- Hessian-vs-functional analysis (Sec 4.6, 4.9)
│   │   ├── layer_sensitivity_profiler.py     64-layer recovery sweep (Fig 2, App B)
│   │   ├── sensitivity_ranking_comparison.py Hessian trace + rho = -0.506 (Fig 4, Sec 4.6)
│   │   ├── hessian_vs_restoration.py         HESS-6 vs REST-6 head-to-head (Sec 4.9, Tab 3)
│   │   ├── matched_budget_baselines.py       RAND-6 / EARLIEST-6-MLP controls (App J)
│   │   ├── bootstrap_ci_tost.py              bootstrap CIs + TOST equivalence (Sec 4.4, 4.6, 4.9)
│   │   ├── mistral_hessian_replication.py    Mistral replication (rho = -0.574)
│   │   ├── cross_task_interference.py        5x3 patch-by-task matrix (Fig 3, Sec 4.3)
│   │   ├── bidirectional_flip_analysis.py    forward/reverse flip breakdown (App F)
│   │   └── baseline_comparison.py            FP16/NF4/GPTQ baselines (Tab 1, Sec 4.1)
│   │
│   ├── multi_model/                       <- cross-architecture probe (Sec 4.7, App H)
│   │   ├── mistral_replication.py            Mistral-7B-Instruct-v0.3 full pipeline
│   │   └── multi_model_probe.py              DeepSeek-R1 / Qwen2.5-Math / Gemma-2
│   │
│   └── plots/                             <- figure generation (no GPU required)
│       ├── generate_plots.py                 reads results/*.json, writes figures/*.png
│       ├── generate_teaser.py                Figure 0 (procedural; no JSON input)
│       └── generate_rho_scatter.py           standalone rho scatter (procedural)
│
├── results/                               <- shipped pre-computed JSON outputs
│   ├── llama/                                LLaMA-3.1-8B-Instruct
│   ├── mistral/                              Mistral-7B-Instruct-v0.3
│   └── multi_model/                          DeepSeek-R1 / Qwen2.5-Math / Gemma-2
│
└── figures/                               <- pre-generated PNGs for visual verification
```

---

## What's in `results/`

Each JSON contains the per-example correctness arrays, McNemar inputs, and aggregate metrics needed to regenerate the figures and statistical tables. Re-running `src/plots/generate_plots.py` against this directory deterministically produces every figure in the paper.

| File | Paper element |
|---|---|
| `llama/quantization_damage.json` | Section 4.1, baseline NF4 damage |
| `llama/baseline_comparison.json` | Table 1 (FP16 / NF4 / GPTQ across 6 benchmarks) |
| `llama/surgical_report.json` | Section 3.2, surgical transplant core results |
| `llama/layer_sensitivity_heatmap.json` | Figure 2, Appendix B (64-sub-layer sweep) |
| `llama/cross_task_interference_report.json` | Figure 3 (5×3 patch-by-task matrix) |
| `llama/multi_seed_significance_report.json` | Figure 4, Appendix A (5-seed McNemar) |
| `llama/math_difficulty_stratification_report.json` | Section 4.5, Appendix E |
| `llama/additional_benchmarks_report.json` | Section 4.5 (HellaSwag + MMLU contrast) |
| `llama/patch_count_ablation_report.json` | Appendix C (patch count ablation) |
| `llama/optimal_mixed_precision_report.json` | Appendix C (6/32 capstone) |
| `llama/sensitivity_ranking_comparison.json` | Figure 4, Section 4.6 (ρ = −0.506) |
| `llama/bidirectional_flip_report.json` | Appendix F (forward/reverse flips) |
| `llama/hessian_vs_restoration.json` | **Table 3, Section 4.9 (HESS-6 vs REST-6, p = 0.001)** |
| `mistral/mistral_replication_report.json` | Section 4.7 (cross-architecture) |
| `mistral/mistral_hessian_replication.json` | Section 4.7, Appendix I (Mistral ρ = −0.574) |
| `multi_model/multi_model_probe.json` | Appendix H (DeepSeek/Qwen/Gemma) |
| `multi_model/multi_quant_probe_ckpt.json` | GPTQ cross-method replication |

---

## Quick reproduction (no GPU)

The shipped JSONs contain every per-example score needed for the paper's claims.

```bash
# clone + install plot-only deps
git clone <THIS_REPO>
cd quantization-mechanistic-probe
pip install matplotlib numpy scipy

# regenerate every figure
python src/plots/generate_plots.py --results-dir results

# regenerate the standalone teaser
python src/plots/generate_teaser.py
python src/plots/generate_rho_scatter.py

# verify against the shipped figures/ directory
diff <(md5 figures/plot13_restoration_vs_hessian.png) <(md5 figures_new/plot13_restoration_vs_hessian.png)
```

To recompute statistical tests from the shipped per-example arrays without GPU:

```bash
pip install statsmodels
python src/mechanistic/hessian_vs_restoration.py --stats-only \
       --input results/llama/hessian_vs_restoration.json
```

---

## Full reproduction (from scratch)

Hardware: one NVIDIA A100 (80GB). All experiments are inference-only; no training.

```bash
# 1. install full dependencies
pip install -r requirements.txt

# 2. export HF token (for model access)
export HF_TOKEN=<your_hf_token>

# 3. set output directory
export RESULTS_DIR=./results

# 4. core probe (LLaMA-3.1-8B-Instruct) — Sections 3-4
python src/probe/quantization_damage_report.py
python src/probe/surgical_repair.py
python src/mechanistic/layer_sensitivity_profiler.py        # ~6 hours, the big sweep
python src/mechanistic/cross_task_interference.py
python src/probe/multi_seed_significance.py
python src/probe/math_difficulty_stratification.py
python src/probe/additional_benchmarks.py
python src/probe/patch_count_ablation.py
python src/probe/optimal_mixed_precision.py
python src/mechanistic/baseline_comparison.py
python src/mechanistic/bidirectional_flip_analysis.py

# 5. Hessian inversion analysis — Sections 4.6, 4.9
python src/mechanistic/sensitivity_ranking_comparison.py    # produces rho = -0.506
python src/mechanistic/hessian_vs_restoration.py            # HESS-6 vs REST-6

# 6. cross-architecture replication — Section 4.7
python src/multi_model/mistral_replication.py
python src/mechanistic/mistral_hessian_replication.py
python src/multi_model/multi_model_probe.py

# 7. regenerate figures
python src/plots/generate_plots.py --results-dir $RESULTS_DIR
```

See `REPRODUCIBILITY.md` for a figure-by-figure command map.

---

## Models and datasets

All assets are publicly released and used under their standard research licenses:

**Models** (HuggingFace):
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- `Qwen/Qwen2.5-Math-7B-Instruct`
- `google/gemma-2-9b-it`

**Datasets** (HuggingFace `datasets`):
- `openai/gsm8k`
- `allenai/ai2_arc` (ARC-Challenge)
- `hendrycks/competition_math`
- `winogrande` (winogrande_xl)
- `piqa`
- `Rowan/hellaswag`
- `cais/mmlu`
- `Salesforce/wikitext` (WikiText-2, for GPTQ calibration)

We use a 500-example evaluation subset for each benchmark (200 for MATH).

---

## License

MIT — see `LICENSE`.

## Citation

To be filled in upon acceptance.
