# Shipped result files

These JSON files contain the per-example correctness arrays, McNemar inputs, and aggregate metrics produced by the experiment scripts. They are shipped here so reviewers can:

1. Regenerate every figure in the paper without a GPU (`python src/plots/generate_plots.py --results-dir results`).
2. Recompute all statistical tests (McNemar, BH-FDR, Spearman ρ) without re-running inference.
3. Verify the headline claims by inspecting the per-example arrays directly.

## File-to-paper mapping

| Subdirectory | Contents |
|---|---|
| `llama/` | LLaMA-3.1-8B-Instruct (the primary model, Sections 2–4) |
| `mistral/` | Mistral-7B-Instruct-v0.3 (cross-architecture replication, Section 4.1, Appendix I) |
| `multi_model/` | DeepSeek-R1-Distill, Qwen2.5-Math, Gemma-2, plus GPTQ checkpoint (Appendix H) |

See `../REPRODUCIBILITY.md` for the figure-by-figure command map.

## Note on checkpoints

The original experiment scripts also write intermediate `*_checkpoint.json` files during long sweeps. Those are excluded from the shipped repo (`.gitignore`) because they are large and only useful for crash recovery during a fresh run. The final aggregated `*_report.json` outputs are everything you need.
