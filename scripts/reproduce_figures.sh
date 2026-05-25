#!/usr/bin/env bash
# Regenerate every figure in the paper from the shipped JSON results.
# No GPU required; only matplotlib + numpy + scipy.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Regenerating figures from results/"
python src/plots/generate_plots.py --results-dir results

echo "==> Regenerating standalone teaser (Figure 0)"
python src/plots/generate_teaser.py

echo "==> Regenerating standalone rho scatter"
python src/plots/generate_rho_scatter.py

echo ""
echo "Done. See figures/ for output."
