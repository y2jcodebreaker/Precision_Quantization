"""Rebuttal statistics: bootstrap CIs and TOST equivalence tests.

Addresses reviewer requests (ShEY-Q2, tAz5-Q2, ShEY-Q5):
  1. Bootstrap 95% CIs on Spearman rho (aggregate + per task).
  2. Bootstrap 95% CIs on HESS-6 / REST-6 recovery rates and the REST/HESS ratio.
  3. TOST equivalence test for "Surgical == FP16" with a pre-specified margin.

Runs on the saved result JSONs; no GPU, no re-inference.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

RNG = np.random.default_rng(42)
B = 10000
DL = Path("results") / "llama"

RANK_JSON = DL / "sensitivity_ranking_comparison.json"
HVR_JSON = DL / "hessian_vs_restoration.json"

TOST_MARGIN = 0.05  # pre-specified equivalence margin: +/- 5 percentage points


def pct_ci(samples, lo=2.5, hi=97.5):
    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


# ---------------------------------------------------------------------------
# 1. Bootstrap CI on Spearman rho (restoration rank vs Hessian rank, n=64)
# ---------------------------------------------------------------------------
def bootstrap_rho():
    data = json.load(open(RANK_JSON))
    per_layer = data["per_layer"]
    rest, hess = [], []
    rest_task = {"gsm8k": [], "arc": [], "math": []}
    for _, v in per_layer.items():
        rest.append(v["restoration_aggregate"])
        hess.append(v["hessian_trace_normalized"])
        for t in rest_task:
            rest_task[t].append(v["restoration_recovery"][t])
    rest = np.array(rest)
    hess = np.array(hess)
    n = len(rest)

    def boot(x, y):
        obs = spearmanr(x, y).correlation
        vals = np.empty(B)
        for i in range(B):
            idx = RNG.integers(0, n, n)
            vals[i] = spearmanr(x[idx], y[idx]).correlation
        return obs, pct_ci(vals)

    out = {}
    obs, ci = boot(rest, hess)
    out["aggregate"] = (obs, ci)
    for t in rest_task:
        obs, ci = boot(np.array(rest_task[t]), hess)
        out[t] = (obs, ci)
    return out


# ---------------------------------------------------------------------------
# 2. Bootstrap CIs on recovery rates (HESS-6, REST-6) and the ratio
# ---------------------------------------------------------------------------
def bootstrap_recovery():
    data = json.load(open(HVR_JSON))
    res = data["results"]
    out = {}
    for task in ["gsm8k", "arc", "math"]:
        pe = res[task]["per_example_correct"]
        fp16 = np.array(pe["fp16"], dtype=bool)
        nf4 = np.array(pe["nf4"], dtype=bool)
        hess = np.array(pe["hessian"], dtype=bool)
        rest = np.array(pe["restoration"], dtype=bool)
        n = len(fp16)

        def recovery(mask_flip, mask_method):
            f = mask_flip.sum()
            return (mask_flip & mask_method).sum() / f if f else np.nan

        flipped = fp16 & ~nf4  # FP16 correct, NF4 wrong
        obs_h = recovery(flipped, hess)
        obs_r = recovery(flipped, rest)
        n_flip = int(flipped.sum())

        h_samp, r_samp, ratio_samp = [], [], []
        for _ in range(B):
            idx = RNG.integers(0, n, n)
            fl = fp16[idx] & ~nf4[idx]
            if fl.sum() == 0:
                continue
            rh = (fl & hess[idx]).sum() / fl.sum()
            rr = (fl & rest[idx]).sum() / fl.sum()
            h_samp.append(rh)
            r_samp.append(rr)
            if rh > 0:
                ratio_samp.append(rr / rh)
        out[task] = {
            "n_flipped": n_flip,
            "hess": (obs_h, pct_ci(h_samp)),
            "rest": (obs_r, pct_ci(r_samp)),
            "ratio_rest_over_hess": (
                (obs_r / obs_h if obs_h > 0 else np.nan),
                pct_ci(ratio_samp) if ratio_samp else (np.nan, np.nan),
            ),
        }
    return out


# ---------------------------------------------------------------------------
# 3. TOST equivalence: Surgical (REST-6) vs FP16, paired, margin +/- 5pp
# ---------------------------------------------------------------------------
def tost_equivalence():
    data = json.load(open(HVR_JSON))
    res = data["results"]
    out = {}
    for task in ["gsm8k", "arc", "math"]:
        pe = res[task]["per_example_correct"]
        fp16 = np.array(pe["fp16"], dtype=int)
        rest = np.array(pe["restoration"], dtype=int)
        n = len(fp16)
        diff = rest - fp16  # per-example paired difference in {-1,0,1}
        obs = diff.mean()  # acc_rest - acc_fp16
        boot = np.array([diff[RNG.integers(0, n, n)].mean() for _ in range(B)])
        ci = pct_ci(boot)
        equivalent = (ci[0] > -TOST_MARGIN) and (ci[1] < TOST_MARGIN)
        out[task] = {
            "acc_fp16": float(fp16.mean()),
            "acc_rest": float(rest.mean()),
            "mean_diff": float(obs),
            "ci90_equiv": ci,  # 95% two-sided CI == 90% TOST CI region
            "margin": TOST_MARGIN,
            "equivalent": bool(equivalent),
        }
    return out


def fmt_ci(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def main():
    print("=" * 70)
    print("1. BOOTSTRAP 95% CI ON SPEARMAN rho  (restoration vs Hessian, n=64)")
    print("=" * 70)
    for k, (obs, ci) in bootstrap_rho().items():
        print(f"  {k:10s}: rho = {obs:+.3f}   95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")

    print()
    print("=" * 70)
    print("2. BOOTSTRAP 95% CI ON RECOVERY RATES  (HESS-6 vs REST-6)")
    print("=" * 70)
    for task, r in bootstrap_recovery().items():
        print(f"  [{task.upper()}]  n_flipped = {r['n_flipped']}")
        oh, ch = r["hess"]
        orr, cr = r["rest"]
        rr, cr2 = r["ratio_rest_over_hess"]
        print(f"    HESS-6 recovery: {oh:6.1%}  95% CI [{ch[0]:.1%}, {ch[1]:.1%}]")
        print(f"    REST-6 recovery: {orr:6.1%}  95% CI [{cr[0]:.1%}, {cr[1]:.1%}]")
        if not np.isnan(rr):
            print(f"    REST/HESS ratio: {rr:5.1f}x  95% CI [{cr2[0]:.1f}x, {cr2[1]:.1f}x]")
    print()
    print("=" * 70)
    print(f"3. TOST EQUIVALENCE  (Surgical/REST-6 vs FP16, margin +/-{TOST_MARGIN:.0%})")
    print("=" * 70)
    for task, r in tost_equivalence().items():
        verdict = "EQUIVALENT" if r["equivalent"] else "not established"
        print(
            f"  [{task.upper()}]  FP16={r['acc_fp16']:.1%}  REST-6={r['acc_rest']:.1%}  "
            f"diff={r['mean_diff']:+.1%}  CI {fmt_ci(r['ci90_equiv'])}  -> {verdict}"
        )


if __name__ == "__main__":
    main()
