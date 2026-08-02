"""Deterministic bound on the c1^2 term that Route A's linearization drops.

Why this exists: the Tier-3 chain comparison (tier3_c1_validation.py) is a
wiring/consistency check, NOT a measurement of the c1^2 effect -- its mean gate
tolerates ~0.09-0.14 sigma_F and its MC noise floor is ~0.03-0.05 sigma_F, while
the effect itself is O(1e-5) sigma. The chains would pass identically with the
c1^2 term deleted or made 1000x larger, so they cannot be the evidence for
"Route A is safe". This script supplies evidence that can actually fail.

The bound. Production marginalizes c1 analytically by treating the model as
linear in it; the sampled-c1 model is exactly quadratic. Their entire difference
is the c1^2 coefficient of m0,

    q_b = 0.5 * d^2 m0_b / dc1_b^2   (constant in c1; H is frozen at theta0)

-- and nothing else, because c1 has no bilinear coupling to the marginalized
block (verified here: dM[:, :, c1] == 0 identically in every bin). The omitted
signal in whitened data space is therefore

    s(c1) = sqrt( sum_b (q_b c1^2)^T C_b^-1 (q_b c1^2) ) = c1^2 * sqrt(sum_b q_b^T C_b^-1 q_b),

which upper-bounds any parameter shift by |dtheta| <= s in sigma units (a
displacement of the data vector by s sigma cannot move any parameter by more
than s sigma_F). Using the raw per-bin C^-1 rather than the nuisance-projected
covariance makes this conservative.

Run: python3 scripts/tier3_c1_bound.py   (seconds; needs the c1-sampled cache)
Writes: cache/tier3_c1_bound.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
TEMPLATES = CACHE / "taylor_templates_lcdm_c1s.npz"
WHITENING = CACHE / "taylor_whitening_lcdm_c1s.npz"
OUT = CACHE / "tier3_c1_bound.json"

#: theta_NL layout of the c1-sampled split: 5 cosmology, then (b1, b2, bG2, c1)
#: per bin -> c1 of bin b sits at 5 + 4b + 3 = 8 + 4b.
N_COSMO_NL = 5
N_BINS = 7
C1_POS = [N_COSMO_NL + 4 * b + 3 for b in range(N_BINS)]

#: The chain gate's own resolution, from cache/tier3_c1_validation.json: mean
#: tolerances 0.090-0.137 sigma_F, MC noise 0.028-0.047 sigma_F. The bound must
#: sit far below these for the "chains cannot see it" statement to hold.
CHAIN_GATE_TOLERANCE_SIGMA_F = 0.090

#: Assert the omitted signal stays negligible even at a 5-sigma prior draw of c1
#: (prior width 1.0125 dimensionless). 1e-2 sigma is ~9x below the tightest chain
#: tolerance and ~1000x above the actual value -- a threshold that would catch a
#: real regression (e.g. a k_nl_rsd change or a template rebuild at a different
#: config) without tripping on float noise.
MAX_SIGNAL_AT_5SIGMA = 1e-2


def main() -> int:
    if not TEMPLATES.exists() or not WHITENING.exists():
        sys.exit(
            f"missing c1-sampled cache ({TEMPLATES.name} / {WHITENING.name}); "
            "build it with: python3 scripts/build_taylor_templates_lcdm.py --c1-sampled"
        )

    z = np.load(TEMPLATES, allow_pickle=False)
    wz = np.load(WHITENING, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    if meta.get("c1_treatment") != "sampled":
        sys.exit(f"templates are not the c1-sampled build: c1_treatment={meta.get('c1_treatment')!r}")

    cov_invs = np.asarray(wz["bin_cov_invs"])
    per_bin = []
    total_chi2 = 0.0
    for b in range(N_BINS):
        H = np.asarray(z[f"H_{b}"])
        dM = np.asarray(z[f"dM_{b}"])
        p = C1_POS[b]

        # c1 must not couple bilinearly to the marginalized block, or q_b would
        # not be the whole story.
        max_dM_c1 = float(np.abs(dM[:, :, p]).max())
        if max_dM_c1 != 0.0:
            sys.exit(
                f"bin {b}: dM[:, :, c1] is not identically zero (max {max_dM_c1:.3e}); "
                "the c1^2 coefficient is no longer the complete model difference "
                "and this bound is invalid"
            )

        q = 0.5 * H[:, p, p]
        chi2 = float(q @ cov_invs[b] @ q)
        total_chi2 += chi2
        per_bin.append({"bin": b, "c1_pos": p, "max_abs_q": float(np.abs(q).max()),
                        "chi2_at_c1_1": chi2, "max_abs_dM_c1": max_dM_c1})

    s1 = float(np.sqrt(total_chi2))          # omitted signal in sigma at |c1| = 1
    signals = {str(c): s1 * c ** 2 for c in (1, 3, 5)}
    ok = signals["5"] < MAX_SIGNAL_AT_5SIGMA

    out = {
        "description": "deterministic upper bound on the c1^2 term dropped by Route A",
        "templates": TEMPLATES.name,
        "c1_positions": C1_POS,
        "per_bin": per_bin,
        "total_chi2_at_c1_1": total_chi2,
        "omitted_signal_sigma": signals,
        "chain_gate_tolerance_sigma_F": CHAIN_GATE_TOLERANCE_SIGMA_F,
        "ratio_gate_over_signal_at_5sigma": CHAIN_GATE_TOLERANCE_SIGMA_F / signals["5"],
        "threshold_at_5sigma": MAX_SIGNAL_AT_5SIGMA,
        "verdict": "PASS" if ok else "FAIL",
    }
    OUT.write_text(json.dumps(out, indent=1))

    print(f"sum_b q^T Cinv q at |c1|=1 = {total_chi2:.4e}")
    for c, s in signals.items():
        print(f"  omitted signal at |c1|={c}: {s:.3e} sigma")
    print(f"chain gate tolerates {CHAIN_GATE_TOLERANCE_SIGMA_F} sigma_F "
          f"-> {out['ratio_gate_over_signal_at_5sigma']:.3g}x the 5-sigma-draw signal "
          "(i.e. the chains have no power here; this bound is the evidence)")
    print(f"-> {OUT}")
    print("VERDICT:", out["verdict"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
