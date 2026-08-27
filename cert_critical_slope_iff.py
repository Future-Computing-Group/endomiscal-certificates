"""Certificate: the critical-slope threshold (thm:critical-slope(iv)), both directions.

For named generators and 20 seeded random admissible generators:
  IF:   the truncated line at Lambda just above Lambda_c attains first-best.
  ONLY IF (family evidence): just below Lambda_c, the truncated line at every
        anchor AND the largest Lambda-Lipschitz rule under the ramp (the
        best-in-class construction of prop:below-threshold) all leave a
        strictly positive gap.  This half is a no-counterexample-found search
        over rule families, not a proof; the proof is the paper's.
Also: Lambda_c >= max(Lambda*, 1/m) (cor:lambda-star); Lambda_c = Lambda* under
the monotone-ramp-slope condition; Lambda_c > Lambda* for the engineered spike
with the printed ratio 5.30.
"""
import numpy as np
import common as C

def run_case(gen, p_min, ratio, verbose=True):
    r0 = C.reserve(gen, p_min, ratio)
    if r0 is None or r0 > 0.995:
        return None
    m = r0 - p_min
    lc, a_star = C.lambda_c(gen, p_min, r0, ratio)
    ls = C.lambda_star(gen, p_min, r0, ratio)
    C.check(f"{gen.name}: Lambda_c >= Lambda* and >= 1/m (rel tol 5e-4 for the grid sup)",
            lc >= ls * (1 - 5e-4) and lc >= (1 / m) * (1 - 5e-4),
            f"Lc={lc:.4f} L*={ls:.4f} 1/m={1/m:.4f}")
    fine = np.linspace(max(0, p_min - 0.02), min(1, r0 + 0.05), 80_001)
    # IF: truncated line just above Lambda_c screens first-best
    lam_hi = max(lc, ls) * 1.005
    gap_hi, _, _ = C.welfare_gap(gen, p_min, ratio, lambda r: C.truncated_line(r0, lam_hi, r), extra_r=fine)
    C.check(f"{gen.name}: line at 1.005*Lambda_c attains first-best", gap_hi < 1e-6, f"gap={gap_hi:.2e}")
    # ONLY IF evidence: 5% below, both families strictly fail
    lam_lo = lc * 0.95
    gap_line, _, _ = C.welfare_gap(gen, p_min, ratio, lambda r: C.truncated_line(r0, lam_lo, r), extra_r=fine)
    rg, qhat = C.lipschitz_envelope_under_ramp(gen, p_min, r0, ratio, lam_lo,
                                               rg=np.unique(np.concatenate([C.RGRID, fine])))
    gap_hat, _, _ = C.welfare_gap(gen, p_min, ratio, lambda r, rg=rg, q=qhat: np.interp(r, rg, q), extra_r=fine)
    floor = max(5e-8, 3 * gap_hi)           # instance-relative: clearly above attainment noise
    C.check(f"{gen.name}: 0.95*Lambda_c leaves a positive gap (line and envelope)",
            gap_line > floor and gap_hat > floor, f"line={gap_line:.2e} envelope={gap_hat:.2e} floor={floor:.1e}")
    return lc, ls

print("== named generators ==")
run_case(C.brier(), 0.4, 0.09)
run_case(C.halfcubic(), 0.4, 0.09)
run_case(C.quartic_mix(), 0.4, 0.09)
run_case(C.logscore(), 0.6, 0.04)
run_case(C.power(4), 0.35, 0.02)
lc, ls = run_case(C.engineered_spike(), 0.4, 0.2)
C.check("spike: Lambda_c/Lambda* = 5.30 (printed ratio)", abs(lc / ls - 5.30) < 0.02, f"{lc/ls:.3f}")

print("== 20 seeded random admissible generators ==")
rng = np.random.default_rng(8675309)
done = 0
while done < 20:
    gen = C.random_gen(rng, done)
    p_min = float(rng.uniform(0.25, 0.6))
    ratio = float(rng.uniform(0.02, 0.15))
    out = run_case(gen, p_min, ratio)
    if out is not None:
        done += 1
print("ALL PASS: critical-slope iff")
