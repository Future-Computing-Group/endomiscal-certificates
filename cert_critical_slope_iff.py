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
    # thm:critical-slope(i)-(ii) on the realized first-best line: cap under the ramp, first full approval at r_0
    qv = C.truncated_line(r0, lam_hi, C.RGRID); rp = C.ramp(gen, p_min, r0, ratio, C.RGRID)
    inside = (C.RGRID > 1e-3) if gen.interior else np.ones_like(C.RGRID, dtype=bool)
    C.check(f"{gen.name}: the first-best line satisfies the cap q <= ramp and q = 0 below p_min",
            np.all(qv[inside] <= rp[inside] + 1e-12) and np.all(qv[C.RGRID <= p_min] == 0.0))
    first_one = float(C.RGRID[np.argmax(qv >= 1.0)])
    C.check(f"{gen.name}: inf{{r : q(r) = 1}} = r_0 on the grid", abs(first_one - r0) <= 2 * C.H, f"{first_one:.6f} vs r0={r0:.6f}")
    # cor:lambda-star's exact criterion: Lambda_c = Lambda* iff C lies above its tangent at r_0 on [p_min, r_0]
    # both sides at the same relative tolerance: the largest chord-over-tangent excess is (Lambda_c/Lambda* - 1)
    a = np.linspace(p_min, r0 - 1e-6, 20_001)
    Ca = gen.D(p_min, a); Cr0 = float(gen.D(p_min, r0)); slope = float(gen.G2(r0) * (r0 - p_min))
    excess = float(np.max((Cr0 - Ca) / (slope * (r0 - a)) - 1.0))
    tangent_support = excess < 2e-3
    equal = abs(lc / ls - 1.0) < 2e-3
    C.check(f"{gen.name}: tangent support of C at r_0 <=> Lambda_c = Lambda*  (support={tangent_support}, equal={equal})", tangent_support == equal)
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

print("== 20 seeded random admissible generators (curvature bumps of widths 0.005 to 0.3, so both classes are drawn) ==")
rng = np.random.default_rng(8675309)
done, corner, drawn = 0, 0, 0
while done < 20:
    gen = C.random_gen(rng, done, sigma_lo=0.005)
    p_min = float(rng.uniform(0.25, 0.6))
    ratio = float(rng.uniform(0.02, 0.15))
    drawn += 1
    out = run_case(gen, p_min, ratio, verbose=False)
    if out is not None:
        done += 1
        if out[0] / out[1] > 1.002: corner += 1
print(f"    drawn {drawn}, accepted {done}, corner class (Lambda_c > Lambda*) {corner}, tangency class {done - corner}")
print("== the corner class Lambda_c > Lambda*: a family of engineered curvature spikes strictly inside (p_min, r_0) ==")
# Random bump mixtures land in the tangency class (a spike must sit inside (p_min, r_0) and fall away before the
# reserve, which a random placement rarely does), so the corner class is exercised by an engineered family.
corner_eng = 0
for c, w in [(0.44, 0.015), (0.45, 0.02), (0.46, 0.025), (0.48, 0.015), (0.50, 0.02)]:
    g = C.from_curvature(f"spike(c={c},w={w})", lambda r, c=c, w=w: 2 + 100 * np.exp(-(((np.asarray(r, dtype=float)) - c) / w) ** 2))
    out = run_case(g, 0.4, 0.2, verbose=False)
    if out is not None and out[0] / out[1] > 1.002: corner_eng += 1
    print(f"    {g.name}: Lambda_c/Lambda* = {out[0]/out[1]:.3f}" if out else f"    {g.name}: not strictly feasible")
C.check("both classes are exercised: the random family in the tangency class, the engineered spikes in the corner class",
        done - corner >= 1 and corner + corner_eng >= 3, f"random corner={corner}, engineered corner={corner_eng}")
print("ALL PASS: critical-slope iff")
