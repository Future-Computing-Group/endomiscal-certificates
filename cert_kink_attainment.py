"""Certificate: the kink rule (thm:kink (i) and (iii)).

(i)  The Bregman ramp attains exact first-best screening for every generator
     tested, including non-constant-curvature ones and the log score: measured
     welfare gap at grid tolerance, induced screening = the first-best
     indicator for every p != p_min.
(iii) A C^1 family under the cap (quadratic corner rounding of width eps) has
     welfare gap O(eps): the gap falls linearly in eps, the under-approval
     channel is first order while the false-approval channel is second order.
"""
import numpy as np
import common as C

def corner_rounded(gen, p_min, r0, ratio, eps):
    def q(r):
        r = np.asarray(r, dtype=float)
        x = gen.D(p_min, r) / ratio
        out = np.where(x <= 1 - eps, x, np.where(x >= 1 + eps, 1.0, 1 - ((1 + eps) - x) ** 2 / (4 * eps)))
        return np.where(r <= p_min, 0.0, np.minimum(out, 1.0))
    return q

def bump(x):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        a = np.exp(-1.0 / np.maximum(x, 1e-300)); b = np.exp(-1.0 / np.maximum(1.0 - x, 1e-300))
    return np.where(x <= 0, 0.0, np.where(x >= 1, 1.0, a / (a + b)))

def paper_family(gen, p_min, r0, ratio, eps):
    """thm:kink(iii)'s q_eps = ramp + (1 - ramp) phi((r - r_0 + eps)/eps) with the standard bump phi."""
    def q(r):
        rp = C.ramp(gen, p_min, r0, ratio, r)
        return rp + (1.0 - rp) * bump((np.asarray(r, dtype=float) - r0 + eps) / eps)
    return q

print("== kink attainment (part i) and the step rule of thm:reserve(i) ==")
for gen, p_min, ratio in [(C.brier(), 0.4, 0.09), (C.halfcubic(), 0.4, 0.09),
                          (C.quartic_mix(), 0.4, 0.09), (C.logscore(), 0.6, 0.04),
                          (C.power(4), 0.35, 0.02), (C.engineered_spike(), 0.4, 0.2)]:
    r0 = C.reserve(gen, p_min, ratio)
    qramp = lambda r, g=gen, pm=p_min, rr=r0, ra=ratio: C.ramp(g, pm, rr, ra, r)
    gap, under, false = C.welfare_gap(gen, p_min, ratio, qramp)
    C.check(f"{gen.name}: ramp attains first-best", gap < 5e-6, f"gap={gap:.2e}")
    # the pointwise identity tau = 1{p >= p_min}, away from a 3e-3 neighbourhood of the indifference type
    types = np.linspace(0.0, 1.0, 4001); types = types[np.abs(types - p_min) > 3e-3]
    if gen.interior: types = types[(types > 1e-3) & (types < 1 - 1e-3)]
    rg = np.unique(np.append(C.RGRID, r0))              # the reserve itself on the report grid, so tau(r_0) = ramp(r_0) = 1 exactly
    tau = C.induced(gen, p_min, ratio, qramp(rg), types, rgrid=rg)
    C.check(f"{gen.name}: ramp induces the first-best indicator pointwise (|p - p_min| > 3e-3)",
            np.max(np.abs(tau - (types >= p_min))) < 1e-6, f"max error={np.max(np.abs(tau - (types >= p_min))):.1e}")
    step = lambda r, rr=r0: (np.asarray(r, dtype=float) >= rr).astype(float)
    gap_s, _, _ = C.welfare_gap(gen, p_min, ratio, step)
    C.check(f"{gen.name}: the step rule at r_0 attains first-best (thm:reserve(i))", gap_s < 5e-6, f"gap={gap_s:.2e}")

print("== C1 infimum zero (part iii) on the paper's own family: under-approval O(eps), false approval O(eps^2) ==")
for gen, p_min, ratio in [(C.brier(), 0.4, 0.09), (C.quartic_mix(), 0.4, 0.09)]:
    r0 = C.reserve(gen, p_min, ratio)
    rows = []
    for eps in [0.16, 0.08, 0.04, 0.02]:
        fine = np.linspace(max(0, r0 - 0.2), min(1, r0 + 0.05), 80_001)
        gap, under, false = C.welfare_gap(gen, p_min, ratio, paper_family(gen, p_min, r0, ratio, eps), extra_r=fine, n_types=8001)
        rows.append((eps, gap, under, false))
        print(f"    {gen.name} eps={eps:5.2f}  gap={gap:.3e}  under={under:.3e}  false={false:.3e}")
    eps_ = np.array([r[0] for r in rows]); und = np.array([r[2] for r in rows]); fal = np.array([r[3] for r in rows])
    su = np.polyfit(np.log(eps_[1:]), np.log(und[1:]), 1)[0]; sf = np.polyfit(np.log(eps_[1:]), np.log(fal[1:]), 1)[0]   # exponents fitted on the three finest widths (the statement is asymptotic)
    C.check(f"{gen.name}: under-approval channel is first order in eps", 0.8 < su < 1.25, f"exponent={su:.2f}")
    C.check(f"{gen.name}: false-approval channel is second order in eps", 1.7 < sf < 2.4, f"exponent={sf:.2f}")
    C.check(f"{gen.name}: the false-approval channel is positive and its share of the loss falls with eps", np.all(fal > 0) and np.all(np.diff(fal / und) < 0), f"false/under={[f'{x:.2f}' for x in fal/und]}")
print("== the quadratic corner rounding, kept as a second C^1 family (strictly under the cap) ==")
gen, p_min, ratio = C.quartic_mix(), 0.4, 0.09
r0 = C.reserve(gen, p_min, ratio)
rows = []
for eps in [0.16, 0.08, 0.04, 0.02]:
    fine = np.linspace(max(0, r0 - 0.05), min(1, r0 + 0.05), 60_001)
    gap, under, false = C.welfare_gap(gen, p_min, ratio, corner_rounded(gen, p_min, r0, ratio, eps), extra_r=fine)
    rows.append((eps, gap, under, false))
for (e1, g1, u1, f1), (e2, g2, u2, f2) in zip(rows, rows[1:]):
    C.check(f"corner rounding: gap halves as eps halves ({e1}->{e2})", 0.35 < g2 / g1 < 0.65, f"ratio={g2/g1:.3f}")
C.check("corner rounding: false-approval channel identically zero (family under the cap)", all(f <= 1e-12 for _, _, _, f in rows))
print("== the weak-feasibility boundary r_0 = 1 and the saturated regime (thm:reserve(i), prop:lambda-sat) ==")
gen, p_min = C.brier(), 0.4
ratio_b = float(gen.D(p_min, 1.0))                     # gamma/beta = D_G(p_min, 1): r_0 = 1 exactly
r0b = C.reserve(gen, p_min, ratio_b)
C.check("Brier at the weak boundary: reserve() returns r_0 = 1", r0b == 1.0, f"r0={r0b}")
gap_b, _, _ = C.welfare_gap(gen, p_min, ratio_b, lambda r: C.ramp(gen, p_min, 1.0, ratio_b, r))
C.check("Brier at the weak boundary: the ramp attains first-best", gap_b < 5e-6, f"gap={gap_b:.2e}")
ratio_s = 1.5 * ratio_b                                # saturated: gamma/beta > D_G(p_min, 1)
lam_sat = float(gen.D(p_min, 1.0)) / ratio_s
C.check("Brier saturated: reserve() reports no reserve", C.reserve(gen, p_min, ratio_s) is None)
q_sat = lambda r: lam_sat * (np.asarray(r, dtype=float) >= 1.0)
types = np.linspace(0.0, 1.0, 4001); types = types[np.abs(types - p_min) > 1e-9]
tau_s = C.induced(gen, p_min, ratio_s, q_sat(C.RGRID), types)
C.check("Brier saturated: q_sat induces tau = lambda_sat 1{p >= p_min} (prop:lambda-sat)",
        np.allclose(tau_s, lam_sat * (types >= p_min), atol=1e-12), f"lambda_sat={lam_sat:.4f}")
print("ALL PASS: kink attainment")
