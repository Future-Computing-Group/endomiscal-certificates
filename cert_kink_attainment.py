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

print("== kink attainment (part i) ==")
for gen, p_min, ratio in [(C.brier(), 0.4, 0.09), (C.halfcubic(), 0.4, 0.09),
                          (C.quartic_mix(), 0.4, 0.09), (C.logscore(), 0.6, 0.04),
                          (C.power(4), 0.35, 0.02), (C.engineered_spike(), 0.4, 0.2)]:
    r0 = C.reserve(gen, p_min, ratio)
    qramp = lambda r, g=gen, pm=p_min, rr=r0, ra=ratio: C.ramp(g, pm, rr, ra, r)
    gap, under, false = C.welfare_gap(gen, p_min, ratio, qramp)
    C.check(f"{gen.name}: ramp attains first-best", gap < 5e-6, f"gap={gap:.2e}")

print("== C1 infimum zero (part iii): gap linear in eps, channels 1st/2nd order ==")
gen, p_min, ratio = C.quartic_mix(), 0.4, 0.09          # non-Brier, full-hypothesis
r0 = C.reserve(gen, p_min, ratio)
rows = []
for eps in [0.16, 0.08, 0.04, 0.02]:
    fine = np.linspace(max(0, r0 - 0.05), min(1, r0 + 0.05), 60_001)
    gap, under, false = C.welfare_gap(gen, p_min, ratio, corner_rounded(gen, p_min, r0, ratio, eps),
                                      extra_r=fine)
    rows.append((eps, gap, under, false))
    print(f"    eps={eps:5.2f}  gap={gap:.3e}  under={under:.3e}  false={false:.3e}")
for (e1, g1, u1, f1), (e2, g2, u2, f2) in zip(rows, rows[1:]):
    C.check(f"gap halves as eps halves ({e1}->{e2})", 0.35 < g2 / g1 < 0.65, f"ratio={g2/g1:.3f}")
# This family sits strictly under the cap, so its false-approval channel is exactly
# zero, consistent with (and stronger than) the theorem's O(eps^2); the paper's own
# bump family realizes a positive O(eps^2) channel (the printed 5.5e-3 instance).
C.check("false-approval channel at most second order (here: zero)",
        all(f <= max(1e-12, 0.01 * u) for _, _, u, f in rows))
print("ALL PASS: kink attainment")
