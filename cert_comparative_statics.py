"""Certificate: comparative statics of the reserve.

(1) p_min = (u_d - u_f)/(u_s - u_f): numeric partial in u_f equals
    (u_d - u_s)/(u_s - u_f)^2 and is strictly negative (u_d < u_s).
(2) Raising p_min raises r_0 for every generator tested (bisection on D_G).
(3) Brier: m = sqrt(gamma/beta) and Lambda* = 2/m do not move with p_min.
"""
import numpy as np
import common as C

print("== comparative statics ==")
us, ud, uf = 1.0, 0.55, 0.2
pmin = lambda us, ud, uf: (ud - uf) / (us - uf)
h = 1e-6
num = (pmin(us, ud, uf + h) - pmin(us, ud, uf - h)) / (2 * h)
formula = (ud - us) / (us - uf) ** 2
C.check("d p_min / d u_f matches (u_d-u_s)/(u_s-u_f)^2", abs(num - formula) < 1e-6, f"num={num:.6f} formula={formula:.6f}")
C.check("d p_min / d u_f < 0", num < 0)

for gen, ratio in [(C.brier(), 0.09), (C.halfcubic(), 0.09), (C.logscore(), 0.04), (C.power(4), 0.02), (C.engineered_spike(), 0.2)]:
    r_lo = C.reserve(gen, 0.35, ratio)
    r_hi = C.reserve(gen, 0.45, ratio)
    C.check(f"{gen.name}: p_min up => r_0 up", r_hi > r_lo, f"{r_lo:.5f} -> {r_hi:.5f}")

for pm in [0.3, 0.4, 0.5]:
    g = C.brier()
    r0 = C.reserve(g, pm, 0.09)
    C.check(f"brier p_min={pm}: m = sqrt(ratio), Lambda* = 2/m",
            abs((r0 - pm) - 0.3) < 1e-9 and abs(C.lambda_star(g, pm, r0, 0.09) - 2 / 0.3) < 1e-6)
print("ALL PASS: comparative statics")
