"""Certificate: the tilt identity (eq:tilt-identity) and the slope premiums.

Checks, for each closed-form generator: Lambda* computed from its definition
(a point value of G'' at the bisected reserve) equals (2/m) kappa(r_0) /
<kappa>_chi with the distance-weighted average computed by direct quadrature.
Then the printed slope premiums at p_min = 0.4, r_0 = 0.7: log +11.6%, p^4
+34.2% relative to Brier, and Brier ratio exactly 1.
"""
import numpy as np
import common as C

print("== tilt identity ==")
for gen, p_min, ratio in [(C.brier(), 0.4, 0.09), (C.halfcubic(), 0.4, 0.09),
                          (C.logscore(), 0.6, 0.04), (C.power(4), 0.35, 0.02),
                          (C.quartic_mix(), 0.4, 0.09)]:
    r0 = C.reserve(gen, p_min, ratio)
    m = r0 - p_min
    lhs = C.lambda_star(gen, p_min, r0, ratio)
    t = np.linspace(p_min, r0, 400_001)
    kappa_chi = np.trapezoid(gen.G2(t) * (t - p_min), t) * 2.0 / m ** 2
    rhs = (2.0 / m) * float(gen.G2(r0)) / kappa_chi
    C.check(f"{gen.name}: Lambda* = (2/m) kappa(r0)/<kappa>_chi", abs(lhs / rhs - 1) < 1e-4,
            f"lhs={lhs:.6f} rhs={rhs:.6f}")

print("== slope premiums at p_min=0.4, r_0=0.7 (m fixed) ==")
p_min, r0 = 0.4, 0.7
m = r0 - p_min
t = np.linspace(p_min, r0, 400_001)
def ratio_of(gen):
    kchi = np.trapezoid(gen.G2(t) * (t - p_min), t) * 2.0 / m ** 2
    return float(gen.G2(r0)) / kchi
rb, rl, rp4 = ratio_of(C.brier()), ratio_of(C.logscore()), ratio_of(C.power(4))
C.check("Brier curvature ratio = 1", abs(rb - 1) < 1e-9, f"{rb:.9f}")
C.check("log premium = 11.6%", abs(rl / rb - 1 - 0.116) < 5e-4, f"{100*(rl/rb-1):.2f}%")
C.check("p^4 premium = 34.2%", abs(rp4 / rb - 1 - 0.342) < 5e-4, f"{100*(rp4/rb-1):.2f}%")
print("ALL PASS: tilt identity")
