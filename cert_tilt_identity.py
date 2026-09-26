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

print("== log score below the unit sum: the premium against Brier can take either sign ==")
def _log_ratio(pm, r0):
    t = np.linspace(pm, r0, 200_001); kap = 1.0 / (t * (1 - t)); w = t - pm
    return float(kap[-1] / (np.trapezoid(kap * w, t) / np.trapezoid(w, t)))
lo, hi = _log_ratio(0.2, 0.4), _log_ratio(0.1, 0.85)
C.check("p_min + r_0 < 1 with ratio below 1 (p_min 0.2, r_0 0.4)", lo < 1.0, f"ratio={lo:.4f}")
C.check("p_min + r_0 < 1 with ratio above 1 (p_min 0.1, r_0 0.85)", hi > 1.0, f"ratio={hi:.4f}")
C.check("at the unit sum the log score is in the class: ratio >= 1 (p_min 0.3, r_0 0.7)", _log_ratio(0.3, 0.7) >= 1.0 - 1e-9)
print("ALL PASS: tilt identity (log-score sign)")

print("== prop:score-selection: both cubic-law constants in closed form at fixed m, and the convex-kappa inequality ==")
p_min, r0 = 0.4, 0.7; m = r0 - p_min
def rho_sigma_kbar(gen):
    t = np.linspace(p_min, r0, 400_001); kap = np.asarray(gen.G2(t), dtype=float); w = t - p_min
    kmean = float(np.trapezoid(kap * w, t) / np.trapezoid(w, t))
    h = 1e-5; kprime = (float(gen.G2(r0 + h)) - float(gen.G2(r0 - h))) / (2 * h)
    return float(gen.G2(r0)) / kmean, m * kprime / float(gen.G2(r0)), float(gen.G1(r0) - gen.G1(p_min)) / m / kmean
rows = {}
for gen in [C.brier(), C.power(3), C.power(4), C.logscore(), C.quartic_mix(), C.halfcubic()]:
    ratio = float(gen.D(p_min, r0))                       # the prize that puts the reserve at 0.7 for this generator
    ls = C.lambda_star(gen, p_min, r0, ratio); sp = C.s_prime_r0(gen, p_min, r0, ratio)
    Ku_def = ls ** 2 * m ** 2 / (6 * sp)
    Gamma0 = float(gen.G1(r0) - gen.G1(p_min)); G1_def = ratio * ls ** 2 / (ratio * ls ** 2 + Gamma0 * sp * m)
    rho, sigma, kb = rho_sigma_kbar(gen)
    Ku_cf = rho * m ** 2 / (3 * (1 + sigma)); G1_cf = rho / (rho + (1 + sigma) * kb)
    rows[gen.name] = (rho, sigma, Ku_def)
    C.check(f"{gen.name}: upper constant Lambda*^2 m^2/(6 s'(r0)) = rho m^2/(3(1+sigma))", abs(Ku_def / Ku_cf - 1) < 1e-7, f"def={Ku_def:.7f} closed={Ku_cf:.7f}")
    C.check(f"{gen.name}: Gamma_1 = rho/(rho + (1+sigma) kbar/<kappa>)", abs(G1_def / G1_cf - 1) < 1e-7, f"def={G1_def:.6f} closed={G1_cf:.6f}")
    C.check(f"{gen.name}: Lambda* = 2 rho / m at fixed m", abs(ls * m / (2 * rho) - 1) < 1e-7, f"Lambda*={ls:.5f} 2rho/m={2*rho/m:.5f}")
    convex = gen.name in ("brier", "power3", "power4", "log", "quartic_mix", "halfcubic")   # kappa convex on [0.4, 0.7] for all six
    if convex and 0 <= sigma <= 2:
        C.check(f"{gen.name}: convex kappa with 0 <= sigma <= 2 gives rho <= 1 + sigma (equality only for Brier)",
                rho <= 1 + sigma + 1e-12 and (gen.name == "brier") == (abs(rho - 1 - sigma) < 1e-9), f"rho={rho:.4f} 1+sigma={1+sigma:.4f}")
Kb = rows["brier"][2]
for name, target in (("power4", 0.723), ("log", 0.710)):
    rho, sigma, Ku = rows[name]
    C.check(f"{name} at p_min 0.4, r_0 0.7: shortfall constant is {target:.3f} of Brier's (printed 3 s.f.)", abs(Ku / Kb - target) < 5e-4, f"ratio={Ku/Kb:.4f} = rho/(1+sigma) = {rho/(1+sigma):.4f}")
print("== the hypotheses of (iii) bite: a non-convex curvature step and a decreasing convex curvature both violate rho <= 1 + sigma ==")
step = C.from_curvature("sigmoid-step", lambda t: 2 + 100 / (1 + np.exp(-(np.asarray(t, dtype=float) - 0.65) / 0.01)))
rho_s, sig_s, _ = rho_sigma_kbar(step)
C.check("curvature step below the reserve (not convex): rho/(1+sigma) > 1, the score loses MORE than Brier", rho_s / (1 + sig_s) > 1, f"rho={rho_s:.3f} sigma={sig_s:.3f} ratio={rho_s/(1+sig_s):.3f}")
dec = C.Gen("p2-p3/12", lambda p: p ** 2 - p ** 3 / 12.0, lambda p: 2 * p - p ** 2 / 4.0, lambda p: 2 - np.asarray(p, dtype=float) / 2.0)
rho_d, sig_d, _ = rho_sigma_kbar(dec)
C.check("affine decreasing curvature (convex, sigma < 0): rho > 1 + sigma, so sigma >= 0 is needed", sig_d < 0 and rho_d > 1 + sig_d, f"rho={rho_d:.4f} 1+sigma={1+sig_d:.4f}")
print("== measured envelope-rule losses at equal m: the power4/Brier ratio approaches rho/(1+sigma) from above ==")
def env_gap(gen, ratio, z):
    lam = C.lambda_star(gen, p_min, r0, ratio) * (1 - z)
    fine = np.linspace(max(0, p_min - 0.02), min(1, r0 + 0.05), 60_001)
    rg, qh = C.lipschitz_envelope_under_ramp(gen, p_min, r0, ratio, lam, rg=np.unique(np.concatenate([C.RGRID, fine])))
    types = np.unique(np.concatenate([np.linspace(0, 1, 801), np.linspace(p_min - 0.02, p_min + m * z + 0.02, 2400)]))
    return C.welfare_gap(gen, p_min, ratio, lambda r, rg=rg, q=qh: np.interp(r, rg, q), types=types, extra_r=fine)[0]
ratios = []
for z in (0.2, 0.1, 0.05):
    gb = env_gap(C.brier(), 0.09, z); g4 = env_gap(C.power(4), float(C.power(4).D(p_min, r0)), z)
    ratios.append(g4 / gb); print(f"    zeta={z}: power4/brier measured = {ratios[-1]:.4f}")
lim = rows["power4"][0] / (1 + rows["power4"][1])
C.check("power4/Brier measured loss ratio decreases toward rho/(1+sigma) = 0.723", ratios[0] > ratios[1] > ratios[2] > lim and ratios[2] - lim < 0.02, f"ratios={[f'{x:.4f}' for x in ratios]} limit={lim:.4f}")
print("ALL PASS: tilt identity (score selection)")
