"""Certificate: the two grid-computed illustrations, run exactly as the paper prints their recipes.

(a) The mollified kink (the paragraph after thm:kink, "Where the smoothing loss sits").
    Brier, p_min = 0.4, gamma/beta = 0.09, eps = 0.02, the C^1 family of the proof of
    thm:kink(iii), q_eps = ramp + (1 - ramp) phi((r - r_0 + eps)/eps) with the standard
    bump phi(x) = e^{-1/x} / (e^{-1/x} + e^{-1/(1-x)}).  Printed: the falsely approved
    interval below p_min has width 5.5e-3; the under-approved interval above p_min is the
    whole of (p_min, r_0); min_{p > p_min} tau = 0.9862 on a 4001-point type grid.
(b) The engineered-spike footnote of cor:lambda-star.  G'' = 2 + 100 exp(-((r-0.45)/0.02)^2),
    p_min = 0.4, gamma/beta = 0.2, F uniform, G and G' by cumulative trapezoid from G'',
    D_G from its defining formula, global argmax on the 200,001-point report grid over 999
    types uniform on [0.001, 0.999].  Printed: the binding chord anchors at a* = 0.4326;
    a truncated line of slope 1.02 Lambda* has zero crossing -0.099, pays q(0) = 0.1525,
    falsely approves every type below p_min (mass 0.40), mis-screens roughly 0.45 of the
    type mass in all, with pointwise screening error reaching 0.80; a line of slope
    1.0005 Lambda_c screens exactly.

Both are floating-point instance checks of numerals the paper labels as grid-computed
illustrations; nothing here is a theorem.
"""
import numpy as np
import common as C

# ------------------------------------------------------------------ (a) mollified kink
print("== (a) the mollified kink at the printed Brier instance ==")
gen, p_min, ratio, eps = C.brier(), 0.4, 0.09, 0.02
r0 = C.reserve(gen, p_min, ratio)
m = r0 - p_min

def bump(x):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        a = np.exp(-1.0 / np.maximum(x, 1e-300))
        b = np.exp(-1.0 / np.maximum(1.0 - x, 1e-300))
    return np.where(x <= 0, 0.0, np.where(x >= 1, 1.0, a / (a + b)))

def q_eps(r):
    rp = C.ramp(gen, p_min, r0, ratio, r)
    return rp + (1.0 - rp) * bump((np.asarray(r, dtype=float) - r0 + eps) / eps)

qv = q_eps(C.RGRID)
C.check("q_eps = ramp on [0, r0 - eps]", np.allclose(qv[C.RGRID <= r0 - eps], C.ramp(gen, p_min, r0, ratio, C.RGRID[C.RGRID <= r0 - eps])))
C.check("q_eps = 1 on [r0, 1]", np.all(qv[C.RGRID >= r0] == 1.0))
C.check("ramp <= q_eps <= 1 and q_eps non-decreasing", np.all(qv >= C.ramp(gen, p_min, r0, ratio, C.RGRID) - 1e-15) and np.all(qv <= 1.0) and np.all(np.diff(qv) >= -1e-15))

types = np.linspace(0.0, 1.0, 4001)
types = types[np.abs(types - p_min) > 1e-9]
tau = C.induced(gen, p_min, ratio, qv, types)
above = (types > p_min)
min_tau_above = float(tau[above].min())
under_band_hi = float(types[above & (tau < 1 - 1e-9)].max())
# q_eps < 1 strictly below r_0, so every type in (p_min, r_0) is under-approved; but the bump is flat to
# all orders at r_0, so within about 0.003 of the reserve 1 - q_eps is below double precision (1e-35 at
# r = r_0 - 0.00025).  The band is therefore checked at tolerance 1e-9 up to that sliver.
C.check("under-approved band is the whole of (p_min, r_0), up to the bump's flatness at r_0",
        np.all(tau[above & (types < r0 - 0.0035)] < 1 - 1e-9) and under_band_hi > r0 - 0.0035
        and np.all(tau[types >= r0 - 1e-9] == 1.0),
        f"band = (p_min, {under_band_hi:.5f}] at tolerance 1e-9, r0={r0:.5f}, m={m:.5f}")
C.check("min_{p>p_min} tau = 0.9862 on the 4001-point type grid", abs(min_tau_above - 0.9862) < 5e-5, f"min tau={min_tau_above:.5f}")

# the falsely approved interval: the set of types below p_min whose global best response receives a
# positive approval; its boundary is found by bisection on the type (the interval's width is a property
# of the rule, not of a grid), with the same approval criterion as the grid check below
def falsely_approved(p):
    return float(C.induced(gen, p_min, ratio, qv, np.array([p]))[0]) > 1e-9
lo, hi = 0.0, p_min - 1e-12
assert not falsely_approved(lo) and falsely_approved(hi)
for _ in range(60):
    mid = 0.5 * (lo + hi)
    if falsely_approved(mid):
        hi = mid
    else:
        lo = mid
boundary = p_min - 0.5 * (lo + hi)
fa_grid = types[(types < p_min) & (tau > 1e-9)]
width_grid = float(p_min - fa_grid.min())
C.check("falsely approved interval below p_min has width 5.5e-3 on the 4001-point type grid",
        abs(width_grid - 5.5e-3) < 1e-12 and len(fa_grid) == 22, f"{len(fa_grid)} grid types, deepest {fa_grid.min():.5f}, width={width_grid:.4e}")
# the sub-grid boundary lies one grid step or less beyond the deepest approved grid type
C.check("the exact boundary of that interval sits within one grid step of the grid value (5.61e-3)",
        width_grid <= boundary < width_grid + 2.5e-4, f"exact width={boundary:.4e}")
C.check("on the type grid the falsely approved types are exactly those inside the exact interval",
        np.all((tau[types < p_min] > 1e-9) == (types[types < p_min] > p_min - boundary)))

# ------------------------------------------------------------------ (b) engineered spike
print("== (b) the engineered-spike footnote of cor:lambda-star ==")
gen, p_min, ratio = C.engineered_spike(), 0.4, 0.2
r0 = C.reserve(gen, p_min, ratio)
lam_c, a_star = C.lambda_c(gen, p_min, r0, ratio)
lam_star = C.lambda_star(gen, p_min, r0, ratio)
C.check("binding chord anchored at a* = 0.4326", abs(a_star - 0.4326) < 5e-5, f"a*={a_star:.5f}")

types = np.linspace(0.001, 0.999, 999)
types = types[np.abs(types - p_min) > 1e-9]        # the F-null indifference type, excluded as the paper states its claims
first_best = (types >= p_min).astype(float)

def screen(lam):
    tau = C.induced(gen, p_min, ratio, C.truncated_line(r0, lam, C.RGRID), types)
    err = np.abs(tau - first_best)
    return tau, float(np.mean(err > 1e-6)), float(err.max())

lam_lo = 1.02 * lam_star
q_lo = C.truncated_line(r0, lam_lo, C.RGRID)
C.check("1.02 Lambda*: zero crossing r_0 - 1/(1.02 Lambda*) = -0.099", abs((r0 - 1.0 / lam_lo) - (-0.099)) < 5e-4, f"{r0 - 1.0/lam_lo:.4f}")
C.check("1.02 Lambda*: q(0) = 0.1525", abs(float(q_lo[0]) - 0.1525) < 5e-5, f"q(0)={float(q_lo[0]):.5f}")
tau_lo, mass_lo, max_lo = screen(lam_lo)
below = types < p_min
C.check("1.02 Lambda*: falsely approves every type below p_min, a mass of 0.40",
        np.all(tau_lo[below] > 1e-6) and abs(float(np.mean(below)) - 0.40) < 5e-3, f"mass below={np.mean(below):.4f}")
C.check("1.02 Lambda*: mis-screens 0.46 of the type mass in all (printed 2 d.p.)", abs(mass_lo - 0.46) < 0.005, f"mass={mass_lo:.4f}")
C.check("1.02 Lambda*: pointwise screening error reaches 0.80", abs(max_lo - 0.80) < 5e-3, f"max error={max_lo:.4f}")
_, mass_hi, max_hi = screen(1.0005 * lam_c)
C.check("1.0005 Lambda_c: screens exactly", mass_hi == 0.0 and max_hi < 1e-6, f"mass={mass_hi:.1e} max error={max_hi:.1e}")
# ------------------------------------------------------------------ (c) the wrong-threshold cost by the independent route
print("== (c) the Brier-formula threshold under G = p^2 + p^3/2: welfare cost from the induced screening ==")
gen, p_min, ratio = C.halfcubic(), 0.4, 0.09
r0 = C.reserve(gen, p_min, ratio)
C.check("halfcubic reserve r_0 = 0.62224 (printed)", abs(r0 - 0.62224) < 5e-6, f"r0={r0:.6f}")
wrong = lambda r: (np.asarray(r, dtype=float) >= 0.7).astype(float)
types = np.unique(np.concatenate([np.linspace(0.0, 1.0, 4001), np.linspace(0.38, 0.72, 68_001)]))
gap_w, under_w, false_w = C.welfare_gap(gen, p_min, ratio, wrong, types=types)
C.check("the wrong threshold costs 3.59e-3 of welfare, measured from the induced screening (printed 3 s.f.)",
        abs(gap_w - 3.59e-3) < 5e-6, f"gap={gap_w:.5e} (under={under_w:.3e}, false={false_w:.1e})")
print("ALL PASS: printed recipes")
