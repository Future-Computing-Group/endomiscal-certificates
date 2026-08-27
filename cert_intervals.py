"""Tier 2: RIGOROUS enclosures of the manuscript's printed numerals (Arb ball arithmetic).

Every value below is certified by ball arithmetic at 300-bit precision, in which
a comparison returns true only when it holds for every point of every ball.  The
enclosures are therefore proofs, not agreements between floating-point runs.

Method.  The reserve is enclosed by a verified sign change: r |-> D_G(p_min, r)
is strictly increasing on [p_min, 1] under (A1), so exhibiting rationals a < b
with D_G(p_min,a) < gamma/beta < D_G(p_min,b) rigorously encloses r_0 in (a,b).
Derived quantities are then evaluated on the ball containing [a,b], so their
enclosures inherit rigour.  D_G is invariant under adding an affine function to
G, so a generator specified by its curvature needs no integration constants:
the engineered-spike generator is integrated in closed form through erf, which
is an independent route to the manuscript's trapezoidal-quadrature recipe.

Not claimed here: the chord supremum Lambda_c over a continuum of anchors is
certified only as a rigorous LOWER bound (the chord at the printed anchor) plus
the sufficient condition of cor:lambda-star, itself verified rigorously by
interval evaluation over a subdivision; and the mollified-kink grid numerals
are float-level only (cert_kink_attainment.py).
"""
from flint import arb, ctx

ctx.prec = 300
SQRT_PI = arb.pi().sqrt()


def encl(lo, hi):
    """Ball containing [lo, hi]."""
    lo, hi = arb(lo), arb(hi)
    return (lo + hi) / 2 + arb(0, ((hi - lo) / 2).upper())


def show(x, digits=10):
    return x.str(digits, radius=True)


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        raise SystemExit(f"RIGOROUS CERTIFICATE FAILED: {label}  {detail}")


# ----------------------------------------------------------------- generators
class G:
    def __init__(self, name, G, G1, G2, G3=None):
        self.name, self._G, self._G1, self._G2, self._G3 = name, G, G1, G2, G3

    def D(self, p, r):                       # defining formula
        return self._G(p) - self._G(r) - self._G1(r) * (p - r)


brier = G("brier", lambda p: p ** 2, lambda p: 2 * p, lambda p: arb(2), lambda p: arb(0))
halfcubic = G("halfcubic", lambda p: p ** 2 + p ** 3 / 2, lambda p: 2 * p + 3 * p ** 2 / 2,
              lambda p: 2 + 3 * p, lambda p: arb(3))
power4 = G("power4", lambda p: p ** 4, lambda p: 4 * p ** 3, lambda p: 12 * p ** 2, lambda p: 24 * p)
logscore = G("log", lambda p: p * p.log() + (1 - p) * (1 - p).log(),
             lambda p: p.log() - (1 - p).log(), lambda p: 1 / (p * (1 - p)),
             lambda p: (2 * p - 1) / (p * (1 - p)) ** 2)

# engineered spike: G'' = 2 + 100 exp(-((r-0.45)/0.02)^2), integrated through erf.
C0, S0, AMP = arb(45) / 100, arb(2) / 100, arb(100)
def _sp2(r):
    return 2 + AMP * (-(((r - C0) / S0) ** 2)).exp()
def _sp1(r):                                  # antiderivative of G'' (constant free)
    x = (r - C0) / S0
    return 2 * r + AMP * S0 * SQRT_PI / 2 * x.erf()
def _sp0(r):                                  # antiderivative of G'
    x = (r - C0) / S0
    return r ** 2 + AMP * S0 ** 2 * SQRT_PI / 2 * (x * x.erf() + (-(x ** 2)).exp() / SQRT_PI)
spike = G("spike", _sp0, _sp1, _sp2)


# ------------------------------------------------------------------- reserve
def reserve_enclosure(gen, p_min, ratio, a, b, refine=80):
    """Rigorously enclose r_0 by a verified sign change, then refine by bisection.

    Each bisection step keeps the bracket valid: the midpoint replaces whichever
    endpoint preserves D(p_min, lo) < ratio < D(p_min, hi), and a step is taken
    only when the comparison is decided rigorously at the working precision.
    """
    p_min, ratio, lo, hi = arb(p_min), arb(ratio), arb(a), arb(b)
    check(f"{gen.name}: sign change brackets r_0 in ({lo.str(8)}, {hi.str(8)})",
          bool(gen.D(p_min, lo) < ratio) and bool(gen.D(p_min, hi) > ratio))
    for _ in range(refine):
        mid = arb(((lo + hi) / 2).mid())
        d = gen.D(p_min, mid)
        if bool(d < ratio):
            lo = mid
        elif bool(d > ratio):
            hi = mid
        else:
            break
    return encl(lo.lower(), hi.upper())


print("== rigorous reserve enclosures ==")
r0_brier = reserve_enclosure(brier, "0.4", "0.09", "0.69999999", "0.70000001")
check("brier: r_0 = 0.7", bool((r0_brier - arb(7) / 10).abs_upper() < arb(1) / 10 ** 7), show(r0_brier))

r0_hc = reserve_enclosure(halfcubic, "0.4", "0.09", "0.622235", "0.622245")
check("halfcubic: r_0 = 0.62224 (printed 5 d.p.)",
      bool((r0_hc - arb(62224) / 100000).abs_upper() < arb(1) / 10 ** 5), show(r0_hc))

r0_log = reserve_enclosure(logscore, "0.6", "0.04", "0.7306825", "0.7306835")
check("log: r_0 = 0.730683 (printed 6 d.p.)",
      bool((r0_log - arb(730683) / 10 ** 6).abs_upper() < arb(1) / 10 ** 6), show(r0_log))

r0_spike = reserve_enclosure(spike, "0.4", "0.2", "0.550835", "0.550845")
check("spike: r_0 = 0.55084 (printed 5 d.p.; erf route vs the paper's quadrature)",
      bool((r0_spike - arb(55084) / 100000).abs_upper() < arb(1) / 10 ** 5), show(r0_spike))

print("== rigorous tangency slopes Lambda* = G''(r_0)(r_0 - p_min)/ratio ==")
def lam_star(gen, r0, p_min, ratio):
    return gen._G2(r0) * (r0 - arb(p_min)) / arb(ratio)

ls_log = lam_star(logscore, r0_log, "0.6", "0.04")
check("log: Lambda* = 16.602 (printed 5 s.f.)",
      bool((ls_log - arb(16602) / 1000).abs_upper() < arb(1) / 1000), show(ls_log))
ls_spike = lam_star(spike, r0_spike, "0.4", "0.2")
check("spike: Lambda* = 1.5084 (printed 5 s.f.)",
      bool((ls_spike - arb(15084) / 10000).abs_upper() < arb(1) / 10000), show(ls_spike))
ls_brier = lam_star(brier, r0_brier, "0.4", "0.09")
check("brier: Lambda* = 2/m", bool((ls_brier - 2 / (r0_brier - arb(4) / 10)).abs_upper() < arb(1) / 10 ** 6), show(ls_brier))

print("== rigorous chord at the printed anchor (lower bound on Lambda_c) ==")
def ramp_at(gen, p_min, r0, ratio, a):
    return gen.D(arb(p_min), a) / arb(ratio)

def chord_ball(a_ball):
    return (1 - ramp_at(spike, "0.4", r0_spike, "0.2", a_ball)) / (r0_spike - a_ball)

# Locate the maximizing anchor by ternary search on ball midpoints (a search, not a
# claim), then certify the chord AT that anchor: any chord is a rigorous lower bound
# on the supremum Lambda_c, so this certifies Lambda_c >= the printed value.
lo_a, hi_a = arb(4) / 10 + arb(1) / 10 ** 6, arb(r0_spike.mid()) - arb(1) / 10 ** 6
for _ in range(200):
    m1 = arb((lo_a + (hi_a - lo_a) / 3).mid())
    m2 = arb((hi_a - (hi_a - lo_a) / 3).mid())
    if float(chord_ball(m1).mid()) < float(chord_ball(m2).mid()):
        lo_a = m1
    else:
        hi_a = m2
a_hat = arb(((lo_a + hi_a) / 2).mid())
a_star = encl((a_hat - arb(1) / 10 ** 12).lower(), (a_hat + arb(1) / 10 ** 12).upper())
chord = chord_ball(a_star)
check("spike: located anchor matches the printed a* = 0.4326 (4 d.p.)",
      bool((a_star - arb(4326) / 10000).abs_upper() < arb(1) / 10000), show(a_star))
check("spike: chord at a* = 7.9902 (printed Lambda_c, 5 s.f.); rigorous LOWER bound on Lambda_c",
      bool((chord - arb(79902) / 10000).abs_upper() < arb(1) / 10000), show(chord))
ratio_53 = chord / ls_spike
check("spike: Lambda_c/Lambda* = 5.30", bool((ratio_53 - arb(530) / 100).abs_upper() < arb(5) / 1000), show(ratio_53))

print("== rigorous verification of cor:lambda-star's sufficient condition ==")
def s_prime_positive(gen, p_min, lo, hi, n=400):
    """s'(r) prop G'''(r)(r-p_min) + G''(r) > 0 on [lo,hi], by interval subdivision."""
    lo, hi = arb(lo), arb(hi)
    step = (hi - lo) / n
    for i in range(n):
        blk = encl((lo + step * i).lower(), (lo + step * (i + 1)).upper())
        val = gen._G3(blk) * (blk - arb(p_min)) + gen._G2(blk)
        if not bool(val > 0):
            return False
    return True

for gen, p_min, r0 in [(brier, "0.4", r0_brier), (halfcubic, "0.4", r0_hc),
                       (logscore, "0.6", r0_log), (power4, "0.35", None)]:
    hi = "0.99" if r0 is None else r0.upper()
    check(f"{gen.name}: s non-decreasing on [p_min, r_0] => Lambda_c = Lambda*",
          s_prime_positive(gen, p_min, p_min, hi))

print("== rigorous welfare cost of the Brier-formula threshold (halfcubic instance) ==")
# The Brier formula prescribes r = p_min + sqrt(gamma/beta) = 0.7, but the true reserve
# is 0.62224.  Types in [p_min, p*) then decline to climb and are falsely rejected,
# where D_G(p*, 0.7) = gamma/beta.  Loss = A f (p* - p_min)^2 / 2 with A = 1, f = 1.
R_WRONG, RATIO = arb(7) / 10, arb(9) / 100
lo, hi = arb("0.484"), arb("0.4852")
check("halfcubic: p* bracketed by a verified sign change",
      bool(halfcubic.D(lo, R_WRONG) > RATIO) and bool(halfcubic.D(hi, R_WRONG) < RATIO))
# p |-> D_G(p, 0.7) is strictly decreasing below 0.7 (dD/dp = G'(p) - G'(r) < 0),
# so the bracket refines by bisection with the comparison reversed.
for _ in range(80):
    mid = arb(((lo + hi) / 2).mid())
    d = halfcubic.D(mid, R_WRONG)
    if bool(d > RATIO):
        lo = mid
    elif bool(d < RATIO):
        hi = mid
    else:
        break
p_star = encl(lo.lower(), hi.upper())
loss = (p_star - arb(4) / 10) ** 2 / 2
check("halfcubic: wrong-threshold welfare cost = 3.59e-3 (printed 3 s.f.)",
      bool((loss - arb(359) / 100000).abs_upper() < arb(5) / 10 ** 6), show(loss))

print("== rigorous slope premiums at p_min = 0.4, r_0 = 0.7 ==")
# <kappa>_chi = (2/m^2) int_{p_min}^{r_0} kappa(t)(t - p_min) dt, analytic per generator.
p_min_a, r0_a = arb(4) / 10, arb(7) / 10
m = r0_a - p_min_a
def premium(kappa_r0, integral):
    return kappa_r0 / (2 / m ** 2 * integral)

int_brier = m ** 2                                                   # int 2(t-a) dt
int_p4 = 12 * ((r0_a ** 4 - p_min_a ** 4) / 4 - p_min_a * (r0_a ** 3 - p_min_a ** 3) / 3)
int_log = -p_min_a * (r0_a.log() - p_min_a.log()) - (1 - p_min_a) * ((1 - r0_a).log() - (1 - p_min_a).log())
r_b = premium(arb(2), int_brier)
r_l = premium(1 / (r0_a * (1 - r0_a)), int_log)
r_p4 = premium(12 * r0_a ** 2, int_p4)
check("brier curvature ratio = 1", bool((r_b - 1).abs_upper() < arb(1) / 10 ** 10), show(r_b))
check("log premium = 11.6% (printed 3 s.f.)", bool(((r_l / r_b - 1) * 100 - arb(116) / 10).abs_upper() < arb(5) / 100), show((r_l / r_b - 1) * 100))
check("p^4 premium = 34.2% (printed 3 s.f.)", bool(((r_p4 / r_b - 1) * 100 - arb(342) / 10).abs_upper() < arb(5) / 100), show((r_p4 / r_b - 1) * 100))

print("ALL PASS: rigorous enclosures")
