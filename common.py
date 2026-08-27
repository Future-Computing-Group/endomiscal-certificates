"""Shared machinery for the EndoMiscal numeric certificates.

Independence discipline: every certificate recomputes DEFINING objects, never
the paper's derived identities.  The Bregman divergence is assembled from its
defining formula D_G(p,r) = G(p) - G(r) - G'(r)(p-r); generators specified by
their curvature G'' are integrated up by cumulative quadrature (the paper's own
reproduction recipe); the agent's report is a global argmax over a fine grid
with the principal-favourable tie-break (largest approval among maximizers);
welfare is integrated from the induced screening against the surplus weight.
The reserve is found by bisection on D_G, the critical slope as the supremum of
chord slopes, the tangency slope from point values of G''.  Nothing below uses
a closed form the manuscript derives.

Conventions: beta = 1 throughout; `ratio` = gamma/beta; surplus Pi(p) =
A (p - p_min) with A = 1; F uniform on [0,1] unless a density is passed.
The type p = p_min is F-null and indifferent by design; screening claims are
asserted for p != p_min, as the manuscript states them.
"""
import numpy as np

N_R = 200_001                      # report grid (matches the manuscript's recipe)
RGRID = np.linspace(0.0, 1.0, N_R)
H = RGRID[1] - RGRID[0]


# ---------------------------------------------------------------- generators
class Gen:
    """A generator with vectorized G, G1 (=G'), G2 (=G'') and a Bregman D."""

    def __init__(self, name, G, G1, G2, interior=False):
        self.name, self.G, self.G1, self.G2 = name, G, G1, G2
        self.interior = interior   # True: D infinite/undefined at endpoints (A1 open-interval class)

    def D(self, p, r):
        """Defining formula.  p scalar or array, r scalar or array."""
        return self.G(p) - self.G(r) - self.G1(r) * (p - r)


def brier():
    return Gen("brier", lambda p: p ** 2, lambda p: 2 * p, lambda p: 2.0 * np.ones_like(np.asarray(p, dtype=float)))


def halfcubic():
    # G = p^2 + p^3/2  (thm:reserve(ii) instance)
    return Gen("halfcubic", lambda p: p ** 2 + 0.5 * p ** 3, lambda p: 2 * p + 1.5 * p ** 2, lambda p: 2 + 3 * np.asarray(p, dtype=float))


def quartic_mix():
    # G = p^2 + p^4/12, curvature 2 + p^2 in [2,3]: non-constant, full-hypothesis
    return Gen("quartic_mix", lambda p: p ** 2 + p ** 4 / 12.0, lambda p: 2 * p + p ** 3 / 3.0, lambda p: 2 + np.asarray(p, dtype=float) ** 2)


def power(k):
    return Gen(f"power{k}", lambda p, k=k: p ** k, lambda p, k=k: k * p ** (k - 1), lambda p, k=k: k * (k - 1) * np.asarray(p, dtype=float) ** (k - 2))


def logscore():
    def G(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
        return p * np.log(p) + (1 - p) * np.log(1 - p)
    def G1(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
        return np.log(p) - np.log(1 - p)
    def G2(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
        return 1.0 / (p * (1 - p))
    return Gen("log", G, G1, G2, interior=True)


def from_curvature(name, gpp):
    """Generator from a curvature callable, by cumulative trapezoid (defining route)."""
    g2 = gpp(RGRID)
    assert np.all(g2 > 0), "curvature must be positive"
    g1 = np.concatenate([[0.0], np.cumsum((g2[1:] + g2[:-1]) * 0.5 * H)])
    g0 = np.concatenate([[0.0], np.cumsum((g1[1:] + g1[:-1]) * 0.5 * H)])
    return Gen(name,
               lambda p, y=g0: np.interp(p, RGRID, y),
               lambda p, y=g1: np.interp(p, RGRID, y),
               lambda p, y=g2: np.interp(p, RGRID, y))


def engineered_spike():
    # cor:lambda-star footnote generator: G'' = 2 + 100 exp(-((r-0.45)/0.02)^2)
    return from_curvature("spike", lambda r: 2 + 100 * np.exp(-(((np.asarray(r, dtype=float)) - 0.45) / 0.02) ** 2))


def random_gen(rng, i):
    """Random admissible generator: positive mixture-of-bumps curvature."""
    c0 = rng.uniform(0.5, 3.0)
    nb = rng.integers(1, 4)
    mus = rng.uniform(0.1, 0.9, nb)
    sigmas = rng.uniform(0.05, 0.3, nb)
    amps = rng.uniform(0.0, 15.0, nb)
    def gpp(r):
        r = np.asarray(r, dtype=float)
        out = c0 * np.ones_like(r)
        for m, s, a in zip(mus, sigmas, amps):
            out = out + a * np.exp(-(((r - m) / s) ** 2))
        return out
    return from_curvature(f"rand{i}", gpp)


# ------------------------------------------------------------------- objects
def reserve(gen, p_min, ratio):
    """r_0 solving D(p_min, r_0) = ratio, by bisection on the defining D."""
    if gen.D(p_min, 1.0 - (1e-12 if gen.interior else 0.0)) <= ratio:
        return None                                   # not strictly feasible
    lo, hi = p_min, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if gen.D(p_min, mid) < ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ramp(gen, p_min, r0, ratio, r):
    r = np.asarray(r, dtype=float)
    return np.where(r <= p_min, 0.0, np.clip(gen.D(p_min, np.minimum(r, r0)) / ratio, 0.0, 1.0))


def lambda_c(gen, p_min, r0, ratio, n_a=80_000):
    """Critical slope: sup over a in [p_min, r_0) of (1 - ramp(a)) / (r_0 - a)."""
    a = np.linspace(p_min, r0 - 1e-9, n_a)
    vals = (1.0 - ramp(gen, p_min, r0, ratio, a)) / (r0 - a)
    return float(np.max(vals)), float(a[int(np.argmax(vals))])


def lambda_star(gen, p_min, r0, ratio):
    """Tangency slope (beta/gamma) G''(r_0) (r_0 - p_min), from a point value of G''."""
    return float(gen.G2(r0) * (r0 - p_min) / ratio)


def s_prime_r0(gen, p_min, r0, ratio, h=1e-5):
    """s'(r_0) by central finite difference on s(r) = G''(r)(r - p_min)/ratio."""
    s = lambda r: gen.G2(r) * (r - p_min) / ratio
    return float((s(r0 + h) - s(r0 - h)) / (2 * h))


# ------------------------------------------------------- agent + welfare
def induced(gen, p_min, ratio, q_vals, types, rgrid=None, chunk=64):
    """Induced approval tau(p) per type: global argmax, largest-q tie-break.

    Chunked broadcasting: for a block of types P and the report grid R,
    D(P,R) = G(P)[:,None] - G(R)[None,:] - G'(R)[None,:] * (P[:,None] - R[None,:]),
    which is the defining formula evaluated blockwise rather than type by type.
    """
    rg = RGRID if rgrid is None else rgrid
    qv = np.asarray(q_vals, dtype=float)
    types = np.asarray(types, dtype=float)
    gR, g1R = np.asarray(gen.G(rg), dtype=float), np.asarray(gen.G1(rg), dtype=float)
    out = np.empty(len(types))
    for i in range(0, len(types), chunk):
        P = types[i:i + chunk]
        gP = np.asarray(gen.G(P), dtype=float)
        D = gP[:, None] - gR[None, :] - g1R[None, :] * (P[:, None] - rg[None, :])
        U = qv[None, :] - D / ratio
        U = np.where(np.isfinite(U), U, -np.inf)
        mx = U.max(axis=1, keepdims=True)
        out[i:i + chunk] = np.where(U >= mx - 1e-13, qv[None, :], -np.inf).max(axis=1)
    return out


def welfare_gap(gen, p_min, ratio, q_of_r, n_types=4001, rgrid=None, extra_r=None, types=None):
    """W* - W(q) with Pi = (p - p_min), F uniform: both error channels, positive."""
    rg = RGRID if rgrid is None else rgrid
    if extra_r is not None:
        rg = np.unique(np.concatenate([rg, extra_r]))
    types = np.linspace(0.0, 1.0, n_types) if types is None else np.asarray(types, dtype=float)
    if gen.interior:
        types = types[(types > 1e-3) & (types < 1 - 1e-3)]
    types = types[np.abs(types - p_min) > 1e-9]       # exclude the F-null type
    tau = induced(gen, p_min, ratio, q_of_r(rg), types, rgrid=rg)
    under = np.where(types > p_min, (1.0 - tau) * (types - p_min), 0.0)
    false = np.where(types < p_min, tau * (p_min - types), 0.0)
    return float(np.trapezoid(under + false, types)), float(np.trapezoid(under, types)), float(np.trapezoid(false, types))


def lipschitz_envelope_under_ramp(gen, p_min, r0, ratio, lam, rg=None):
    """hat q_Lambda: largest Lambda-Lipschitz function below min(ramp, 1), 0 below p_min.

    Two-pass inf-convolution on the grid (defining construction, no closed form).
    """
    rg = RGRID if rg is None else rg
    cap = np.minimum(ramp(gen, p_min, r0, ratio, rg), 1.0)
    cap = np.where(rg < p_min, 0.0, cap)
    h = np.diff(rg)
    q = cap.copy()
    for i in range(1, len(q)):                        # forward pass
        q[i] = min(q[i], q[i - 1] + lam * h[i - 1])
    for i in range(len(q) - 2, -1, -1):               # backward pass
        q[i] = min(q[i], q[i + 1] + lam * h[i])
    return rg, q


def truncated_line(r0, lam, r):
    """q_Lambda(r) = med{0, 1 - Lambda (r_0 - r), 1} (thm:critical-slope(iv))."""
    return np.clip(1.0 - lam * (r0 - np.asarray(r, dtype=float)), 0.0, 1.0)


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        raise SystemExit(f"CERTIFICATE FAILED: {label}  {detail}")
