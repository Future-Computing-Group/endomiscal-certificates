#!/usr/bin/env python3
"""Figure 1 of the paper: the approval rules of Section 4 at the Brier instance, and the reports they induce.

Instance (the one the paper's worked numbers use): Brier score G(p) = p^2, p_min = 0.4, gamma/beta = 0.09, so
C(r) = (r - 0.4)^2, r_0 = 0.7, m = 0.3 and Lambda_c = the ramp slope at the reserve = 2/m. Every curve is computed from the defining
objects in common.py (reserve by bisection, ramp, truncated line, the Lipschitz envelope under the ramp, the mollified
ramp of Theorem 4.8(ii) with the standard bump, and the agent's global argmax on the report grid).

Usage: fig_rules.py OUT.pdf [OUT.png]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import brier, reserve, ramp, truncated_line, lipschitz_envelope_under_ramp, RGRID

P_MIN, RATIO = 0.4, 0.09
gen = brier()
r0 = reserve(gen, P_MIN, RATIO)
m = r0 - P_MIN
lam_c = gen.G2(r0) * m / RATIO                       # tangency slope = critical slope for Brier (supplement S.9)
r = RGRID


def bump(x):
    x = np.clip(x, 0.0, 1.0)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        a = np.exp(-1.0 / np.maximum(x, 1e-300))
        b = np.exp(-1.0 / np.maximum(1.0 - x, 1e-300))
    return np.where(x <= 0, 0.0, np.where(x >= 1, 1.0, a / (a + b)))


def argmax_report(q_vals, types):
    """r*(p): the agent's global argmax of q(r) - D(p, r)/ratio over the report grid, largest-q tie-break."""
    out = np.empty(len(types))
    gR, g1R = gen.G(r), gen.G1(r)
    for i, p in enumerate(types):
        D = gen.G(p) - gR - g1R * (p - r)
        U = q_vals - D / RATIO
        mx = U.max()
        cand = np.where(U >= mx - 1e-13)[0]
        out[i] = r[cand[np.argmax(q_vals[cand])]]
    return out


cap = np.where(r <= P_MIN, 0.0, gen.D(P_MIN, r) / RATIO)          # (beta/gamma) C(r), uncapped
q_ramp = ramp(gen, P_MIN, r0, RATIO, r)
q_step = (r >= r0).astype(float)
q_line = truncated_line(r0, lam_c, r)
eps = 0.08
q_moll = q_ramp + (1.0 - q_ramp) * bump((r - r0 + eps) / eps)
lam_low = 0.6 * lam_c
_, q_env = lipschitz_envelope_under_ramp(gen, P_MIN, r0, RATIO, lam_low, rg=r)

types = np.linspace(0.0, 1.0, 801)
types = types[np.abs(types - P_MIN) > 1e-9]
rs_step = argmax_report(q_step, types)
rs_env = argmax_report(q_env, types)

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.6, 2.05))
for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(0, 1)

mask = cap <= 1.12
ax1.plot(r[mask], cap[mask], color="0.75", lw=1.0, ls=":", label=r"cap $(\beta/\gamma)C$")
ax1.plot(r, q_step, color="black", lw=1.2, label="step at $r_0$")
ax1.plot(r, q_ramp, color="tab:blue", lw=1.4, label="ramp")
ax1.plot(r, q_line, color="tab:orange", lw=1.2, ls="--", label=r"line, slope $\Lambda_c$")
ax1.plot(r, q_moll, color="tab:green", lw=1.0, label=r"mollified ($\varepsilon=0.08$)")
ax1.plot(r, q_env, color="tab:red", lw=1.0, ls="-.", label=r"under cap, $0.6\,\Lambda_c$")
ax1.set_ylim(0, 1.12)
ax1.set_xlabel("report $r$")
ax1.set_ylabel("approval probability $q(r)$")
ax1.set_xticks([0, P_MIN, r0, 1]); ax1.set_xticklabels(["0", r"$p_{\min}$", "$r_0$", "1"])
ax1.set_yticks([0, 1])
ax1.legend(frameon=False, fontsize=6.8, loc="upper left", handlelength=1.5, borderaxespad=0.2, labelspacing=0.35)
ax1.set_title("(a) approval rules", fontsize=8.5, loc="left")

ax2.plot(types, types, color="0.75", lw=0.8, ls=":", label="truthful report")
ax2.plot(types, rs_step, color="black", lw=1.2, label="step rule")
ax2.plot(types, rs_env, color="tab:red", lw=1.0, ls="-.", label=r"slope-$0.6\,\Lambda_c$ rule")
ax2.set_ylim(0, 1)
ax2.set_xlabel("type $p$")
ax2.set_ylabel("induced report $r^*(p)$")
ax2.set_xticks([0, P_MIN, r0, 1]); ax2.set_xticklabels(["0", r"$p_{\min}$", "$r_0$", "1"])
ax2.set_yticks([0, P_MIN, r0, 1]); ax2.set_yticklabels(["0", r"$p_{\min}$", "$r_0$", "1"])
ax2.legend(frameon=False, fontsize=6.8, loc="lower right", bbox_to_anchor=(1.0, 0.0), handlelength=1.3, borderaxespad=0.3)
ax2.set_title("(b) induced reports", fontsize=8.5, loc="left")

fig.tight_layout()
for out in sys.argv[1:]:
    fig.savefig(out, dpi=200 if out.endswith(".png") else None)
print(f"r0={r0:.6f} m={m:.6f} Lambda_c={lam_c:.6f}; pooled band under the step rule: {types[(rs_step > types + 1e-9)].min():.4f}..{types[(rs_step > types + 1e-9)].max():.4f}")
