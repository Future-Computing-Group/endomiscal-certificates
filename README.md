# Numerical certificates for *The Endogeneity of Miscalibration*

Independent machine checks of the computed claims in *The Endogeneity of Miscalibration: Impossibility and Escape in Scored Reporting* (Lovén and Tarkoma). Every numeral the paper prints is recomputed here and asserted against the printed value, and the oversight-design results of Section 4 are exhibited on instances; the marketplace section and the multi-dimensional appendix, which print no numerals, have no code here.

## Independence discipline

A certificate that re-evaluates the derivation it is meant to check inherits that derivation's mistakes and confirms them. So no measured quantity here is computed from a closed form the paper derives; the constants of the paper's proved bounds are evaluated from their defining ingredients and then tested against measured gaps. Instead:

- the Bregman divergence is assembled from its defining formula, `D_G(p,r) = G(p) - G(r) - G'(r)(p - r)`;
- generators given by their curvature are integrated up (by quadrature in the floating-point certificates, in closed form through `erf` in the rigorous one), and `D_G` is invariant under adding an affine function to `G`, so no integration constant is needed;
- the agent's report is a global argmax over a fine report grid, with the principal-favourable tie-break, rather than a first-order condition;
- welfare is integrated from the induced screening against the surplus weight;
- the reserve is found by bisection on `D_G`, the critical slope as a supremum of chord slopes, the tangency slope from a point value of the curvature.

The one type excluded everywhere is the break-even type, at which the agent is exactly indifferent by construction and the paper's attainment claims are stated for all other types.

## What is checked

| Script | Claims |
|---|---|
| `cert_tilt_identity.py` | The tilt identity, on five generators; the curvature ratio equals 1 for Brier; the printed slope premiums of the log and quartic scores; the log score's premium against Brier takes either sign below the unit sum; the score-selection proposition: both cubic-law constants in closed form at fixed miscalibration on six generators, the convex-curvature inequality with equality only for Brier, the two printed shortfall ratios, the measured power-four-to-Brier loss ratio descending toward its limit, and two witnesses (a non-convex curvature step, a decreasing affine curvature) on which the inequality's hypotheses bite. |
| `cert_comparative_statics.py` | The break-even type's derivative in the failure payoff, sign included; a higher break-even type raises the reserve, on every generator tested; the Brier invariances. |
| `cert_kink_attainment.py` | The Bregman ramp and the step rule at the reserve attain exact first-best screening for every generator tested, including non-constant curvature and the log score, pointwise on the type grid away from the indifference type; on the paper's own mollified family the under-approval channel is first order in the smoothing width and the false-approval channel second order, on Brier and on a non-Brier generator; the quadratic corner rounding is kept as a second family under the cap; the weak-feasibility boundary (the ramp at a reserve of one) and the saturated regime (the reduced-intensity rule) are exhibited on Brier. |
| `cert_critical_slope_iff.py` | Both directions of the critical-slope threshold on named generators, on twenty seeded random admissible generators (all of which land in the tangency class) and on an engineered family of curvature spikes inside the reserve interval (the corner class): a truncated line just above the critical slope attains first-best and satisfies the cap and pinch of the theorem, while just below it neither the truncated-line family nor the largest Lipschitz rule under the ramp does; the tangent-support criterion for the critical slope to equal the tangency slope is asserted in both directions on every generator. The lower half is a no-counterexample-found search over rule families, not a proof. |
| `cert_cubic_law.py` | The cubic law below the critical slope: for Brier under a uniform type distribution the measured gap matches the exact closed form, not merely its leading term, and under two non-uniform densities the gap scales by the density at the marginal type, as the printed constant says; for other generators the log-log exponent is three and the leading constant is approached; the lower-bound constant is recomputed from its ingredients, giving the printed ratio between the two constants; and no rule in a search family beats the lower bound. |
| `cert_printed_recipes.py` | The grid-computed illustrations, run exactly as the paper prints their recipes: the mollified kink at the Brier instance with the standard bump transition (falsely approved width 5.5e-3 on the 4001-point type grid, exact boundary 5.61e-3, minimum approval 0.9862 above the marginal type, the under-approved band the whole interval up to the reserve), the engineered-spike footnote (chord anchor 0.4326; the line at 1.02 times the tangency slope pays 0.1525 at a zero report, mis-screens 0.46 of the 999-type mass with pointwise error 0.80; the line at 1.0005 times the critical slope screens exactly), and the wrong-threshold welfare cost 3.59e-3 measured from the induced screening rather than from the paper's reduction. |
| `cert_intervals.py` | Rigorous enclosures, in Arb ball arithmetic at 300-bit precision, of every printed constant: the reserves, the tangency slopes, the critical slope (enclosed on BOTH sides, see below), the ratio between critical and tangency slope, the welfare cost of using the Brier threshold formula under a non-quadratic generator, the slope premiums, and the two shortfall-constant ratios of the score-selection proposition. A comparison in ball arithmetic is true only when it holds for every point of every ball, so these are proofs of the enclosures rather than agreements between floating-point runs; the branch-and-bound asserts that it never truncated its live set. |

## Rigour, stated honestly

`cert_intervals.py` is rigorous: enclosures are certified by verified sign changes and interval evaluation, including the paper's sufficient condition for the critical slope to equal the tangency slope, verified by subdivision. The critical slope is a supremum over a continuum of chord anchors and is enclosed on both sides: below by an attained chord, above by branch and bound over anchor boxes. Two details make the upper bound work. On a box both the numerator and the distance to the reserve are decreasing in the anchor, so the quotient is bounded using thin endpoints; feeding a ball-valued anchor into the quotient instead lets its dependency compound and the bound goes slack by three orders of magnitude. Near the reserve the quotient is 0/0, but there the chord is the mean of the ramp slope over the remaining interval, hence at most its supremum, which is enclosed directly.

One thing it does not claim: the grid-defined numerals of the mollified-kink instance and of the engineered-spike footnote are covered at floating-point level only, by `cert_printed_recipes.py`.

The other six scripts are floating-point instance and property checks. They can exhibit a failure and cannot establish a theorem; the theorems are proved in the paper.

## Running

```
pip install numpy python-flint      # python-flint is needed only by cert_intervals.py
python3 run_all.py
```

Each script prints one line per assertion and exits non-zero on the first failure. The full suite takes a few minutes, dominated by the randomized threshold search.

## Citation

Archived release: [10.5281/zenodo.22983104](https://doi.org/10.5281/zenodo.22983104) (concept DOI [10.5281/zenodo.22129804](https://doi.org/10.5281/zenodo.22129804), always resolving to the latest version).

## Licence

MIT, see `LICENSE`.
