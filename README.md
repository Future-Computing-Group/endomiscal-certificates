# Numerical certificates for *The Endogeneity of Miscalibration*

Independent machine checks of the computed claims in *The Endogeneity of Miscalibration: Impossibility and Escape in Scored Reporting* (Lovén and Tarkoma). Every numeral the paper prints, and every structural claim that can be exhibited on instances, is recomputed here and asserted against the printed value.

## Independence discipline

A certificate that re-evaluates the derivation it is meant to check inherits that derivation's mistakes and confirms them. So nothing here uses a closed form the paper derives. Instead:

- the Bregman divergence is assembled from its defining formula, `D_G(p,r) = G(p) - G(r) - G'(r)(p - r)`;
- generators given by their curvature are integrated up (by quadrature in the floating-point certificates, in closed form through `erf` in the rigorous one), and `D_G` is invariant under adding an affine function to `G`, so no integration constant is needed;
- the agent's report is a global argmax over a fine report grid, with the principal-favourable tie-break, rather than a first-order condition;
- welfare is integrated from the induced screening against the surplus weight;
- the reserve is found by bisection on `D_G`, the critical slope as a supremum of chord slopes, the tangency slope from a point value of the curvature.

The one type excluded everywhere is the break-even type, at which the agent is exactly indifferent by construction and the paper's attainment claims are stated for all other types.

## What is checked

| Script | Claims |
|---|---|
| `cert_tilt_identity.py` | The tilt identity, on five generators; the curvature ratio equals 1 for Brier; the printed slope premiums of the log and quartic scores. |
| `cert_comparative_statics.py` | The break-even type's derivative in the failure payoff, sign included; a higher break-even type raises the reserve, on every generator tested; the Brier invariances. |
| `cert_kink_attainment.py` | The Bregman ramp attains exact first-best screening for every generator tested, including non-constant curvature and the log score; a continuously differentiable family under the cap has welfare gap linear in the smoothing width, with the false-approval channel of higher order. |
| `cert_critical_slope_iff.py` | Both directions of the critical-slope threshold on named generators and twenty seeded random admissible generators: a truncated line just above the critical slope attains first-best, while just below it neither the truncated-line family nor the largest Lipschitz rule under the ramp does. The lower half is a no-counterexample-found search over rule families, not a proof. |
| `cert_cubic_law.py` | The cubic law below the critical slope: for Brier under a uniform type distribution the measured gap matches the exact closed form, not merely its leading term; for other generators the log-log exponent is three and the leading constant is approached; the lower-bound constant is recomputed from its ingredients, giving the printed ratio between the two constants; and no rule in a search family beats the lower bound. |
| `cert_intervals.py` | Rigorous enclosures, in Arb ball arithmetic at 300-bit precision, of every printed numeral: the reserves, the tangency slopes, the critical slope (enclosed on BOTH sides, see below), the ratio between critical and tangency slope, the welfare cost of using the Brier threshold formula under a non-quadratic generator, and the slope premiums. A comparison in ball arithmetic is true only when it holds for every point of every ball, so these are proofs of the enclosures rather than agreements between floating-point runs. |

## Rigour, stated honestly

`cert_intervals.py` is rigorous: enclosures are certified by verified sign changes and interval evaluation, including the paper's sufficient condition for the critical slope to equal the tangency slope, verified by subdivision. The critical slope is a supremum over a continuum of chord anchors and is enclosed on both sides: below by an attained chord, above by branch and bound over anchor boxes. Two details make the upper bound work. On a box both the numerator and the distance to the reserve are decreasing in the anchor, so the quotient is bounded using thin endpoints; feeding a ball-valued anchor into the quotient instead lets its dependency compound and the bound goes slack by three orders of magnitude. Near the reserve the quotient is 0/0, but there the chord is the mean of the ramp slope over the remaining interval, hence at most its supremum, which is enclosed directly.

One thing it does not claim: the grid-defined numerals of the mollified-kink instance are covered at floating-point level only.

The other five scripts are floating-point instance and property checks. They can exhibit a failure and cannot establish a theorem; the theorems are proved in the paper.

## Running

```
pip install numpy python-flint      # python-flint is needed only by cert_intervals.py
python3 run_all.py
```

Each script prints one line per assertion and exits non-zero on the first failure. The full suite takes a few minutes, dominated by the randomized threshold search.

## Citation

Archived release: [10.5281/zenodo.22143871](https://doi.org/10.5281/zenodo.22143871) (concept DOI [10.5281/zenodo.22129804](https://doi.org/10.5281/zenodo.22129804), always resolving to the latest version).

## Licence

MIT, see `LICENSE`.
