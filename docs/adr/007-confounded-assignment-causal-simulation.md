# ADR-007 — Confounded-Assignment Causal Simulation Design

**Status:** Accepted

## Context

The propensity/uplift work in `propensity.py` already demonstrates a T-Learner
and a Qini curve, but its treatment flag (`synthetic_attribution.exposed`) is
assigned uniformly at random, independent of every covariate — so the true
treatment effect is zero for every customer by construction (ADR-002). That
is the right design for demonstrating uplift machinery end-to-end, but it
cannot demonstrate observational causal inference: PSM, IPW, DiD, and Double
ML all exist to correct for confounded (non-random) treatment assignment, and
there is nothing to correct for when the true effect is uniformly zero.

Online Retail II has no experiment or campaign-launch data with a real,
known causal effect to validate an estimator against. Reusing the existing
`exposed`/`target_conversion` pair for this purpose was considered and
rejected: it would require overloading one column with two incompatible
ground truths (zero effect for the uplift demo, non-zero heterogeneous effect
for this one), which is a correctness bug, not a design choice.

## Decision

`causal_simulation.py` builds its own, fully synthetic potential-outcomes
model, independent of `attribution.py` and `propensity.py`:

- `p0(x) = sigmoid(intercept + recency_coef·recency_z + frequency_coef·frequency_z + monetary_coef·monetary_z)`
  — the untreated conversion probability, a function of standardized RFM
  features only (reusing `segmentation.build_rfm_matrix`). It never reads the
  real `target_conversion` column, so the ground truth stays closed-form and
  exactly known.
- `tau(x) = base_ate + segment_uplift[segment_name]` — a treatment effect
  heterogeneous by RFM segment, mirroring the diminishing-returns logic
  already used by the Hill response curves in `simulator.py`: near-ceiling
  segments (Champions) get less incremental lift, segments with more room to
  move (At Risk, Hibernating) get more.
- `p1(x) = clip(p0(x) + tau(x), 0, 1)`.

This one model is then *observed* under two different assignment mechanisms,
both parameterized in `configs/causal.yaml`:

- **Layer 1 (random):** `T ~ Bernoulli(0.5)`, independent of `x` — a
  randomized controlled trial by construction. A naive treated-vs-control
  comparison is unbiased here.
- **Layer 2 (confounded):** `P(T=1|x) = sigmoid(confound_intercept +
  confounding_strength · monetary_z)` — higher-value customers are
  preferentially targeted, the most common and realistic selection bias in
  marketing campaign assignment. `confounding_strength` is a tunable knob:
  increasing it makes the naive estimator diverge further from the truth
  while PSM/IPW/DiD/DML should stay close, which is the demonstration this
  whole extension exists to make.

Because both layers share the exact same `p0`/`p1`/`tau`, the true ATE and
the true CATE-by-segment are identical and known in both — only Layer 2
requires correction. That symmetry (not an accident of shared random seeds)
is what makes recovery-of-truth in `causal_diagnostics.py` a meaningful test.

**Difference-in-Differences panel.** `build_did_panel` constructs a
customer × month panel around a synthetic campaign-launch date. Its outcome
(`did_outcome`) is also fully synthetic: a linear time trend shared by both
arms pre-launch (parallel trends by construction) plus the same `tau(x)`
applied as a post-launch bump for treated customers only. `fact_transactions`
is used exclusively to attach each customer's *real* monthly transaction
activity as a descriptive covariate for the dashboard — never as the
effect-bearing outcome, because Online Retail II has a large, real Nov/Dec
seasonality spike that would otherwise contaminate the parallel-trends check
with a genuine trend break unrelated to the synthetic effect. The launch date
(`did.launch_date = "2011-05-15"`) is deliberately chosen mid-year to avoid
that window. `did.violate_parallel_trends` injects an extra pre-period slope
for the treated arm only, so the parallel-trends diagnostic in
`causal_estimators.difference_in_differences` has a real violation to catch
in its dedicated test — the assumption is never just assumed to hold.

Every artifact this module produces carries the "Synthetic" label wherever it
surfaces on the dashboard, the same convention established by ADR-002.

## Consequences

- Every causal estimate produced downstream is a recovery-of-truth
  demonstration against a known synthetic effect, not a real causal finding
  about Online Retail II customers — the same honesty boundary ADR-002
  already draws for the media-attribution layer.
- `configs/causal.yaml` is the single place that controls effect size,
  heterogeneity, and confounding strength; sweeping `confounding_strength`
  and re-running the estimator comparison is how the degradation of the naive
  estimator (and the relative robustness of PSM/IPW/DML) gets shown, without
  touching any code.
- `attribution.py`, `propensity.py`, and the existing T-Learner/Qini uplift
  demo are untouched — this module is strictly additive and does not
  reinterpret or reuse their outputs.
- If a future revision wants the DiD panel's real-activity covariate to
  actually influence `did_outcome` (for more realism), that is a deliberate
  design change requiring a new ADR — not an implicit side effect of touching
  `fact_transactions`.
