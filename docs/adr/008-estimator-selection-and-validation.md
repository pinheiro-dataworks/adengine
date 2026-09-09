# ADR-008 — Estimator Selection and Validation Criteria

**Status:** Accepted

## Context

The extension needs to demonstrate observational causal inference —
correcting a confounded treatment assignment without a real experiment —
covering the methods a marketing analytics role is expected to know:
matching, weighting, panel-based identification, and a modern
machine-learning estimator. It also needs a validation criterion that is
more rigorous than "the estimate looks plausible."

`econml` (Microsoft's causal-ML library) was the natural first candidate for
the Double ML estimator. Checking its published PyPI metadata before writing
any code showed it pins `scikit-learn<1.10,>=1.6` (tightening this project's
current `scikit-learn>=1.5`) and pulls in `numba`, `shap`, `lightgbm`, and
`statsmodels` as hard dependencies — a large jump in install footprint and a
second unmaintained-dependency risk sitting right next to the one already
documented for `lifetimes` in the README's Known Limitations. `ortools` was
the equivalent candidate for the allocation optimizer (ADR-009) — same
question, different module.

## Decision

**Five estimators**, all returning a common `ATEEstimate`:

1. `naive_diff_in_means` — kept deliberately in the comparison, not deleted
   once something better exists. It is unbiased on Layer 1 (random
   assignment) and visibly biased on Layer 2 (confounded) — that contrast
   *is* the demonstration, not a strawman to be embarrassed by.
2. `propensity_score_matching` — 1:1 nearest-neighbor matching on a logistic
   propensity score (same model family as `propensity.py`'s baseline),
   evaluated by post-matching covariate balance (SMD).
3. `inverse_propensity_weighting` — stabilized, truncated Horvitz-Thompson/
   Hajek weighting.
4. `difference_in_differences` — closed-form 2×2 DiD on the customer × month
   panel from `causal_simulation.build_did_panel`, always paired with an
   explicit parallel-trends pre-period check.
5. `double_ml_ate` — cross-fitted partialling-out Double ML (Chernozhukov et
   al. 2018), **implemented directly on `sklearn`** instead of `econml`:
   K-fold cross-fitting, `HistGradientBoostingRegressor` for `E[Y|X]` and
   `HistGradientBoostingClassifier` for `E[T|X]` — the same model family
   already used throughout `propensity.py` — residuals computed strictly
   out-of-fold, and a closed-form asymptotic variance for the confidence
   interval. This is the published estimator formula, not an invented
   approximation; implementing it directly keeps the project's dependency
   list unchanged (zero new packages in `pyproject.toml` for this entire
   extension) and mirrors how the project already hand-rolls the T-Learner
   and Qini curve in `propensity.py` instead of depending on a specialized
   uplift library. `ortools` receives the same treatment in ADR-009.

**Validation criterion: recovery of truth.** Because `causal_simulation.py`
(ADR-007) provides a closed-form, known `true_ate`/`true_cate_by_segment`,
every estimator can be scored by absolute error against ground truth — a
materially stronger bar than "does the sign make sense." This mirrors the
acceptance-bar pattern already established for segmentation stability in
ADR-006 (ARI ≥ 0.85): a numeric threshold decided in advance, not eyeballed
after the fact. `causal_diagnostics.recovery_of_truth` reports every
estimator side by side, ranked by absolute error, **including whichever one
performs worst** — omitting a poor performer would misrepresent the
comparison, not flatter it.

Two supporting diagnostics complete the validation: `covariate_balance_table`
(SMD < 0.1 pre/post matching, the same convention used in this project's own
test suite) and `rosenbaum_sensitivity` (a binary-outcome, McNemar-type
Rosenbaum-bounds sweep via `scipy.stats.binomtest` — no new dependency),
which quantifies how large a hidden confounder would need to be before the
PSM conclusion could flip.

## Consequences

- The full comparison — naive vs. PSM vs. IPW vs. DiD vs. DML, against a
  known ground truth — runs on `pandas`, `numpy`, `scipy`, and `scikit-learn`
  only, all already declared in `pyproject.toml`. No dependency was added for
  this extension.
- `configs/causal.yaml`'s `confounding.confounding_strength` is the intended
  lever for showing *degradation*: increasing it should widen the gap
  between naive and the four correction methods. Sweeping it and re-running
  the comparison is a config change, not a code change.
- If a future need (e.g., heterogeneous CATE estimation with a forest-based
  learner, which genuinely needs `econml`'s `CausalForestDML` or similar) can't
  be reasonably hand-rolled, that is grounds for revisiting this decision —
  but plain ATE estimation, which is what this extension demonstrates, does
  not meet that bar.
- Every number produced here is validated against a synthetic ground truth,
  not a real causal claim about Online Retail II customers — the same
  boundary ADR-007 and ADR-002 already draw.
