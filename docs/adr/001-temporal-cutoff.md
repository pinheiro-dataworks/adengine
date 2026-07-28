# ADR-001 — Temporal Anti-Leakage Design

**Status:** Accepted

## Context

The propensity target is "will this customer buy again in the next 90 days?"
If features and target are computed over the same time window, the model
partially learns the future and offline metrics stop meaning anything —
this is the single most common way retail ML projects quietly lie to
themselves.

## Decision

Fix explicit cutoffs and never compute a feature from data past them:

- **T_train = 2011-03-31**, **T_test = 2011-06-30** — feature cutoffs for
  model training and temporal validation (see ADR-004).
- **Target window = 90 days** after the cutoff, e.g. for T_test the target
  is realized purchase activity in `(2011-06-30, 2011-09-28]`.
- The dataset runs through 2011-12-09, so there is a genuinely unobserved
  window beyond `T_test + 90d` — nothing downstream ever reads past it.

Every feature function in `features.py` takes an explicit `as_of:
pd.Timestamp` and filters with `invoice_date <= as_of` before it touches a
row. `tests/test_features.py::test_as_of_boundary_is_never_crossed` asserts
this directly: adding a post-`as_of` transaction to the fixture must not
change a single feature value, only the (separately computed) target.

Segmentation and the media-metrics "current state" views use `T_test` as
the same "as of today" snapshot — one cutoff for "what do we know," not a
second ad hoc one.

## Consequences

- Customers acquired after `T_train` are excluded from training — accepted
  loss of sample size in exchange for an honest estimate of production
  performance.
- Every new feature added to the project must accept `as_of` and prove it
  in a test before it ships — this is treated as a non-negotiable contract,
  not a style preference.
