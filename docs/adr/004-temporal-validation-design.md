# ADR-004 — Propensity Model Temporal Validation Design

**Status:** Accepted

## Context

A model needs an honest train/test split. Two candidate designs:

1. **Split by customer** — partition customers into train/test sets once,
   compute features for both at the same `as_of`.
2. **Split by time** — same customers can appear in both sets, but their
   feature snapshot is taken at two different cutoffs (`T_train`,
   `T_test`), mirroring how the model is actually re-scored in production.

## Decision

Split by time (option 2): train on features as of `T_train = 2011-03-31`
with target realized in `(2011-03-31, 2011-06-29]`; test on features as of
`T_test = 2011-06-30` with target realized in `(2011-06-30, 2011-09-28]`.
A customer can legitimately appear in both sets — with two different
"photographs" of their behavior — because that is exactly what happens
when a production model is retrained and re-scored on a rolling basis.

Splitting by customer instead would waste temporal signal: half the
customer base would never contribute a training example from their most
recent behavior, and the resulting metric would answer "does this
generalize to unseen customers" rather than the actually-relevant "does
this generalize to the next quarter," which is the question a media team
is asking when they act on these scores.

## Consequences

- Some information leakage risk exists in theory (the same customer
  contributes to both splits), but not through the feature/target
  boundary within either snapshot — ADR-001's `as_of` contract is enforced
  identically for both cutoffs, so nothing from `(T_train, T_test]`
  leaks into the T_train feature set.
- This only works because the target window for T_train (ending
  2011-06-29) is fully resolved before `T_test`'s feature cutoff
  (2011-06-30) — a one-day margin, deliberately non-overlapping.
