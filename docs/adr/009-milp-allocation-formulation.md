# ADR-009 — MILP Budget-Constrained Allocation Formulation

**Status:** Accepted

## Context

Given a per-customer CATE estimate and a fixed budget, decide which
customers receive a targeted intervention (a retention nudge, a discount, a
personalized outreach) to maximize total incremental value — a decision this
project's existing Hill-curve budget simulator (`simulator.py`) does not
make: that module decides *how much to spend per channel*, not *which
individual customers* within a channel's spend actually receive treatment.
The two questions are complementary, not competing, and this module does not
touch or replace `simulator.py`.

`ortools` (Google's CP-SAT constraint-programming solver) was the original
candidate, per the same reasoning process as ADR-008: check what the problem
actually requires before reaching for a specialized framework. This
allocation decision is, precisely, a 0/1 knapsack — maximize a linear
objective (`sum(cate_i * x_i)`) subject to a linear budget constraint
(`sum(cost_i * x_i) <= budget`) with binary decision variables, optionally
extended with a few more linear inequality constraints for per-segment
spend caps. CP-SAT is built for combinatorial problems with complex logical
or scheduling constraints (`AllDifferent`, disjunctive intervals, Boolean
implication chains) — genuine overkill for a linear knapsack, and its
`ortools` package is a large, self-contained C++ binary, a much heavier
install than anything else in this project's dependency list.

`scipy.optimize.milp` — a mixed-integer linear solver wrapping HiGHS,
available since SciPy 1.9 — solves exactly this class of problem, and
`pyproject.toml` already pins `scipy>=1.13`. `simulator.py` already depends
on `scipy.optimize` for its own (continuous, SLSQP) budget optimization, so
this is not even a new import path within the project's dependency graph,
just a different function from the same package.

## Decision

**Formulation** (`allocation_optimizer.optimize_targeting`):

- Decision variables: `x_i ∈ {0, 1}` — one binary variable per eligible
  customer (customers with `cate_hat <= 0` are excluded before optimization;
  spending budget on a predicted-negative or zero effect is never rational,
  and dropping them shrinks the problem).
- Objective: `maximize Σ(cate_hat_i * x_i)`, passed to `scipy.optimize.milp`
  as `minimize -cate_hat_i * x_i` (the solver only minimizes).
- Constraints: `Σ(cost_i * x_i) <= budget` as a `LinearConstraint`, plus one
  additional row per entry in `segment_capacity_pct` (e.g. `{"Champions":
  0.2}` caps total spend on that segment at 20% of budget) — a direct
  illustration that adding a real business constraint to this formulation is
  one more constraint-matrix row, not a rewrite.
- `integrality = 1` on every variable, bounds `[0, 1]` — the standard MILP
  encoding of a binary decision.

**Per-customer CATE source.** None of the five ATE estimators in
`causal_estimators.py` (ADR-008) produce a per-customer score — they
estimate one aggregate number. `causal_estimators.t_learner_cate` (a
T-Learner: two `HistGradientBoostingClassifier` models, one per treatment
arm, each scored on every customer) fills that gap, reusing the same
T-Learner pattern already used for the zero-effect uplift/Qini demo in
`propensity.py`. Its docstring is explicit about a real limitation:
individual-level CATE prediction from a single binary outcome per customer
is inherently noisy (irreducible Bernoulli variance dominates), even though
the *segment-average* predicted CATE tracks the true CATE-by-segment well
(verified in `tests/test_causal_estimators.py`, Spearman correlation 0.9 at
N=6000). The optimizer is therefore validated on aggregate outcomes
(total incremental value, budget adherence), not on any single customer's
predicted score being individually correct.

**Validation:** `allocation_optimizer.greedy_baseline` sorts customers by
`cate/cost` and fills the budget — optimal for the *continuous relaxation*
of the knapsack, but not always for the strict 0/1 case.
`tests/test_allocation_optimizer.py` includes the textbook counterexample
where greedy is provably suboptimal (picks value 160 against a true optimum
of 220) and confirms the MILP finds the true optimum every time — the same
"the MILP solution should never lose to the heuristic" property this
project's Definition of Done calls for.

## Consequences

- Zero new dependencies: `allocation_optimizer.py` uses only `numpy`,
  `pandas`, and `scipy.optimize` — all already declared in `pyproject.toml`.
- `configs/allocation.yaml` holds the budget, the synthetic treatment-cost
  model, and any segment capacity constraints — adding a new business
  constraint (a channel cap, a minimum-treated-per-segment floor) is a
  config change plus one more `LinearConstraint` row, not new solver
  machinery.
- This module reads `causal_simulation.parquet` (ADR-007) and
  `causal_estimators.t_learner_cate` (ADR-008) but does not modify either —
  strictly additive, same rule as the rest of this extension.
- If a future requirement genuinely needs combinatorial or scheduling
  constraints CP-SAT is built for (e.g. sequencing multi-touch campaigns
  with cooldown windows), that would be new grounds to reconsider `ortools`
  — the plain budget-constrained knapsack this ADR covers does not meet
  that bar.
