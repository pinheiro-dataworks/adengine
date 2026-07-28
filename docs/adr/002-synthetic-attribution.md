# ADR-002 — Synthetic Media Attribution

**Status:** Accepted

## Context

Online Retail II is a pure transactional export — no channel, no cost, no
campaign data. The project still needs to demonstrate a real media-metrics
engine (CPA, ROAS, LTV:CAC, uplift) end to end, and a metrics engine with
nothing to compute against is not a demonstration of anything.

## Decision

Generate a channel, a synthetic acquisition cost, and a campaign-exposure
flag per customer via a probabilistic rule set fully parameterized in
`configs/attribution.yaml`, conditioned on each customer's real historical
monetary value (customers above the 75th percentile get their channel-mix
probabilities shifted toward `paid_search` and away from `email`/`display`)
so the layer produces analytically interesting, not merely random, output.

Every table, chart, and page built from this layer carries an explicit
"Synthetic" tag in the dashboard and is documented as synthetic here and in
the README's limitations section.

**Calibration note:** the first pass used CAC parameters in the £4–£22
range (chosen to visually match a portfolio prototype's placeholder
numbers). Against this dataset's real historical customer value
(BG/NBD-projected 12-month LTV of £1,200–£2,200), that produced LTV:CAC
ratios of 70×–570× — directionally fine but implausible to present. CAC
parameters were rescaled roughly 8× (see `configs/attribution.yaml`) so
LTV:CAC lands in a defensible 8×–70× range and ROAS (computed on realized
90-day-window revenue, not lifetime revenue — see `metrics.compute_window_revenue`)
lands in a 1.7×–11× range. The *ranking* across channels was never the
issue; only the absolute scale needed calibrating against real data.

## Consequences

- Every media-metric number is a directional demonstration of the engine,
  not a real performance claim — swapping in a real ad-platform export
  requires new config values, not new code, which is the point of keeping
  the generation rules in YAML rather than hardcoded.
- The uplift/Qini analysis in `propensity.py` inherits this: the exposure
  flag is assigned independently of outcome, so the true treatment effect
  is zero for every customer by construction. A Qini curve hugging the
  random-targeting line is the *correct* result, not a failure — see
  ADR discussion in the Propensity dashboard page.
