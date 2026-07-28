# ADR-005 — Hill Function for Budget Response Curves

**Status:** Accepted

## Context

The budget simulator needs a functional form mapping spend → expected
conversions per channel, fit from `vmax` (audience ceiling) and `k`
(half-saturation spend). Two natural candidates:

- **Log-response**: `a · ln(1 + b·spend)`
- **Hill function**: `vmax · spend^s / (k^s + spend^s)`

## Decision

Use the Hill function. It has an explicit ceiling (`vmax`), which is
physically correct — an audience is finite, so conversions must saturate,
not grow without bound. The log-response form is simpler to fit but never
saturates: at high spend it keeps producing incremental conversions
forever, which lets `scipy.optimize.minimize` push all budget into
whichever channel has the largest log-slope and report an unrealistic
result at the extremes.

Parameters are anchored to real pipeline output, not hand-tuned:
`vmax = addressable_audience_size × mean_calibrated_propensity` for that
channel's customers (`simulator.derive_hill_params`), and
`k = observed CPA` from `metrics.channel_metrics`. The shape parameter `s`
defaults to 1.3 for every channel (`configs/model.yaml`).

## Consequences

- The objective (sum of per-channel Hill functions) is concave in the
  region SLSQP searches, so a single local optimum is effectively global —
  verified in practice with a 10-point multistart
  (`configs/model.yaml: simulator.optimizer.multistart_points`), which
  costs milliseconds and is cheap insurance rather than a real necessity
  here.
- Extending the simulator with more channels only requires new `vmax`/`k`
  values derived the same way — no change to the optimization logic.
