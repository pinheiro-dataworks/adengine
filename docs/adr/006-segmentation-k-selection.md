# ADR-006 — Segmentation K Selection

**Status:** Accepted

## Context

K-Means needs a chosen K. The standard numeric criteria (elbow / inertia,
silhouette score) don't always agree with what's actually useful to a
marketing team, and on this dataset they actively disagree with each
other.

## Decision

The full sweep (`configs/model.yaml: segmentation.k_range = [2, 10]`) on
the real customer base gives:

| K | Inertia | Silhouette |
|---|---------|------------|
| 2 | 7,519 | **0.424** (global max) |
| 4 | 4,170 | 0.328 |
| **5** | **3,419** | **0.342** (local peak) |
| 6 | 2,892 | 0.342 (tied local peak) |

Silhouette is maximized at **K=2**, but that split is just "one-time
buyers vs. everyone else" — a real structural split, but not one with five
distinct marketing actions attached to it. **K=5** is chosen instead: it
sits at a local silhouette peak exactly where the inertia elbow flattens,
and produces the five classic RFM tiers (Champions, Loyal Customers,
Potential Loyalists, At Risk, Hibernating), each independently actionable
(`segmentation.SEGMENT_ACTIONS`). This is the business-interpretability
tie-break the project's methodology calls for when the purely numeric
criterion doesn't point to an actionable answer.

`configs/model.yaml: segmentation.chosen_k` pins this decision explicitly
rather than leaving it to `argmax(silhouette)` at run time, so the choice
is version-controlled and requires a deliberate edit (and a note here) to
change.

## Stability validation

K=5 clears the stability bar the doc requires before a segmentation is
treated as production-ready: re-fitting across 20 reseeded runs gives
ARI = 0.998 (mean); re-fitting on 50 bootstrap resamples of 80% of the
base gives ARI = 0.952 (mean), comfortably above the 0.85 acceptance
threshold (`segmentation.stability.ari_acceptance_threshold`).

## Consequences

- The naming rule (`segmentation.name_segments`) further splits the
  "good recency + high frequency" quadrant by monetary value into
  Champions vs. Loyal Customers — a quadrant-only rule cannot tell those
  two apart when more than one cluster lands in the same quadrant, which
  happens at K=5.
- If a future re-run of the pipeline on updated data shifts the optimal K,
  `chosen_k` must be revisited manually — this is a deliberate design
  choice, not an oversight.
