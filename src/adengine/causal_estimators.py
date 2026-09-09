"""Observational causal-effect estimators — ADR-008.

Five estimators, one common return type (ATEEstimate), all validated in
causal_diagnostics.recovery_of_truth against the known ground truth from
causal_simulation.true_effect_summary:

  naive_diff_in_means   deliberately biased under confounding -- the
                        didactic contrast every other estimator is judged
                        against.
  propensity_score_matching   1:1 nearest-neighbor matching on the
                        propensity score, with a covariate-balance check.
  inverse_propensity_weighting   stabilized, truncated Horvitz-Thompson/
                        Hajek estimator.
  difference_in_differences   closed-form DiD on the customer x month panel,
                        with an explicit parallel-trends pre-period check.
  double_ml_ate         cross-fitted partialling-out (Chernozhukov et al.
                        2018), implemented directly on sklearn -- see ADR-008
                        for why this is not delegated to econml.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from adengine.logging_conf import get_logger

logger = get_logger("causal_estimators")


@dataclass(frozen=True)
class ATEEstimate:
    method: str
    ate: float
    ci_low: float
    ci_high: float
    n_treated: int
    n_control: int
    extra: dict = field(default_factory=dict)


def naive_diff_in_means(df: pd.DataFrame, outcome_col: str, treatment_col: str) -> ATEEstimate:
    """E[Y|T=1] - E[Y|T=0]. Unbiased on a randomized layer, biased under
    confounding -- that contrast is the whole point of keeping this estimator
    in the comparison instead of only shipping the estimators that "work".
    """
    treated = df.loc[df[treatment_col], outcome_col]
    control = df.loc[~df[treatment_col], outcome_col]
    ate = float(treated.mean() - control.mean())
    se = float(np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control)))
    return ATEEstimate("naive_diff_in_means", ate, ate - 1.96 * se, ate + 1.96 * se, len(treated), len(control))


def _fit_propensity_scores(df: pd.DataFrame, feature_cols: list[str], treatment_col: str, seed: int) -> np.ndarray:
    X = df[feature_cols].to_numpy()
    t = df[treatment_col].to_numpy().astype(int)
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(X, t)
    return model.predict_proba(X)[:, 1]


def propensity_score_matching(
    df: pd.DataFrame,
    feature_cols: list[str],
    treatment_col: str,
    outcome_col: str,
    caliper: float = 0.2,
    seed: int = 42,
) -> tuple[ATEEstimate, pd.DataFrame]:
    """1:1 nearest-neighbor matching on the propensity score, with replacement,
    caliper expressed in propensity-score standard-deviation units.

    Returns the estimate and the matched-pairs frame (original covariates +
    a matched_group column) -- causal_diagnostics.covariate_balance_table
    compares this against the full `df` to report the pre/post-matching SMD.
    """
    df = df.reset_index(drop=True)
    propensity = _fit_propensity_scores(df, feature_cols, treatment_col, seed)
    df = df.assign(_propensity=propensity)

    treated_idx = df.index[df[treatment_col]].to_numpy()
    control_idx = df.index[~df[treatment_col]].to_numpy()
    caliper_width = caliper * df["_propensity"].std()

    nn = NearestNeighbors(n_neighbors=1).fit(df.loc[control_idx, ["_propensity"]])
    distances, neighbors = nn.kneighbors(df.loc[treated_idx, ["_propensity"]])

    within_caliper = distances[:, 0] <= caliper_width
    matched_treated = treated_idx[within_caliper]
    matched_control = control_idx[neighbors[within_caliper, 0]]

    treated_outcome = df.loc[matched_treated, outcome_col].to_numpy(dtype=float)
    control_outcome = df.loc[matched_control, outcome_col].to_numpy(dtype=float)
    diffs = treated_outcome - control_outcome
    ate = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else float("nan")

    matched_df = pd.concat([
        df.loc[matched_treated].assign(matched_group="treated"),
        df.loc[matched_control].assign(matched_group="control"),
    ], ignore_index=True)

    estimate = ATEEstimate(
        method="psm",
        ate=ate,
        ci_low=ate - 1.96 * se,
        ci_high=ate + 1.96 * se,
        n_treated=len(matched_treated),
        n_control=len(matched_control),
        extra={
            "n_dropped_outside_caliper": int((~within_caliper).sum()),
            "caliper_width": float(caliper_width),
        },
    )
    return estimate, matched_df
