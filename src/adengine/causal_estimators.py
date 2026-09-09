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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
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


def inverse_propensity_weighting(
    df: pd.DataFrame,
    feature_cols: list[str],
    treatment_col: str,
    outcome_col: str,
    weight_trunc_pct: float = 99,
    seed: int = 42,
) -> ATEEstimate:
    """Stabilized (Hajek) inverse-propensity weighting.

    Stabilization multiplies each weight by the marginal treatment rate
    instead of using raw 1/e(x), which keeps weights centered near 1 for a
    well-specified propensity model. Truncating at the weight_trunc_pct
    percentile bounds the variance blow-up a handful of near-zero propensity
    scores would otherwise cause.
    """
    df = df.reset_index(drop=True)
    propensity = _fit_propensity_scores(df, feature_cols, treatment_col, seed)
    treatment = df[treatment_col].to_numpy().astype(bool)
    outcome = df[outcome_col].to_numpy(dtype=float)

    p_treat_marginal = treatment.mean()
    raw_weight = np.where(
        treatment, p_treat_marginal / propensity, (1 - p_treat_marginal) / (1 - propensity)
    )

    cap = float(np.percentile(raw_weight, weight_trunc_pct))
    weight = np.minimum(raw_weight, cap)
    n_truncated = int((raw_weight > cap).sum())

    treated_mean = float(np.sum(weight[treatment] * outcome[treatment]) / np.sum(weight[treatment]))
    control_mean = float(np.sum(weight[~treatment] * outcome[~treatment]) / np.sum(weight[~treatment]))
    ate = treated_mean - control_mean

    # Weighted-residual sandwich approximation for the SE of each Hajek mean.
    var_treated = np.sum((weight[treatment] * (outcome[treatment] - treated_mean)) ** 2) / (np.sum(weight[treatment]) ** 2)
    var_control = np.sum((weight[~treatment] * (outcome[~treatment] - control_mean)) ** 2) / (np.sum(weight[~treatment]) ** 2)
    se = float(np.sqrt(var_treated + var_control))

    return ATEEstimate(
        method="ipw",
        ate=ate,
        ci_low=ate - 1.96 * se,
        ci_high=ate + 1.96 * se,
        n_treated=int(treatment.sum()),
        n_control=int((~treatment).sum()),
        extra={
            "n_weights_truncated": n_truncated,
            "weight_cap": cap,
            "max_weight_before_truncation": float(raw_weight.max()),
        },
    )


def difference_in_differences(
    panel: pd.DataFrame,
    outcome_col: str = "did_outcome",
    treatment_col: str = "treatment",
    is_post_col: str = "is_post",
    relative_month_col: str = "relative_month",
    parallel_trends_slope_threshold: float = 0.015,
) -> ATEEstimate:
    """Closed-form 2x2 DiD on a customer x month panel (built by
    causal_simulation.build_did_panel):

        ate = (treated_post - treated_pre) - (control_post - control_pre)

    The parallel-trends pre-period check is always run and always reported
    in `extra` -- pass or fail -- never silently omitted, since a DiD
    estimate without that check reported is not trustworthy on its own.
    """
    def _group_mean(treatment: bool, is_post: bool) -> float:
        mask = (panel[treatment_col] == treatment) & (panel[is_post_col] == is_post)
        return float(panel.loc[mask, outcome_col].mean())

    def _group_var_n(treatment: bool, is_post: bool) -> tuple[float, int]:
        mask = (panel[treatment_col] == treatment) & (panel[is_post_col] == is_post)
        values = panel.loc[mask, outcome_col]
        return float(values.var(ddof=1)), len(values)

    treated_pre, treated_post = _group_mean(True, False), _group_mean(True, True)
    control_pre, control_post = _group_mean(False, False), _group_mean(False, True)
    ate = (treated_post - treated_pre) - (control_post - control_pre)

    var_tp, n_tp = _group_var_n(True, True)
    var_tr, n_tr = _group_var_n(True, False)
    var_cp, n_cp = _group_var_n(False, True)
    var_cr, n_cr = _group_var_n(False, False)
    se = float(np.sqrt(var_tp / n_tp + var_tr / n_tr + var_cp / n_cp + var_cr / n_cr))

    def _pre_period_slope(treatment: bool) -> float:
        pre = panel[(~panel[is_post_col]) & (panel[treatment_col] == treatment)]
        by_month = pre.groupby(relative_month_col)[outcome_col].mean()
        return float(np.polyfit(by_month.index.to_numpy(), by_month.to_numpy(), 1)[0])

    slope_treated, slope_control = _pre_period_slope(True), _pre_period_slope(False)
    slope_diff = abs(slope_treated - slope_control)
    parallel_trends_holds = slope_diff < parallel_trends_slope_threshold

    if not parallel_trends_holds:
        logger.warning(
            "difference_in_differences: parallel-trends check FAILED "
            f"(slope diff {slope_diff:.4f} >= threshold {parallel_trends_slope_threshold}); "
            "the DiD estimate below should not be trusted as-is."
        )

    return ATEEstimate(
        method="did",
        ate=float(ate),
        ci_low=float(ate - 1.96 * se),
        ci_high=float(ate + 1.96 * se),
        n_treated=n_tp + n_tr,
        n_control=n_cp + n_cr,
        extra={
            "pre_trend_slope_treated": slope_treated,
            "pre_trend_slope_control": slope_control,
            "pre_trend_slope_diff": slope_diff,
            "parallel_trends_holds": bool(parallel_trends_holds),
        },
    )


def double_ml_ate(
    df: pd.DataFrame,
    feature_cols: list[str],
    treatment_col: str,
    outcome_col: str,
    n_folds: int = 5,
    seed: int = 42,
) -> ATEEstimate:
    """Cross-fitted partialling-out Double ML (Chernozhukov et al. 2018,
    "Double/Debiased Machine Learning for Treatment and Causal Parameters"),
    implemented directly on sklearn -- see ADR-008 for why this is not
    delegated to econml.

    Partially linear model: Y = theta*T + g(X) + eps, T = m(X) + v(X). Both
    nuisance functions (g = E[Y|X], m = E[T|X]) are fit on K-1 folds and
    predicted out-of-fold, so theta is estimated from residuals the nuisance
    models never saw -- the Neyman-orthogonality property that makes this
    estimator robust to nuisance-model bias, unlike a naive plug-in.
    """
    df = df.reset_index(drop=True)
    X = df[feature_cols].to_numpy()
    treatment = df[treatment_col].to_numpy().astype(float)
    outcome = df[outcome_col].to_numpy().astype(float)
    n = len(df)

    outcome_resid = np.zeros(n)
    treatment_resid = np.zeros(n)

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in kfold.split(X):
        outcome_model = HistGradientBoostingRegressor(random_state=seed)
        outcome_model.fit(X[train_idx], outcome[train_idx])
        g_hat = outcome_model.predict(X[test_idx])

        treatment_model = HistGradientBoostingClassifier(random_state=seed)
        treatment_model.fit(X[train_idx], treatment[train_idx].astype(int))
        m_hat = treatment_model.predict_proba(X[test_idx])[:, 1]

        outcome_resid[test_idx] = outcome[test_idx] - g_hat
        treatment_resid[test_idx] = treatment[test_idx] - m_hat

    theta_hat = float(np.sum(treatment_resid * outcome_resid) / np.sum(treatment_resid ** 2))

    # Closed-form asymptotic variance for the partialling-out estimator
    # (Chernozhukov et al. 2018, eq. 2.3): Var(theta) ~ E[psi^2] / n, where
    # psi_i = v_i * (eps_i - theta*v_i) is the (estimated) Neyman-orthogonal score.
    psi = treatment_resid * (outcome_resid - theta_hat * treatment_resid)
    sigma_sq = np.mean(psi ** 2) / (np.mean(treatment_resid ** 2) ** 2)
    se = float(np.sqrt(sigma_sq / n))

    return ATEEstimate(
        method="dml",
        ate=theta_hat,
        ci_low=theta_hat - 1.96 * se,
        ci_high=theta_hat + 1.96 * se,
        n_treated=int(treatment.sum()),
        n_control=int((1 - treatment).sum()),
        extra={"n_folds": n_folds, "se": se},
    )
