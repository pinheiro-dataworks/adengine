"""Propensity-to-convert model — logistic baseline, calibrated HistGBM, uplift.

Temporal validation only (§5.2, ADR-004): the model is trained on features
computed as of T_train and evaluated on features computed as of T_test — never
a random k-fold split, which would let the same customer's future snapshot
leak into training. Calibration is the load-bearing metric here: the budget
simulator (simulator.py) consumes these scores as literal probabilities, so a
miscalibrated model would make the optimizer arithmetically wrong, not just
imprecise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from adengine.logging_conf import get_logger, log_step

logger = get_logger("propensity")

FEATURE_COLUMNS = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "product_diversity",
    "avg_days_between_orders",
    "single_buyer",
    "spend_trend_slope",
    "tenure_days",
    "is_uk",
]
TARGET_COLUMN = "target_conversion"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["avg_order_value"] = out["monetary"] / out["frequency"]
    out["single_buyer"] = out["single_buyer"].astype(int)
    out["is_uk"] = out["is_uk"].astype(int)
    return out


def build_baseline_pipeline(cfg: dict) -> Pipeline:
    log_cfg = cfg["propensity"]["logistic"]
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight=log_cfg["class_weight"], max_iter=log_cfg["max_iter"])),
    ])


def build_hist_gbm(cfg: dict) -> HistGradientBoostingClassifier:
    gbm_cfg = cfg["propensity"]["hist_gbm"]
    return HistGradientBoostingClassifier(
        max_iter=gbm_cfg["max_iter"],
        early_stopping=gbm_cfg["early_stopping"],
        validation_fraction=gbm_cfg["validation_fraction"],
        learning_rate=gbm_cfg["learning_rate"],
        max_depth=gbm_cfg["max_depth"],
        l2_regularization=gbm_cfg["l2_regularization"],
        random_state=cfg["random_state"],
    )


def train_propensity_models(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: dict) -> dict:
    train = prepare_features(train_df)
    test = prepare_features(test_df)
    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN].astype(int)
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN].astype(int)

    with log_step(logger, "propensity.fit_baseline") as rec:
        baseline = build_baseline_pipeline(cfg).fit(X_train, y_train)
        baseline_proba = baseline.predict_proba(X_test)[:, 1]
        rec["roc_auc"] = float(roc_auc_score(y_test, baseline_proba))

    with log_step(logger, "propensity.fit_gbm_raw") as rec:
        gbm_raw = build_hist_gbm(cfg).fit(X_train, y_train)
        gbm_raw_proba = gbm_raw.predict_proba(X_test)[:, 1]
        rec["roc_auc"] = float(roc_auc_score(y_test, gbm_raw_proba))

    cal_cfg = cfg["propensity"]["calibration"]
    with log_step(logger, "propensity.fit_gbm_calibrated") as rec:
        gbm_calibrated = CalibratedClassifierCV(
            build_hist_gbm(cfg), method=cal_cfg["method"], cv=cal_cfg["cv"]
        ).fit(X_train, y_train)
        gbm_cal_proba = gbm_calibrated.predict_proba(X_test)[:, 1]
        rec["roc_auc"] = float(roc_auc_score(y_test, gbm_cal_proba))

    metrics = {
        "baseline": {
            "roc_auc": float(roc_auc_score(y_test, baseline_proba)),
            "pr_auc": float(average_precision_score(y_test, baseline_proba)),
            "brier": float(brier_score_loss(y_test, baseline_proba)),
        },
        "gbm_raw": {
            "roc_auc": float(roc_auc_score(y_test, gbm_raw_proba)),
            "pr_auc": float(average_precision_score(y_test, gbm_raw_proba)),
            "brier": float(brier_score_loss(y_test, gbm_raw_proba)),
        },
        "gbm_calibrated": {
            "roc_auc": float(roc_auc_score(y_test, gbm_cal_proba)),
            "pr_auc": float(average_precision_score(y_test, gbm_cal_proba)),
            "brier": float(brier_score_loss(y_test, gbm_cal_proba)),
        },
    }

    roc_baseline = roc_curve(y_test, baseline_proba)
    roc_gbm = roc_curve(y_test, gbm_cal_proba)

    calib_raw = calibration_curve(y_test, gbm_raw_proba, n_bins=10, strategy="uniform")
    calib_cal = calibration_curve(y_test, gbm_cal_proba, n_bins=10, strategy="uniform")

    with log_step(logger, "propensity.permutation_importance") as rec:
        pi_cfg = cfg["propensity"]["permutation_importance"]
        pi = permutation_importance(
            gbm_calibrated, X_test, y_test,
            n_repeats=pi_cfg["n_repeats"], scoring=pi_cfg["scoring"], random_state=cfg["random_state"],
        )
        importances = (
            pd.DataFrame({"feature": FEATURE_COLUMNS, "importance_mean": pi.importances_mean, "importance_std": pi.importances_std})
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
        rec["top_feature"] = importances.iloc[0]["feature"]

    scored_test = test_df[["customer_id"]].copy()
    scored_test["score_baseline"] = baseline_proba
    scored_test["score_gbm_calibrated"] = gbm_cal_proba
    scored_test["target_conversion"] = y_test.to_numpy()

    return {
        "baseline_model": baseline,
        "gbm_raw_model": gbm_raw,
        "gbm_calibrated_model": gbm_calibrated,
        "metrics": metrics,
        "roc_curve": {
            "baseline": {"fpr": roc_baseline[0].tolist(), "tpr": roc_baseline[1].tolist()},
            "gbm": {"fpr": roc_gbm[0].tolist(), "tpr": roc_gbm[1].tolist()},
        },
        "calibration_curve": {
            "raw": {"predicted": calib_raw[1].tolist(), "observed": calib_raw[0].tolist()},
            "calibrated": {"predicted": calib_cal[1].tolist(), "observed": calib_cal[0].tolist()},
        },
        "permutation_importance": importances,
        "scored_test": scored_test,
    }


def qini_curve(y_true: np.ndarray, uplift_score: np.ndarray, treatment: np.ndarray, n_bins: int = 20) -> pd.DataFrame:
    """Cumulative incremental-conversions curve, ranked by predicted uplift."""
    order = np.argsort(-uplift_score)
    y, t = y_true[order], treatment[order]
    n = len(y)
    cuts = np.linspace(0, n, n_bins + 1)[1:].astype(int)

    rows = [{"pct_targeted": 0.0, "incremental_conversions": 0.0}]
    for cut in cuts:
        y_cum, t_cum = y[:cut], t[:cut]
        treated_conv = y_cum[t_cum == 1].sum()
        control_conv = y_cum[t_cum == 0].sum()
        # Qini: incremental conversions scaled to equal-sized treatment/control.
        incremental = treated_conv - control_conv * (t_cum.sum() / max((1 - t_cum).sum(), 1))
        rows.append({"pct_targeted": round(100 * cut / n, 1), "incremental_conversions": float(incremental)})

    curve = pd.DataFrame(rows)
    # Random-targeting reference: a straight line from (0,0) to the overall uplift.
    overall = curve["incremental_conversions"].iloc[-1]
    curve["random_reference"] = curve["pct_targeted"] / 100 * overall
    return curve


def qini_coefficient(curve: pd.DataFrame, n_test: int) -> float:
    """Area between the model curve and the random-targeting line, normalized
    by population size so the number is comparable across test-set sizes.
    Not the bounded [-1, 1] textbook statistic — with a synthetic, randomly
    assigned treatment (ADR-002) the true uplift is zero everywhere, so this
    is expected to hover near 0 with sampling noise in either direction.
    """
    area_model = np.trapezoid(curve["incremental_conversions"], curve["pct_targeted"])
    area_random = np.trapezoid(curve["random_reference"], curve["pct_targeted"])
    return float((area_model - area_random) / n_test)


def fit_uplift_t_learner(features_df: pd.DataFrame, attribution_df: pd.DataFrame, cfg: dict) -> dict:
    """T-Learner on the synthetic exposure flag (§5.5) — directional demo, not a causal claim.

    Uses a dedicated random split of the "current" snapshot (not the T_train /
    T_test propensity split): the exposure flag is a synthetic label attached
    to that single snapshot, so there is no leakage concern in splitting it
    randomly, and doing so keeps this extension decoupled from the temporal
    validation used for the propensity model itself.
    """
    from sklearn.model_selection import train_test_split

    df = prepare_features(features_df).merge(attribution_df[["customer_id", "exposed"]], on="customer_id")
    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=cfg["random_state"], stratify=df[["exposed", TARGET_COLUMN]].astype(str).agg("-".join, axis=1)
    )

    X_train, y_train, t_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN].astype(int), train_df["exposed"]
    X_test, y_test, t_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN].astype(int), test_df["exposed"]

    with log_step(logger, "propensity.uplift_fit_treatment_arm"):
        model_treated = build_hist_gbm(cfg).fit(X_train[t_train], y_train[t_train])
    with log_step(logger, "propensity.uplift_fit_control_arm"):
        model_control = build_hist_gbm(cfg).fit(X_train[~t_train], y_train[~t_train])

    p_treated = model_treated.predict_proba(X_test)[:, 1]
    p_control = model_control.predict_proba(X_test)[:, 1]
    uplift = p_treated - p_control

    curve = qini_curve(y_test.to_numpy(), uplift, t_test.to_numpy())
    qini = qini_coefficient(curve, n_test=len(y_test))

    return {
        "model_treated": model_treated,
        "model_control": model_control,
        "qini_curve": curve,
        "qini_coefficient": qini,
        "uplift_by_decile": _uplift_by_decile(uplift, y_test.to_numpy(), t_test.to_numpy()),
    }


def _uplift_by_decile(uplift: np.ndarray, y: np.ndarray, t: np.ndarray) -> pd.DataFrame:
    decile = pd.qcut(-uplift, 10, labels=False, duplicates="drop")
    df = pd.DataFrame({"decile": decile, "y": y, "t": t})
    rows = []
    for d, g in df.groupby("decile"):
        treated_rate = g.loc[g["t"] == 1, "y"].mean() if (g["t"] == 1).any() else np.nan
        control_rate = g.loc[g["t"] == 0, "y"].mean() if (g["t"] == 0).any() else np.nan
        rows.append({"decile": int(d) + 1, "treated_conv_rate": treated_rate, "control_conv_rate": control_rate,
                      "uplift": (treated_rate - control_rate) if pd.notna(treated_rate) and pd.notna(control_rate) else np.nan})
    return pd.DataFrame(rows).sort_values("decile")


def main() -> None:
    import argparse
    import json

    import joblib

    from adengine.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipe_cfg = load_config(args.pipeline_config)
    marts_dir = Path(pipe_cfg["paths"]["marts_dir"])
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(marts_dir / "customer_features_train.parquet")
    test_df = pd.read_parquet(marts_dir / "customer_features_test.parquet")

    result = train_propensity_models(train_df, test_df, cfg)

    joblib.dump(result["baseline_model"], models_dir / "logistic_baseline.joblib")
    joblib.dump(result["gbm_calibrated_model"], models_dir / "histgbm_calibrated.joblib")
    result["scored_test"].to_parquet(marts_dir / "propensity_scores.parquet", index=False)
    result["permutation_importance"].to_parquet(marts_dir / "permutation_importance.parquet", index=False)

    artifact = {
        "metrics": result["metrics"],
        "roc_curve": result["roc_curve"],
        "calibration_curve": result["calibration_curve"],
        "feature_columns": FEATURE_COLUMNS,
    }
    (marts_dir / "propensity_summary.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    logger.info(json.dumps({"step": "propensity.summary", "metrics": result["metrics"]}))

    attribution_path = marts_dir / "synthetic_attribution.parquet"
    if attribution_path.exists():
        attribution_df = pd.read_parquet(attribution_path)
        uplift_result = fit_uplift_t_learner(test_df, attribution_df, cfg)
        uplift_result["qini_curve"].to_parquet(marts_dir / "qini_curve.parquet", index=False)
        uplift_result["uplift_by_decile"].to_parquet(marts_dir / "uplift_by_decile.parquet", index=False)
        (marts_dir / "uplift_summary.json").write_text(
            json.dumps({"qini_coefficient": uplift_result["qini_coefficient"]}, indent=2), encoding="utf-8"
        )
        logger.info(json.dumps({"step": "propensity.uplift_summary", "qini_coefficient": uplift_result["qini_coefficient"]}))

    logger.info('{"step": "propensity.done"}')


if __name__ == "__main__":
    main()
