"""Small forecast experiments with expanding chronological validation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SERIES = ["reporter", "product", "flow"]


@dataclass(frozen=True)
class Candidate:
    name: str
    transform: str = "log"
    seasonal: bool = True
    alpha: float = 10.0
    baseline: str | None = None


CANDIDATES = {
    item.name: item
    for item in [
        Candidate("seasonal_naive", baseline="lag_12"),
        Candidate("last_month", baseline="lag_1"),
        Candidate("ridge_log_lags_a10", seasonal=False),
        Candidate("ridge_log_seasonal_a1", alpha=1.0),
        Candidate("ridge_log_seasonal_a10"),
        Candidate("ridge_log_seasonal_a100", alpha=100.0),
        Candidate("ridge_raw_seasonal_a10", transform="raw"),
    ]
}


def make_features(series, transform="log", seasonal=True):
    if not series.index.equals(pd.date_range(series.index.min(), series.index.max(), freq="MS")):
        raise ValueError(
            "Forecast input must have a complete monthly calendar; retain gaps as NaN."
        )
    if transform not in {"raw", "log"}:
        raise ValueError("Unknown transformation.")
    transformed = np.log1p(series) if transform == "log" else series
    frame = pd.DataFrame(
        {
            "lag_1": transformed.shift(1),
            "lag_12": transformed.shift(12),
            "past_3_mean": transformed.shift(1).rolling(3, min_periods=3).mean(),
        }
    )
    if seasonal:
        frame["month_sin"] = np.sin(2 * np.pi * series.index.month / 12)
        frame["month_cos"] = np.cos(2 * np.pi * series.index.month / 12)
    return frame


def predict_period(monthly, candidate, train_end, test_start, test_end):
    cutoff, start, end = map(pd.Timestamp, [train_end, test_start, test_end])
    if not cutoff < start <= end:
        raise ValueError("Training must precede the evaluation period.")
    frames, audit = [], []
    for key, group in monthly.groupby(SERIES, sort=True):
        series = group.set_index("month").value_eur.sort_index()
        series = series.reindex(pd.date_range(series.index.min(), series.index.max(), freq="MS"))
        # All candidates share the same admissible rows, including the simpler baselines.
        common = make_features(series, "raw", True).notna().all(axis=1) & series.notna()
        features = make_features(series, candidate.transform, candidate.seasonal)
        train = common & (series.index <= cutoff)
        test = common & (series.index >= start) & (series.index <= end)
        if train.sum() < 24 or test.sum() == 0:
            raise ValueError(f"Insufficient comparable history for {key}.")
        if candidate.baseline:
            lag = 12 if candidate.baseline == "lag_12" else 1
            prediction = series.shift(lag).loc[test].to_numpy()
        else:
            model = make_pipeline(StandardScaler(), Ridge(alpha=candidate.alpha))
            target = (
                np.log1p(series.loc[train]) if candidate.transform == "log" else series.loc[train]
            )
            model.fit(features.loc[train], target)
            prediction = model.predict(features.loc[test])
            if candidate.transform == "log":
                prediction = np.expm1(prediction)
            prediction = np.maximum(prediction, 0)
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite forecast.")
        result = pd.DataFrame(
            {
                "month": series.index[test],
                "actual_eur": series.loc[test].to_numpy(),
                "predicted_eur": prediction,
                "model": candidate.name,
                "fit_cutoff": cutoff,
                "latest_input_month": series.index[test] - pd.offsets.MonthBegin(1),
            }
        )
        result[SERIES] = key
        frames.append(result)
        audit.append(
            {
                **dict(zip(SERIES, key, strict=True)),
                "model": candidate.name,
                "train_rows": int(train.sum()),
                "test_rows": int(test.sum()),
                "latest_train_target": series.index[train].max(),
                "first_test_target": series.index[test].min(),
            }
        )
    if not frames:
        raise ValueError("No monthly series.")
    return pd.concat(frames, ignore_index=True), pd.DataFrame(audit)


def metrics(predictions, groups=("model",)):
    values = predictions[["actual_eur", "predicted_eur"]]
    if not np.isfinite(values).all().all() or values.actual_eur.lt(0).any():
        raise ValueError("Metrics require finite actuals and predictions, and nonnegative actuals.")
    rows = []
    grouper = groups[0] if len(groups) == 1 else list(groups)
    for key, frame in predictions.groupby(grouper, sort=True):
        key = (key,) if len(groups) == 1 else key
        errors = frame.predicted_eur - frame.actual_eur
        denominator = frame.actual_eur.sum()
        rows.append(
            {
                **dict(zip(groups, key, strict=True)),
                "n": len(frame),
                "mae_eur": float(errors.abs().mean()),
                "wape": float(errors.abs().sum() / denominator) if denominator else np.nan,
                "bias_eur": float(errors.mean()),
            }
        )
    return pd.DataFrame(rows)


def development_experiments(monthly):
    predictions, audits = [], []
    for validation_year in [2022, 2023]:
        for candidate in CANDIDATES.values():
            frame, audit = predict_period(
                monthly,
                candidate,
                f"{validation_year - 1}-12-01",
                f"{validation_year}-01-01",
                f"{validation_year}-12-01",
            )
            frame["validation_year"] = validation_year
            audit["validation_year"] = validation_year
            predictions.append(frame)
            audits.append(audit)
    return pd.concat(predictions, ignore_index=True), pd.concat(audits, ignore_index=True)


def choose_candidate(development_predictions):
    if development_predictions.month.max() > pd.Timestamp("2023-12-01"):
        raise ValueError("Selection must not include the 2024-2025 evaluation period.")
    fold_metrics = metrics(development_predictions, ("model", "validation_year"))
    ranking = (
        fold_metrics.groupby("model")
        .agg(mean_validation_wape=("wape", "mean"), worst_validation_wape=("wape", "max"))
        .sort_values(["mean_validation_wape", "worst_validation_wape"])
    )
    return str(ranking.index[0]), ranking.reset_index()


def final_comparison(monthly, selected):
    frames, audits = [], []
    for name in dict.fromkeys(["seasonal_naive", "last_month", selected]):
        frame, audit = predict_period(
            monthly, CANDIDATES[name], "2023-12-01", "2024-01-01", "2025-12-01"
        )
        frames.append(frame)
        audits.append(audit)
    return pd.concat(frames, ignore_index=True), pd.concat(audits, ignore_index=True)


def paired_errors(predictions, candidate, baseline="seasonal_naive"):
    key = ["month", *SERIES]
    baseline = predictions.loc[
        predictions.model.eq(baseline), [*key, "actual_eur", "predicted_eur"]
    ]
    other = predictions.loc[predictions.model.eq(candidate), [*key, "predicted_eur"]]
    pairs = baseline.merge(
        other, on=key, suffixes=("_baseline", "_candidate"), validate="one_to_one"
    )
    if len(pairs) != len(baseline) or len(other) != len(baseline):
        raise ValueError("Methods were evaluated on different observations.")
    pairs["baseline_absolute_error"] = (pairs.predicted_eur_baseline - pairs.actual_eur).abs()
    pairs["candidate_absolute_error"] = (pairs.predicted_eur_candidate - pairs.actual_eur).abs()
    return pairs
