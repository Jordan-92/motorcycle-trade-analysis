"""Small forecast experiments with expanding chronological validation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SERIES = ["reporter", "product", "flow"]
# At the start of a target month, allow for publication after the reference month.
# This is a conservative scenario, not reconstructed historical data vintages.
AVAILABLE_LAG = 4
SELECTION_END = pd.Timestamp("2023-09-01")


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
        Candidate("latest_known", baseline="known_value"),
        Candidate("ridge_raw_lags_a10", transform="raw", seasonal=False),
        Candidate("ridge_log_seasonal_a10"),
        Candidate("ridge_raw_seasonal_a10", transform="raw"),
    ]
}


def make_features(series, transform="raw", seasonal=True, available_lag=AVAILABLE_LAG):
    if not series.index.equals(pd.date_range(series.index.min(), series.index.max(), freq="MS")):
        raise ValueError(
            "Forecast input must have a complete monthly calendar; retain gaps as NaN."
        )
    if transform not in {"raw", "log"}:
        raise ValueError("Unknown transformation.")
    if not isinstance(available_lag, int) or not 1 <= available_lag <= 12:
        raise ValueError("Availability lag must be 1 to 12 calendar months.")
    transformed = np.log1p(series) if transform == "log" else series
    frame = pd.DataFrame(
        {
            "known_value": transformed.shift(available_lag),
            "lag_12": transformed.shift(12),
            "known_3_mean": transformed.shift(available_lag).rolling(3, min_periods=3).mean(),
        }
    )
    if seasonal:
        frame["month_sin"] = np.sin(2 * np.pi * series.index.month / 12)
        frame["month_cos"] = np.cos(2 * np.pi * series.index.month / 12)
    return frame


def predict_period(monthly, candidate, test_start, test_end, available_lag=AVAILABLE_LAG):
    start, end = map(pd.Timestamp, [test_start, test_end])
    if start > end:
        raise ValueError("Invalid evaluation period.")
    frames, audit = [], []
    for key, group in monthly.groupby(SERIES, sort=True):
        series = group.set_index("month").value_eur.sort_index()
        series = series.reindex(pd.date_range(series.index.min(), series.index.max(), freq="MS"))
        # All candidates share the same admissible rows, including the simpler baselines.
        common = (
            make_features(series, "raw", True, available_lag).notna().all(axis=1) & series.notna()
        )
        features = make_features(series, candidate.transform, candidate.seasonal, available_lag)
        test = common & (series.index >= start) & (series.index <= end)
        if not test.any():
            raise ValueError(f"No comparable evaluation months for {key}.")
        for month in series.index[test]:
            cutoff = month - pd.DateOffset(months=available_lag)
            train = common & (series.index <= cutoff)
            if train.sum() < 24:
                raise ValueError(f"Insufficient comparable history for {key} at {month}.")
            if candidate.baseline:
                lag = 12 if candidate.baseline == "lag_12" else available_lag
                prediction = float(series.shift(lag).loc[month])
            else:
                model = make_pipeline(StandardScaler(), Ridge(alpha=candidate.alpha))
                target = (
                    np.log1p(series.loc[train])
                    if candidate.transform == "log"
                    else series.loc[train]
                )
                model.fit(features.loc[train], target)
                prediction = float(model.predict(features.loc[[month]])[0])
                if candidate.transform == "log":
                    prediction = float(np.expm1(prediction))
                prediction = max(prediction, 0)
            if not np.isfinite(prediction):
                raise ValueError("Nonfinite forecast.")
            frames.append(
                {
                    **dict(zip(SERIES, key, strict=True)),
                    "month": month,
                    "actual_eur": float(series.loc[month]),
                    "predicted_eur": prediction,
                    "model": candidate.name,
                    "latest_input_month": cutoff,
                }
            )
            audit.append(
                {
                    **dict(zip(SERIES, key, strict=True)),
                    "model": candidate.name,
                    "forecast_month": month,
                    "train_rows": int(train.sum()),
                    "latest_train_target": series.index[train].max(),
                    "available_through": cutoff,
                }
            )
    if not frames:
        raise ValueError("No monthly series.")
    return pd.DataFrame(frames), pd.DataFrame(audit)


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
    for start, end in [("2022-01-01", "2022-12-01"), ("2023-01-01", "2023-09-01")]:
        for candidate in CANDIDATES.values():
            frame, audit = predict_period(
                monthly,
                candidate,
                start,
                end,
            )
            frame["validation_year"] = int(start[:4])
            audit["validation_year"] = int(start[:4])
            predictions.append(frame)
            audits.append(audit)
    return pd.concat(predictions, ignore_index=True), pd.concat(audits, ignore_index=True)


def choose_candidate(development_predictions):
    if development_predictions.month.max() > SELECTION_END:
        raise ValueError("Selection must use targets available before January 2024.")
    fold_metrics = metrics(development_predictions, ("model", "validation_year"))
    ranking = (
        fold_metrics.groupby("model")
        .agg(mean_validation_wape=("wape", "mean"), worst_validation_wape=("wape", "max"))
        .sort_values(["mean_validation_wape", "worst_validation_wape"])
    )
    return str(ranking.index[0]), ranking.reset_index()


def final_comparison(monthly, selected):
    frames, audits = [], []
    for name in dict.fromkeys(["seasonal_naive", "latest_known", selected]):
        frame, audit = predict_period(monthly, CANDIDATES[name], "2024-01-01", "2025-12-01")
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
