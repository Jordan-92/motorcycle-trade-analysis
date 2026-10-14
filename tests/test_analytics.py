import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from moto_study.forecasting import (
    CANDIDATES,
    choose_candidate,
    make_features,
    metrics,
    predict_period,
)
from moto_study.risk import FEATURES, compare_review_queues, fit_forest, historical_features


def test_calendar_gap_not_previous_row(monthly):
    series = monthly.set_index("month").value_eur
    series.loc["2023-12-01"] = np.nan
    features = make_features(series)
    assert pd.isna(features.loc["2024-01-01", "lag_1"])


def test_features_reject_missing_calendar(monthly):
    series = monthly.set_index("month").value_eur.drop(pd.Timestamp("2023-12-01"))
    with pytest.raises(ValueError, match="complete monthly"):
        make_features(series)


def test_future_changes_cannot_change_past_forecast(monthly):
    candidate = CANDIDATES["ridge_raw_seasonal_a10"]
    before, _ = predict_period(monthly, candidate, "2023-12-01", "2024-01-01", "2024-12-01")
    changed = monthly.copy()
    changed.loc[changed.month.ge("2025-01-01"), "value_eur"] *= 100
    after, _ = predict_period(changed, candidate, "2023-12-01", "2024-01-01", "2024-12-01")
    assert_frame_equal(before, after)


def test_methods_share_eligible_months(monthly):
    monthly.loc[monthly.month.eq("2024-06-01"), "value_eur"] = np.nan
    frames = [
        predict_period(monthly, candidate, "2023-12-01", "2024-01-01", "2025-12-01")[0]
        for candidate in CANDIDATES.values()
    ]
    assert all(frame.month.tolist() == frames[0].month.tolist() for frame in frames)
    assert all((frame.latest_input_month < frame.month).all() for frame in frames)


def test_selection_rejects_later_evaluation_data():
    with pytest.raises(ValueError, match="Selection must not"):
        choose_candidate(pd.DataFrame({"month": [pd.Timestamp("2024-01-01")]}))


def test_metric_denominator_is_explicit():
    results = metrics(
        pd.DataFrame({"model": "example", "actual_eur": [100, 200], "predicted_eur": [80, 230]})
    ).iloc[0]
    assert (
        results.mae_eur == 25 and results.wape == pytest.approx(50 / 300) and results.bias_eur == 5
    )


@pytest.mark.parametrize("column", ["actual_eur", "predicted_eur"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_metrics_reject_silently_dropped_errors(column, bad):
    predictions = pd.DataFrame(
        {"model": "example", "actual_eur": [100.0, 200.0], "predicted_eur": [80.0, 230.0]}
    )
    predictions.loc[0, column] = bad
    with pytest.raises(ValueError, match="finite"):
        metrics(predictions)


def test_future_changes_cannot_change_past_risk_features(trade_history):
    before = historical_features(trade_history)
    changed = trade_history.copy()
    changed.loc[changed.month.gt("2023-12-01"), "value_eur"] *= 100
    after = historical_features(changed)
    assert_frame_equal(
        before.loc[before.month.le("2023-12-01"), FEATURES],
        after.loc[after.month.le("2023-12-01"), FEATURES],
    )


def test_flagged_value_excluded_from_current_and_history(trade_history):
    flagged = trade_history.copy()
    flagged.loc[flagged.month.eq("2023-01-01"), "value_eur_status"] = "c"
    before = historical_features(flagged)
    flagged.loc[flagged.month.eq("2023-01-01"), "value_eur"] *= 1e6
    after = historical_features(flagged)
    assert_frame_equal(before[FEATURES], after[FEATURES])
    assert not after.loc[after.month.eq("2023-01-01"), "eligible"].any()


def test_forest_threshold_does_not_use_later_rows(trade_history):
    features = historical_features(trade_history)
    _, _, training = fit_forest(features)
    features.loc[features.month.gt("2023-12-01"), FEATURES] += 100
    _, _, training_again = fit_forest(features)
    np.testing.assert_array_equal(training, training_again)


def test_bad_window_configuration_rejected(trade_history):
    with pytest.raises(ValueError, match="Invalid historical"):
        historical_features(trade_history, window=6, min_history=9)


@pytest.fixture
def review_scores():
    return pd.DataFrame(
        {
            "month": pd.to_datetime(
                ["2023-12-01", "2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01", "2026-01-01"]
            ),
            "reporter": "BE",
            "partner": ["JP", "JP", "DE", "FR", "IT", "JP"],
            "product": "87115000",
            "flow": "1",
            "eligible": [True, True, True, True, False, True],
            "iforest_score": [100, 3, 5, 4, 100, 100],
            "robust_score": [100, 6, 5, 4, 100, 100],
        }
    )


def test_equal_queues_share_pool_and_budget(review_scores):
    result = compare_review_queues(review_scores, 2).set_index("partner")
    assert set(result.index) == {"JP", "DE", "FR"}
    assert result.loc["JP", "queue_group"] == "rule only"
    assert result.loc["DE", "queue_group"] == "both"
    assert result.loc["FR", "queue_group"] == "forest only"
    assert result.forest_rank.le(2).sum() == result.rule_rank.le(2).sum() == 2


def test_review_ties_do_not_depend_on_input_order(review_scores):
    review_scores[["iforest_score", "robust_score"]] = 1.0
    expected = compare_review_queues(review_scores, 2)
    actual = compare_review_queues(review_scores.sample(frac=1, random_state=42), 2)
    assert_frame_equal(expected, actual)
    assert expected.loc[expected.forest_rank.eq(1), "partner"].item() == "DE"


@pytest.mark.parametrize("budget", [0, 4, 1.5])
def test_invalid_review_capacity_rejected(review_scores, budget):
    with pytest.raises(ValueError, match="budget"):
        compare_review_queues(review_scores, budget)


def test_review_does_not_rank_missing_scores(review_scores):
    review_scores.loc[1, "iforest_score"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compare_review_queues(review_scores, 2)


def test_review_duplicate_keys_rejected(review_scores):
    with pytest.raises(ValueError, match="Duplicate"):
        compare_review_queues(pd.concat([review_scores, review_scores.iloc[[1]]]), 2)
