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
    series.loc["2023-09-01"] = np.nan
    features = make_features(series)
    assert pd.isna(features.loc["2024-01-01", "known_value"])


def test_features_reject_missing_calendar(monthly):
    series = monthly.set_index("month").value_eur.drop(pd.Timestamp("2023-12-01"))
    with pytest.raises(ValueError, match="complete monthly"):
        make_features(series)


def test_future_changes_cannot_change_past_forecast(monthly):
    candidate = CANDIDATES["ridge_raw_seasonal_a10"]
    before, _ = predict_period(monthly, candidate, "2024-01-01", "2024-12-01")
    changed = monthly.copy()
    changed.loc[changed.month.ge("2025-01-01"), "value_eur"] *= 100
    after, _ = predict_period(changed, candidate, "2024-01-01", "2024-12-01")
    assert_frame_equal(before, after)


def test_methods_share_eligible_months(monthly):
    monthly.loc[monthly.month.eq("2024-06-01"), "value_eur"] = np.nan
    frames = [
        predict_period(monthly, candidate, "2024-01-01", "2025-12-01")[0]
        for candidate in CANDIDATES.values()
    ]
    assert all(frame.month.tolist() == frames[0].month.tolist() for frame in frames)
    assert all((frame.latest_input_month < frame.month).all() for frame in frames)


def test_selection_rejects_later_evaluation_data():
    with pytest.raises(ValueError, match="Selection must use"):
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
    future = changed.month.gt("2023-12-01")
    changed.loc[future, "value_eur"] *= 100
    changed.loc[future, ["net_mass_kg", "unit_value_eur_per_kg"]] *= 10
    after = historical_features(changed)
    assert not before.loc[before.month.gt("2023-12-01"), FEATURES].equals(
        after.loc[after.month.gt("2023-12-01"), FEATURES]
    )
    assert_frame_equal(
        before.loc[before.month.le("2023-12-01"), FEATURES],
        after.loc[after.month.le("2023-12-01"), FEATURES],
    )


def test_flagged_value_excluded_from_current_and_history(trade_history):
    flagged = trade_history.copy()
    flagged.loc[flagged.month.eq("2023-01-01"), "value_eur_status"] = "c"
    before = historical_features(flagged)
    flagged.loc[flagged.month.eq("2023-01-01"), ["value_eur", "unit_value_eur_per_kg"]] *= 1e6
    after = historical_features(flagged)
    assert_frame_equal(before[FEATURES], after[FEATURES])
    assert not after.loc[after.month.eq("2023-01-01"), "eligible"].any()


def test_forest_fit_does_not_use_later_rows(trade_history):
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
            "value_eur": 20_000,
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


@pytest.mark.parametrize("budget", [0, -1, 1.5])
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


def test_unpublished_targets_do_not_affect_forecast(monthly):
    candidate = CANDIDATES["ridge_raw_seasonal_a10"]
    before, audit = predict_period(monthly, candidate, "2024-01-01", "2024-01-01")
    monthly.loc[monthly.month.between("2023-10-01", "2023-12-01"), "value_eur"] *= 100
    after, _ = predict_period(monthly, candidate, "2024-01-01", "2024-01-01")
    assert_frame_equal(before, after)
    assert audit.latest_train_target.max() == pd.Timestamp("2023-09-01")


def test_repeated_seasonal_peak_is_not_an_anomaly(trade_history):
    frame = trade_history.copy()
    frame["net_mass_kg"] = np.where(frame.month.dt.month.eq(6), 20_000.0, 1_000.0)
    frame["value_eur"] = frame.net_mass_kg * 30
    frame["unit_value_eur_per_kg"] = 30.0
    result = historical_features(frame)
    later = result.loc[result.eligible & result.month.ge("2021-01-01")]
    assert len(later) > 0
    assert np.allclose(later[FEATURES], 0)


def test_capacity_is_per_month_and_small_values_are_excluded(review_scores):
    next_month = review_scores.loc[review_scores.month.eq("2024-01-01")].copy()
    next_month["month"] = pd.Timestamp("2024-02-01")
    next_month["month"] = next_month.month.astype(review_scores.month.dtype)
    frame = pd.concat([review_scores, next_month], ignore_index=True)
    frame.loc[frame.partner.eq("FR"), "value_eur"] = 15
    ranked = compare_review_queues(frame, 1)
    assert "FR" not in set(ranked.partner)
    assert ranked.loc[ranked.forest_rank.le(ranked.slots)].groupby("month").size().eq(1).all()
    review_scores.loc[review_scores.partner.eq("FR"), "value_eur"] = 15
    earlier = compare_review_queues(review_scores, 1)
    assert_frame_equal(
        ranked.loc[ranked.month.eq("2024-01-01")].reset_index(drop=True),
        earlier.loc[earlier.partner.ne("FR")].assign(monthly_pool=2).reset_index(drop=True),
    )


def test_current_change_cannot_enter_its_own_reference(trade_history):
    before = historical_features(trade_history)
    current = trade_history.month.eq("2024-01-01")
    trade_history.loc[current, ["value_eur", "unit_value_eur_per_kg"]] *= 100
    after = historical_features(trade_history)
    cols = [
        f + suffix
        for f in FEATURES
        for suffix in ["_reference_median", "_reference_scale", "_history_count"]
    ]
    assert_frame_equal(
        before.loc[before.month.eq("2024-01-01"), cols],
        after.loc[after.month.eq("2024-01-01"), cols],
    )
    assert after.loc[after.month.eq("2024-01-01"), "unit_value_z"].min() > 40
