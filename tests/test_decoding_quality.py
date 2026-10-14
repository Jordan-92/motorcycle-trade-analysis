import copy

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from moto_study.decoding import category_codes, cell_at, decode_trade
from moto_study.quality import (
    assert_quality,
    monthly_totals,
    prepare_trade,
    quality_checks,
)


def test_direct_cell_and_decoded_row_agree(cube, source):
    selection = dict(
        freq="M",
        reporter="BE",
        partner="JP",
        product="87115000",
        flow="1",
        indicators="VALUE_IN_EUROS",
        time="2024-01",
    )
    assert cell_at(cube, selection)["value"] == 10000
    row = decode_trade(cube, source).query("partner == 'JP'").iloc[0]
    assert row.value_eur == 10000 and row.mass_100kg == 5


def test_dense_sparse_equivalent(cube, source):
    dense = copy.deepcopy(cube)
    dense["value"] = [cube["value"][str(i)] for i in range(12)]
    assert_frame_equal(decode_trade(cube, source), decode_trade(dense, source))


def test_status_only_missing_cell_preserved(cube, source):
    del cube["value"]["0"]
    cube["status"] = {"0": "c"}
    row = decode_trade(cube, source).iloc[0]
    assert pd.isna(row.value_eur) and row.value_eur_status == "c"


def test_absent_flag_is_not_a_missing_measure(cube, source):
    row = decode_trade(cube, source).iloc[0]
    assert row.value_eur == 10000 and row.value_eur_status == ""


@pytest.mark.parametrize("problem", ["missing", "flagged"])
def test_partial_monthly_sum_is_not_a_forecast_target(trade_history, problem):
    row = trade_history.index[0]
    if problem == "missing":
        trade_history.loc[row, "value_eur"] = np.nan
    else:
        trade_history.loc[row, "value_eur_status"] = "c"
    monthly = monthly_totals(trade_history)
    first = monthly.iloc[0]
    assert np.isnan(first.value_eur) and first.n_partners == 2
    assert first.n_missing_value == (problem == "missing")
    assert first.n_flagged_value == (problem == "flagged")
    assert monthly.value_eur.iloc[1:].notna().all()


def test_observed_zero_total_remains_zero(trade_history):
    trade_history.loc[trade_history.month.eq("2018-01-01"), "value_eur"] = 0
    assert monthly_totals(trade_history).value_eur.iloc[0] == 0


def test_unknown_units_stop(cube, source):
    cube["dimension"]["indicators"]["category"]["index"] = {
        "DOLLARS": 0,
        "QUANTITY_IN_100KG": 1,
        "SUPPLEMENTARY_QUANTITY": 2,
    }
    with pytest.raises(ValueError, match="Unknown indicator"):
        decode_trade(cube, source)


def test_invalid_category_positions_stop():
    with pytest.raises(ValueError, match="unique and contiguous"):
        category_codes({"category": {"index": {"a": 0, "b": 0}}})


def test_out_of_bounds_stops(cube, source):
    cube["value"]["99"] = 1
    with pytest.raises(ValueError, match="outside"):
        decode_trade(cube, source)


def test_scope_conversion_and_exclusions(cube, source, scope):
    raw = decode_trade(cube, source)
    trade, excluded, quarantine = prepare_trade(raw, scope)
    assert len(trade) == 2 and len(excluded) == 2 and quarantine.empty
    assert (trade.net_mass_kg == 500).all() and (trade.unit_value_eur_per_kg == 20).all()
    assert_quality(quality_checks(trade, [source["source_id"]]))


def test_duplicate_grain_is_not_silently_deduplicated(cube, source, scope):
    raw = decode_trade(cube, source)
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_trade(pd.concat([raw, raw.iloc[:1]]), scope)


def test_zero_mass_not_imputed(cube, source, scope):
    raw = decode_trade(cube, source)
    raw.loc[0, "mass_100kg"] = 0
    trade, _, _ = prepare_trade(raw, scope)
    assert trade.net_mass_kg.iloc[0] == 0 and np.isnan(trade.unit_value_eur_per_kg.iloc[0])


def test_negative_quarantined_not_clipped(cube, source, scope):
    raw = decode_trade(cube, source)
    raw.loc[0, "value_eur"] = -1
    trade, _, quarantine = prepare_trade(raw, scope)
    assert len(quarantine) == 1 and quarantine.value_eur.iloc[0] == -1
    assert len(trade) == 1


def test_missing_partition_stops(cube, source, scope):
    scope["end_year"] = 2025
    with pytest.raises(ValueError, match="partition"):
        prepare_trade(decode_trade(cube, source), scope)
