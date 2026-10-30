import copy

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from moto_study.decoding import category_codes, cell_at, decode_trade
from moto_study.quality import (
    assert_quality,
    country_totals,
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


def test_route_subset_is_not_a_national_forecast_target(trade_history):
    with pytest.raises(ValueError, match="reconciled"):
        monthly_totals(trade_history)


def test_world_and_unallocated_values_reconcile(cube, source, scope):
    raw = decode_trade(cube, source)
    extra = raw.loc[raw.partner.eq("JP")].assign(partner="QW")
    raw.loc[raw.partner.eq("WORLD"), ["value_eur", "mass_100kg", "supplementary_quantity"]] *= 2
    raw = pd.concat([raw, extra], ignore_index=True)
    trade, _, _ = prepare_trade(raw, scope)
    totals = country_totals(raw, trade)
    assert totals.value_eur.eq(20_000).all()
    assert totals.known_value_share.eq(0.5).all()
    assert totals.unallocated_value_eur.eq(10_000).all()
    assert monthly_totals(totals).value_eur.eq(20_000).all()
    totals.loc[0, "value_eur_status"] = "c"
    assert pd.isna(monthly_totals(totals).loc[0, "value_eur"])


def test_wrong_world_total_fails(cube, source, scope):
    raw = decode_trade(cube, source)
    raw.loc[raw.partner.eq("WORLD"), "value_eur"] += 1
    trade, _, _ = prepare_trade(raw, scope)
    with pytest.raises(ValueError, match="reconcile"):
        country_totals(raw, trade)


@pytest.mark.parametrize("partner", ["XS", "XK", "XC", "XL"])
def test_valid_non_iso_partners_are_kept(cube, source, scope, partner):
    raw = decode_trade(cube, source).replace({"partner": {"JP": partner}})
    trade, _, _ = prepare_trade(raw, scope)
    assert set(trade.partner) == {partner}
    assert country_totals(raw, trade).known_value_share.eq(1).all()
    assert trade.items_reported.eq(2).all()
    assert trade.value_eur_per_item.eq(5_000).all()


def test_unknown_partner_requires_review(cube, source, scope):
    raw = decode_trade(cube, source).replace({"partner": {"JP": "QQ"}})
    with pytest.raises(ValueError, match="unknown Eurostat"):
        prepare_trade(raw, scope)


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


def test_belgian_world_total_includes_large_unallocated_component():
    # Real source regression: dropping QW would halve these monthly national values.
    from pathlib import Path

    from moto_study.io import read_json, select_source

    root = Path(__file__).resolve().parents[1]
    source = select_source(root, reporter="BE", product="87115000", year=2023)
    raw = decode_trade(read_json(root, source), source)
    scope = dict(
        reporters=["BE"], products=["87115000"], flows=["1", "2"], start_year=2023, end_year=2023
    )
    trade, _, _ = prepare_trade(raw, scope)
    totals = country_totals(raw, trade)
    march = totals.loc[totals.month.eq("2023-03-01") & totals.flow.eq("1")].iloc[0]
    assert march.value_eur == 72_723_079
    assert march.known_partner_value_eur == 35_357_424
    assert march.unallocated_value_eur == 37_365_655
