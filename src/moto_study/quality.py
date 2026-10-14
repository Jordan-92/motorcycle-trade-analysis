"""Explicit scope decisions and data contracts, independent of notebook presentation."""

import itertools

import numpy as np
import pandas as pd
import pycountry

from .decoding import GRAIN, MEASURES, ROUTE

COUNTRIES = {country.alpha_2 for country in pycountry.countries}
ALIASES = {"UK": "GB", "EL": "GR"}


def assert_source_scope(raw, scope):
    for field, allowed in [
        ("reporter", scope["reporters"]),
        ("product", scope["products"]),
        ("flow", scope["flows"]),
    ]:
        if not raw[field].isin(allowed).all():
            raise ValueError(f"Unexpected {field} in source data.")
    if not raw.month.dt.year.between(scope["start_year"], scope["end_year"]).all():
        raise ValueError("Unexpected observation period.")
    if not raw.month.dt.day.eq(1).all():
        raise ValueError("Monthly dates must be month starts.")
    if raw.duplicated(GRAIN).any():
        raise ValueError("Duplicate observation grain in raw inputs.")
    for field in ["reporter", "partner", "product", "flow"]:
        if raw[field].isna().any() or not raw[field].eq(raw[field].str.strip()).all():
            raise ValueError(f"Missing or inconsistently formatted {field} identifier.")


def prepare_trade(raw, scope):
    assert_source_scope(raw, scope)
    frame = raw.copy()
    frame["partner_original"] = frame.partner
    frame["partner"] = frame.partner.replace(ALIASES)
    excluded = frame.loc[~frame.partner.isin(COUNTRIES) | frame.reporter.eq(frame.partner)].copy()
    excluded["exclusion_reason"] = np.where(
        excluded.reporter.eq(excluded.partner), "self_partner", "aggregate_or_special_partner"
    )
    frame = frame.drop(index=excluded.index).copy()
    if frame.duplicated(GRAIN).any():
        raise ValueError("Country aliases created a duplicate observation grain.")
    bad = pd.Series(False, index=frame.index)
    for field in MEASURES.values():
        bad |= frame[field].lt(0) | (frame[field].notna() & ~np.isfinite(frame[field]))
    quarantine = frame.loc[bad].copy()
    quarantine["exclusion_reason"] = "negative_or_nonfinite_measure"
    frame = frame.loc[~bad].copy()
    if frame.empty:
        raise ValueError("No accepted observations.")
    frame["net_mass_kg"] = frame.mass_100kg * 100
    frame["unit_value_eur_per_kg"] = frame.value_eur.div(
        frame.net_mass_kg.where(frame.net_mass_kg.gt(0))
    )
    expected = set(
        itertools.product(
            scope["reporters"], scope["products"], range(scope["start_year"], scope["end_year"] + 1)
        )
    )
    observed = set(zip(frame.reporter, frame["product"], frame.month.dt.year, strict=True))
    if expected - observed:
        raise ValueError("Entire reporter/product/year partition absent.")
    return frame.sort_values(GRAIN).reset_index(drop=True), excluded, quarantine


def quality_checks(trade, source_ids):
    """One result per meaningful contract; missingness and eligibility are reported separately."""
    flags = [name + "_status" for name in MEASURES.values()]
    all_measures = [*MEASURES.values(), "net_mass_kg"]
    bad_numeric = sum(
        int((trade[name].lt(0) | (trade[name].notna() & ~np.isfinite(trade[name]))).sum())
        for name in all_measures
    )
    checks = [
        (
            "Unique observation grain",
            int(trade.duplicated(GRAIN).sum()),
            "Avoid multiplying monetary totals.",
        ),
        (
            "Known source reference",
            int((~trade.source_id.isin(source_ids)).sum()),
            "Every observation must trace to evidence.",
        ),
        (
            "Detailed CN8 identifiers",
            int((~trade["product"].str.fullmatch(r"8711\d{4}")).sum()),
            "Do not mix parent totals and child products.",
        ),
        (
            "Individual partner codes",
            int((~trade.partner.isin(COUNTRIES)).sum()),
            "Exclude overlapping regional totals.",
        ),
        (
            "Valid direction codes",
            int((~trade.flow.isin(["1", "2"])).sum()),
            "Keep imports and exports separate.",
        ),
        (
            "Finite nonnegative measures",
            bad_numeric,
            "Invalid measures cannot enter analysis silently.",
        ),
        (
            "Mass conversion",
            int((~np.isclose(trade.net_mass_kg, trade.mass_100kg * 100, equal_nan=True)).sum()),
            "Convert 100 kg units exactly once.",
        ),
        (
            "Undefined ratio at zero/missing mass",
            int((trade.net_mass_kg.fillna(0).le(0) & trade.unit_value_eur_per_kg.notna()).sum()),
            "Do not manufacture EUR/kg at zero mass.",
        ),
        (
            "Positive-mass ratio identity",
            int(
                (
                    trade.net_mass_kg.gt(0)
                    & ~np.isclose(
                        trade.unit_value_eur_per_kg * trade.net_mass_kg,
                        trade.value_eur,
                        equal_nan=True,
                    )
                ).sum()
            ),
            "Check the derived relationship.",
        ),
        (
            "Observation flags preserved",
            int(trade[flags].isna().any(axis=1).sum()),
            "An empty flag and a dropped flag are different.",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "check": name,
                "violations": count,
                "status": "PASS" if count == 0 else "FAIL",
                "why_it_matters": why,
            }
            for name, count, why in checks
        ]
    )


def assert_quality(checks):
    failed = checks.loc[checks.status.ne("PASS"), "check"].tolist()
    if failed:
        raise ValueError(f"Failed quality contracts: {failed}")


def monthly_totals(trade):
    # A partial sum is not a complete target. Keep its coverage counts visible.
    monthly = (
        trade.assign(flagged_value=trade.value_eur_status.fillna("").ne(""))
        .groupby(["month", "reporter", "product", "flow"], observed=True)
        .agg(
            value_eur=("value_eur", lambda x: x.sum(min_count=1)),
            n_observed=("partner", "size"),
            n_partners=("partner", "nunique"),
            n_missing_value=("value_eur", lambda x: x.isna().sum()),
            n_flagged_value=("flagged_value", "sum"),
        )
        .reset_index()
    )
    incomplete = monthly.n_missing_value.gt(0) | monthly.n_flagged_value.gt(0)
    monthly.loc[incomplete, "value_eur"] = np.nan
    return monthly


def route_coverage(trade):
    rows = []
    for key, group in trade.groupby(ROUTE, sort=True):
        dates = pd.date_range(group.month.min(), group.month.max(), freq="MS")
        rows.append(
            {
                **dict(zip(ROUTE, key, strict=True)),
                "observed_months": len(group),
                "first_month": group.month.min(),
                "last_month": group.month.max(),
                "months_between_first_last": len(dates),
                "internal_gaps": len(dates) - len(group),
            }
        )
    return pd.DataFrame(rows)
