"""Explicit scope decisions and data contracts, independent of notebook presentation."""

import itertools

import numpy as np
import pandas as pd
import pycountry

from .decoding import GRAIN, MEASURES, ROUTE

# Eurostat's geonomenclature has valid partners outside the ISO list.
COUNTRIES = {country.alpha_2 for country in pycountry.countries} | {"XS", "XK", "XC", "XL"}
ALIASES = {"UK": "GB", "EL": "GR"}
UNALLOCATED = {"QS", "QV", "QW"}
TOTAL_KEY = ["month", "reporter", "product", "flow"]
ITEM_PRODUCTS = {"87113090", "87114000", "87115000"}  # CN supplementary unit: p/st.


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
    totals = frame.partner.eq("WORLD") | frame.partner.str.startswith(("EXT_", "INT_"))
    unknown = set(frame.loc[~totals, "partner"]) - COUNTRIES - UNALLOCATED
    if unknown:
        raise ValueError(f"Review unknown Eurostat partner codes: {sorted(unknown)}")
    excluded = frame.loc[~frame.partner.isin(COUNTRIES) | frame.reporter.eq(frame.partner)].copy()
    excluded["exclusion_reason"] = np.where(
        excluded.partner.isin(UNALLOCATED), "unallocated_partner", "overlapping_total"
    )
    excluded.loc[excluded.reporter.eq(excluded.partner), "exclusion_reason"] = "self_partner"
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
    valid_items = (
        frame["product"].isin(ITEM_PRODUCTS)
        & frame.supplementary_quantity.gt(0)
        & frame.supplementary_quantity_status.eq("")
    )
    frame["items_reported"] = frame.supplementary_quantity.where(valid_items)
    frame["value_eur_per_item"] = (
        frame.value_eur.where(frame.value_eur_status.eq("")) / frame.items_reported
    )
    frame["mass_kg_per_item"] = (
        frame.net_mass_kg.where(frame.mass_100kg_status.eq("")) / frame.items_reported
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


def country_totals(raw, trade):
    """Use WORLD once; reconcile it with detailed and unallocated partners."""
    world = raw.loc[raw.partner.eq("WORLD")].set_index(TOTAL_KEY).sort_index()
    if world.empty or world.index.has_duplicates:
        raise ValueError("Require one WORLD total per country, product, direction and month.")
    unallocated = raw.loc[raw.partner.isin(UNALLOCATED)]
    component_columns = [*TOTAL_KEY, *MEASURES.values()]
    components = pd.concat(
        [x[component_columns] for x in [trade, unallocated] if not x.empty], ignore_index=True
    )
    if set(map(tuple, components[TOTAL_KEY].to_numpy())) != set(world.index):
        raise ValueError("Country totals and partner components cover different months.")
    columns = list(MEASURES.values())
    if components[columns].isna().any().any() or world[columns].isna().any().any():
        raise ValueError("Missing measurements prevent country-total reconciliation.")
    if not np.isfinite(components[columns]).all().all() or components[columns].lt(0).any().any():
        raise ValueError("Invalid partner component.")
    summed = components.groupby(TOTAL_KEY)[columns].sum().reindex(world.index)
    if not np.isfinite(world[columns]).all().all() or world[columns].lt(0).any().any():
        raise ValueError("Invalid country total.")
    if not np.allclose(summed, world[columns], rtol=1e-10, atol=0.01):
        raise ValueError("Partner components do not reconcile with WORLD totals.")
    result = world[[*columns, *[f"{c}_status" for c in columns], "source_id"]].copy()
    result["known_partner_value_eur"] = (
        trade.groupby(TOTAL_KEY).value_eur.sum().reindex(world.index, fill_value=0)
    )
    result["unallocated_value_eur"] = (
        unallocated.groupby(TOTAL_KEY).value_eur.sum().reindex(world.index, fill_value=0)
    )
    result["known_value_share"] = result.known_partner_value_eur / result.value_eur.where(
        result.value_eur.gt(0)
    )
    result["n_partners"] = (
        trade.groupby(TOTAL_KEY).partner.nunique().reindex(world.index, fill_value=0)
    )
    result["net_mass_kg"] = result.mass_100kg * 100
    return result.reset_index()


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


def monthly_totals(totals):
    """Forecast the reconciled national total, not a changing subset of partners."""
    if "known_partner_value_eur" not in totals or totals.duplicated(TOTAL_KEY).any():
        raise ValueError("Use the reconciled country_totals table from notebook 01.")
    result = totals.copy()
    result.loc[~result.value_eur_status.eq(""), "value_eur"] = np.nan
    return result


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
