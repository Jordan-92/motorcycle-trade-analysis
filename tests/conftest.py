import itertools

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def cube():
    names = ["freq", "reporter", "partner", "product", "flow", "indicators", "time"]
    members = [
        ["M"],
        ["BE"],
        ["JP", "WORLD"],
        ["87115000"],
        ["1"],
        ["VALUE_IN_EUROS", "QUANTITY_IN_100KG", "SUPPLEMENTARY_QUANTITY"],
        ["2024-01", "2024-02"],
    ]
    values = {}
    for i, row in enumerate(itertools.product(*members)):
        values[str(i)] = {
            "VALUE_IN_EUROS": 10000,
            "QUANTITY_IN_100KG": 5,
            "SUPPLEMENTARY_QUANTITY": 2,
        }[row[5]]
    return {
        "class": "dataset",
        "id": names,
        "size": list(map(len, members)),
        "dimension": {
            name: {"category": {"index": dict(zip(codes, range(len(codes)), strict=True))}}
            for name, codes in zip(names, members, strict=True)
        },
        "value": values,
    }


@pytest.fixture
def source():
    return {"source_id": "synthetic-source"}


@pytest.fixture
def scope():
    return {
        "reporters": ["BE"],
        "products": ["87115000"],
        "flows": ["1"],
        "start_year": 2024,
        "end_year": 2024,
    }


@pytest.fixture
def monthly():
    dates = pd.date_range("2018-01-01", "2025-12-01", freq="MS")
    return pd.DataFrame(
        {
            "month": dates,
            "reporter": "BE",
            "product": "87115000",
            "flow": "1",
            "value_eur": 10000
            + np.arange(len(dates)) * 40
            + 2000 * np.sin(dates.month * 2 * np.pi / 12),
        }
    )


@pytest.fixture
def trade_history(monthly):
    parts = []
    for partner, multiplier in [("JP", 1), ("DE", 0.7)]:
        frame = monthly.copy()
        frame["partner"] = partner
        frame["value_eur"] *= multiplier
        frame["net_mass_kg"] = frame.value_eur / 30
        frame["unit_value_eur_per_kg"] = 30.0
        frame["value_eur_status"] = ""
        frame["mass_100kg_status"] = ""
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)
