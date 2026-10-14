"""Strict JSON-stat decoding, retaining flags and the observed cell grain."""

import math

import pandas as pd

GRAIN = ["month", "reporter", "partner", "product", "flow"]
ROUTE = ["reporter", "partner", "product", "flow"]
MEASURES = {
    "VALUE_IN_EUROS": "value_eur",
    "QUANTITY_IN_100KG": "mass_100kg",
    "SUPPLEMENTARY_QUANTITY": "supplementary_quantity",
}
EXPECTED = {"freq", "reporter", "partner", "product", "flow", "indicators", "time"}


def category_codes(dimension):
    index = dimension["category"]["index"]
    if isinstance(index, list):
        if len(set(index)) != len(index):
            raise ValueError("Duplicate category codes.")
        return index
    if sorted(index.values()) != list(range(len(index))):
        raise ValueError("Category positions must be unique and contiguous.")
    return [code for code, _ in sorted(index.items(), key=lambda item: item[1])]


def sparse(values):
    if values is None:
        return {}
    if isinstance(values, list):
        return {str(i): value for i, value in enumerate(values) if value is not None}
    return values


def cube_schema(payload):
    dimensions, sizes = payload.get("id", []), payload.get("size", [])
    if (
        payload.get("class") != "dataset"
        or set(dimensions) != EXPECTED
        or len(dimensions) != len(EXPECTED)
    ):
        raise ValueError("Unexpected JSON-stat dimensions.")
    codes = [category_codes(payload["dimension"][name]) for name in dimensions]
    if [len(c) for c in codes] != sizes:
        raise ValueError("Dimension size mismatch.")
    return dimensions, sizes, codes


def cell_at(payload, selections):
    """Read a named raw cell directly; useful for a transparent single-row audit."""
    dimensions, sizes, codes = cube_schema(payload)
    offset = 0
    for name, size, members in zip(dimensions, sizes, codes, strict=True):
        offset = offset * size + members.index(selections[name])
    return {
        "flat_index": offset,
        "value": sparse(payload.get("value")).get(str(offset)),
        "status": sparse(payload.get("status")).get(str(offset), ""),
    }


def decode_trade(payload, record):
    dimensions, sizes, codes = cube_schema(payload)
    values, statuses = sparse(payload.get("value")), sparse(payload.get("status"))
    rows = {}
    for flat in sorted(values.keys() | statuses.keys(), key=int):
        offset = int(flat)
        if offset < 0 or offset >= math.prod(sizes):
            raise ValueError("Cell outside cube bounds.")
        coordinates = [0] * len(sizes)
        for position in range(len(sizes) - 1, -1, -1):
            offset, coordinates[position] = divmod(offset, sizes[position])
        cell = {name: codes[i][coordinates[i]] for i, name in enumerate(dimensions)}
        if cell["freq"] != "M":
            raise ValueError("Non-monthly observation.")
        indicator = cell["indicators"]
        if indicator not in MEASURES:
            raise ValueError(f"Unknown indicator: {indicator}")
        key = tuple(cell[name] for name in ["time", "reporter", "partner", "product", "flow"])
        if key not in rows:
            rows[key] = dict(zip(GRAIN, key, strict=True))
            rows[key].update({name: None for name in MEASURES.values()})
            rows[key].update({f"{name}_status": "" for name in MEASURES.values()})
            for field in ["reporter", "partner", "product"]:
                rows[key][f"{field}_label"] = (
                    payload["dimension"][field]["category"]
                    .get("label", {})
                    .get(cell[field], cell[field])
                )
            rows[key].update(
                {"source_id": record["source_id"], "source_updated": payload.get("updated", "")}
            )
        value = values.get(flat)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError("Non-numeric source measure.")
        rows[key][MEASURES[indicator]] = value
        rows[key][MEASURES[indicator] + "_status"] = statuses.get(flat, "")
    if not rows:
        raise ValueError("Empty source response.")
    frame = pd.DataFrame(rows.values())
    frame["month"] = pd.to_datetime(frame.month + "-01", errors="raise")
    for name in MEASURES.values():
        frame[name] = pd.to_numeric(frame[name], errors="raise").astype(float)
    return frame
