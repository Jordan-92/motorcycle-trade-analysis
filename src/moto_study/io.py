"""Read and verify the frozen source files and checked tables."""

import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd


def project_root(start=None):
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "data/source_manifest.json").exists() and (
            candidate / "pyproject.toml"
        ).exists():
            return candidate
    raise FileNotFoundError("Start JupyterLab inside the motorcycle-trade-study project.")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(root):
    return json.loads((Path(root) / "data/source_manifest.json").read_text())


def source_path(root, record):
    root = Path(root).resolve()
    path = (root / record["path"]).resolve()
    if not path.is_relative_to(root / "data/raw"):
        raise ValueError("Source path escapes the frozen raw-data directory.")
    return path


def read_bytes(root, record):
    path = source_path(root, record)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != record["sha256"]:
        raise ValueError(f"Source checksum mismatch: {record['source_id']}")
    return content


def read_json(root, record):
    return json.loads(read_bytes(root, record))


def inventory(root):
    rows = []
    for item in manifest(root)["sources"]:
        query = parse_qs(urlparse(item["url"]).query)
        rows.append(
            {
                **item,
                "reporter": query.get("reporter", [None])[0],
                "product": query.get("product", [None])[0],
                "year": int(query["sinceTimePeriod"][0][:4])
                if "sinceTimePeriod" in query
                else None,
            }
        )
    return pd.DataFrame(rows)


def select_source(root, reporter="BE", product="87115000", year=2024):
    rows = inventory(root)
    selected = rows.loc[
        rows.reporter.eq(reporter) & rows["product"].eq(product) & rows.year.eq(year)
    ]
    if len(selected) != 1:
        raise ValueError("Expected exactly one source for the requested reporter/product/year.")
    return selected.iloc[0].to_dict()


def verify_sources(root):
    rows = []
    for record in manifest(root)["sources"]:
        content = read_bytes(root, record)
        rows.append(
            {
                "source_id": record["source_id"],
                "provider": record["provider"],
                "bytes": len(content),
                "checksum_ok": True,
            }
        )
    result = pd.DataFrame(rows)
    if result.source_id.duplicated().any():
        raise ValueError("Duplicate source identifiers.")
    return result


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def save_prepared(root, trade, checks):
    folder = Path(root) / "data/processed"
    folder.mkdir(parents=True, exist_ok=True)
    trade.to_parquet(folder / "trade.parquet", index=False)
    checks.to_csv(folder / "quality_checks.csv", index=False)
    save_json(
        folder / "prepared_manifest.json",
        {
            "source_manifest_sha256": sha256(Path(root) / "data/source_manifest.json"),
            "files": {
                name: sha256(folder / name) for name in ["trade.parquet", "quality_checks.csv"]
            },
            "trade_rows": len(trade),
        },
    )


def load_prepared(root):
    folder = Path(root) / "data/processed"
    if not (folder / "prepared_manifest.json").exists():
        raise FileNotFoundError(
            "Run notebook 01 from top to bottom to create the checked analysis tables."
        )
    metadata = json.loads((folder / "prepared_manifest.json").read_text())
    if metadata["source_manifest_sha256"] != sha256(Path(root) / "data/source_manifest.json"):
        raise ValueError("Source inventory changed. Rerun notebook 01.")
    for name, expected in metadata["files"].items():
        if sha256(folder / name) != expected:
            raise ValueError(f"Prepared file changed: {name}. Rerun notebook 01.")
    return pd.read_parquet(folder / "trade.parquet")
