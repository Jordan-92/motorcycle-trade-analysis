"""Manually verify the study in order, with one clean local kernel per notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook", help="Execute just this notebook filename; prerequisites must exist."
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    local_jupyter = root / ".jupyter"
    kernel_dir = local_jupyter / "kernels" / "moto-study-local"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Motorcycle study (local)",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )
    os.environ["JUPYTER_PATH"] = (
        str(local_jupyter) + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    )
    os.environ["JUPYTER_RUNTIME_DIR"] = str(local_jupyter / "runtime")
    os.environ["MPLCONFIGDIR"] = str(local_jupyter / "matplotlib")
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    notebooks = sorted((root / "notebooks").glob("[0-9][0-9]_*.ipynb"))
    if args.notebook:
        notebooks = [path for path in notebooks if path.name == args.notebook]
        if len(notebooks) != 1:
            parser.error(
                "The notebook filename must match one of notebooks/01_*.ipynb to 04_*.ipynb."
            )
    output_dir = root / "data/processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    log = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.system(),
        "dependencies": {
            name: version(name)
            for name in ["pandas", "numpy", "scikit-learn", "jupyterlab", "nbclient"]
        },
        "source_manifest_sha256": hashlib.sha256(
            (root / "data/source_manifest.json").read_bytes()
        ).hexdigest(),
        "scope": "single notebook" if args.notebook else "all notebooks in fresh separate kernels",
        "notebooks": [],
    }
    for path in notebooks:
        print(f"Executing {path.name}", flush=True)
        start = time.monotonic()
        book = nbformat.read(path, as_version=4)
        for cell in book.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        NotebookClient(
            book,
            timeout=600,
            kernel_name="moto-study-local",
            allow_errors=False,
            resources={"metadata": {"path": str(root / "notebooks")}},
        ).execute()
        for cell in book.cells:
            cell.metadata.pop("execution", None)
        nbformat.validate(book)
        nbformat.write(book, path)
        result = {
            "notebook": path.name,
            "seconds": round(time.monotonic() - start, 2),
            "code_cells": sum(cell.cell_type == "code" for cell in book.cells),
            "markdown_cells": sum(cell.cell_type == "markdown" for cell in book.cells),
            "embedded_figures": sum(
                "image/png" in output.get("data", {})
                for cell in book.cells
                for output in cell.get("outputs", [])
            ),
            "errors": sum(
                output.output_type == "error"
                for cell in book.cells
                for output in cell.get("outputs", [])
            ),
            "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        log["notebooks"].append(result)
        print(json.dumps(result), flush=True)
    log["completed_utc"] = datetime.now(timezone.utc).isoformat()
    name = (
        "execution-log.json"
        if not args.notebook
        else f"execution-log-{Path(args.notebook).stem}.json"
    )
    (output_dir / name).write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    print(
        "All requested notebooks executed successfully; outputs are saved in the notebooks.",
        flush=True,
    )


if __name__ == "__main__":
    main()
