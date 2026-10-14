# Motorcycle trade analysis

**Python · pandas · scikit-learn · JupyterLab**  
Six European reporters · 2018–2025 · 80,652 monthly trade observations

A personal project with three objectives: describe trade patterns, compare monthly forecasts and experiment with statistical prioritization for fraud-related review.

## Read the analysis

1. [Descriptive analysis](notebooks/01_descriptive_analysis.ipynb) — start with one observation; check units, coverage and seasonal patterns.
2. [Monthly forecasts](notebooks/02_monthly_forecasts.ipynb) — work through one model fit, compare baselines and explain a large miss.
3. [Statistical prioritization](notebooks/03_statistical_prioritization.ipynb) — compare equal-sized queues and investigate two disagreements.

Each notebook contains executed code and figures. Bold sentences give the findings and decisions; short notes explain the terms and how to read the results.

## Findings

- Spring/early-summer imports exceed winter in **33/36 country-years** (2018–2023). A year-round route cohort gives similar results.
- Belgian import growth is **2.68%**, versus **9.75%** for continuously reported routes. Coverage changes the interpretation.
- Ridge WAPE is **20.63% / 24.93%** in 2024 / 2025, versus **24.94% / 28.02%** for the same-month-last-year baseline. Large misses remain.
- With **127 review slots each**, the rule and forest share **99 cases** and each selects **28 different cases**. Without fraud labels, neither ranking can be declared more accurate.

## Run

Python 3.12. From the project folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
jupyter lab
```

Run notebooks 01–03 in order. To check the full sequence:

```bash
python scripts/execute_notebooks.py
python -m pytest -W error
```

Raw files are bundled. Reusable functions are in `src/moto_study/`; tests are in `tests/`. Prepared data and the review queue go to `data/processed/`.

## Limits

Three product codes cover part of motorcycle trade. Rows are monthly aggregates, not individual transactions or retail sales. Revised, already-inspected data support retrospective comparisons; unusual values do not establish fraud.

[Sources and units](docs/DATA_SOURCES.md) · [Verification](docs/VERIFICATION.md)
