# Motorcycle trade analysis

**Python · pandas · scikit-learn · Jupyter · OpenAI API**

A personal study of motorcycle trade in six European countries, 2018–2025. I check the data, compare monthly forecasts and experiment with priorities for fraud-related review. The analysis starts with one source observation.

## Read the notebooks

1. [Understand the data](notebooks/01_descriptive_analysis.ipynb) — units, national totals, growth, concentration and seasonality.
2. [Compare forecasts](notebooks/02_monthly_forecasts.ipynb) — correlation, publication delays, simple baselines and a forecast miss.
3. [Prioritize observations](notebooks/03_statistical_prioritization.ipynb) — seasonal deviations, equal monthly workloads and sensitivity checks.
4. [Review assistant](notebooks/04_review_assistant.ipynb) — read-only LLM tools, verified facts and saved live responses.

Outputs are included. **Bold sentences give the findings and decisions.** Short notes explain the technical terms.

## What I found

- **Scope matters.** In March 2023, Belgian imports above 800 cc include EUR 37.4m without an identified partner. Dropping it would roughly halve the national total. Reconciliation now checks all 3,456 country/product/direction/month totals.
- **Seasonality helps, but forecasts remain imperfect.** Ridge WAPE is 24.40% / 25.90% in 2024 / 2025, versus 25.63% / 27.87% for the same month last year. It improves 23/36 series. WAPE measures total absolute error relative to total value; it is not an accuracy rate.
- **Unusual does not mean fraudulent.** With five slots per month, the rule and forest share 103 of their 120 selections. Rankings are sensitive to settings, and many routes lack enough history. I keep the transparent rule as the reference.

The data checks also reveal an implausible item count. I retain the source and explain the issue instead of silently correcting it.

## Run locally

Python 3.12, from this folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
jupyter lab
```

Run notebooks 01–04 in order. The default run is offline:

```bash
python scripts/execute_notebooks.py
python -m pytest -W error
```

Raw data are bundled. Small reusable functions live in `src/moto_study/`; checks live in `tests/`. Generated tables go to `data/processed/`.

## Optional assistant

Install `requirements-assistant.txt`, then set `RUN_LLM = True` in notebook 04. It uses `OPENAI_API_KEY` or a hidden input prompt. Only selected public-data evidence is sent to OpenAI; API use is billed.

The LLM chooses read-only checks, supported findings and a next step. Python calculates and renders the factual sentences. Unsupported answers are rejected. Saved prompts, responses and evidence are in [assistant_checks.json](docs/assistant_checks.json); notebook 04 reviews the results. This is a small integration check, not evidence of fraud accuracy or reviewer time saved.

[Sources and limits](docs/DATA_SOURCES.md): revised monthly aggregates, a partial motorcycle market and no verified fraud outcomes. Forecasts remain retrospective despite the publication-delay assumption.
