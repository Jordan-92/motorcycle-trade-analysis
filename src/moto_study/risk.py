"""Route-relative review signals; no fraud labels or supervised accuracy claims."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .decoding import GRAIN, ROUTE

FEATURES = ["value_z", "mass_z", "unit_value_z"]


def historical_features(trade, window=12, min_history=9, scale_floor=0.10):
    if min_history > window or min_history < 1 or scale_floor <= 0:
        raise ValueError("Invalid historical reference configuration.")
    parts = []
    for _, group in trade.groupby(ROUTE, sort=True):
        group = group.sort_values("month").set_index("month")
        original = group.index
        monthly = group.reindex(pd.date_range(original.min(), original.max(), freq="MS"))
        for source, feature, flags in [
            ("value_eur", "value_z", ["value_eur_status"]),
            ("net_mass_kg", "mass_z", ["mass_100kg_status"]),
            ("unit_value_eur_per_kg", "unit_value_z", ["value_eur_status", "mass_100kg_status"]),
        ]:
            clean = monthly[flags].fillna("").eq("").all(axis=1)
            log = np.log(monthly[source].where(monthly[source].gt(0) & clean))
            history = log.shift(1).rolling(window, min_periods=min_history)
            mad = history.apply(lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=True)
            monthly[feature + "_reference_scale"] = 1.4826 * mad
            monthly[feature] = (log - history.median()) / (1.4826 * mad).clip(lower=scale_floor)
        parts.append(monthly.loc[original].rename_axis("month").reset_index())
    result = pd.concat(parts, ignore_index=True)
    result["eligible"] = np.isfinite(result[FEATURES]).all(axis=1)
    result["robust_score"] = result[FEATURES].abs().max(axis=1).where(result.eligible)
    result["rule_alert"] = result.robust_score.ge(4)
    result["reason"] = result[FEATURES].abs().fillna(-np.inf).idxmax(axis=1)
    result.loc[~result.eligible, "reason"] = "insufficient_positive_unflagged_history"
    return result.sort_values(GRAIN).reset_index(drop=True)


def eligibility_reasons(features):
    positive = features[["value_eur", "net_mass_kg", "unit_value_eur_per_kg"]].gt(0).all(axis=1)
    flagged = features[["value_eur_status", "mass_100kg_status"]].fillna("").ne("").any(axis=1)
    # Mutually exclusive, priority-ordered reasons avoid double-counting limitations.
    reason = pd.Series("eligible", index=features.index)
    reason.loc[~features.eligible] = "insufficient historical reference"
    reason.loc[~positive] = "current missing/nonpositive measure"
    reason.loc[flagged] = "current source flag"
    return reason.value_counts().rename_axis("reason").reset_index(name="observations")


def fit_forest(features, cutoff="2023-12-01", seed=42):
    train = features.eligible & features.month.le(pd.Timestamp(cutoff))
    if train.sum() < 100:
        raise ValueError("Too few eligible training observations.")
    model = IsolationForest(
        n_estimators=150,
        max_samples=min(2048, int(train.sum())),
        random_state=seed,
        n_jobs=1,
        contamination="auto",
    )
    model.fit(features.loc[train, FEATURES])
    result = features.copy()
    result["iforest_score"] = np.nan
    result.loc[result.eligible, "iforest_score"] = -model.score_samples(
        result.loc[result.eligible, FEATURES]
    )
    training_scores = result.loc[train, "iforest_score"].to_numpy()
    return result, model, training_scores


def threshold_sensitivity(
    scored, training_scores, evaluation_start="2024-01-01", evaluation_end="2025-12-01"
):
    frame = scored.loc[scored.eligible & scored.month.between(evaluation_start, evaluation_end)]
    rows = []
    for quantile in [0.975, 0.99, 0.995]:
        threshold = float(np.quantile(training_scores, quantile))
        selected = frame.iforest_score.gt(threshold)
        rows.append(
            {
                "training_quantile": quantile,
                "threshold": threshold,
                "eligible_rows": len(frame),
                "flagged_rows": int(selected.sum()),
                "flag_rate": float(selected.mean()),
            }
        )
    return pd.DataFrame(rows)


def compare_review_queues(scored, budget, start="2024-01-01", end="2025-12-01"):
    """Compare rankings on the same eligible rows with the same number of review slots."""
    frame = (
        scored.loc[scored.eligible & scored.month.between(start, end)].copy().reset_index(drop=True)
    )
    if not isinstance(budget, int) or isinstance(budget, bool) or not 0 < budget <= len(frame):
        raise ValueError("Review budget must be a positive integer within the eligible pool.")
    if frame.duplicated(GRAIN).any():
        raise ValueError("Duplicate observation in review pool.")
    if not np.isfinite(frame[["iforest_score", "robust_score"]]).all().all():
        raise ValueError("Review rankings require finite scores.")
    for score, rank in [("iforest_score", "forest_rank"), ("robust_score", "rule_rank")]:
        # Resolve score ties by observation key, so input order cannot change the queue.
        order = frame.sort_values([score, *GRAIN], ascending=[False, *[True] * len(GRAIN)]).index
        frame.loc[order, rank] = np.arange(1, len(frame) + 1)
        frame[rank] = frame[rank].astype(int)
    forest, rule = frame.forest_rank.le(budget), frame.rule_rank.le(budget)
    frame["queue_group"] = np.select(
        [forest & rule, forest & ~rule, rule & ~forest],
        ["both", "forest only", "rule only"],
        default="neither",
    )
    return frame.sort_values(GRAIN).reset_index(drop=True)
