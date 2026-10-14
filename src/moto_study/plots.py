"""Consistent plots; notebook cells still state the question and interpretation."""

import matplotlib.pyplot as plt

COLORS = ["#246b5c", "#cd8844", "#567897", "#885f84", "#607264", "#bc5c50"]


MODEL_LABELS = {
    "last_month": "Previous month",
    "seasonal_naive": "Same month last year",
    "ridge_raw_seasonal_a10": "Ridge: raw EUR + season, alpha 10",
    "ridge_log_lags_a10": "Ridge: log EUR lags, alpha 10",
    "ridge_log_seasonal_a1": "Ridge: log EUR + season, alpha 1",
    "ridge_log_seasonal_a10": "Ridge: log EUR + season, alpha 10",
    "ridge_log_seasonal_a100": "Ridge: log EUR + season, alpha 100",
}


def set_style():
    plt.rcParams.update(
        {
            "figure.figsize": (10, 4.5),
            "figure.dpi": 115,
            "savefig.dpi": 160,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlelocation": "left",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "font.size": 10,
            "legend.frameon": False,
            "axes.prop_cycle": plt.cycler(color=COLORS),
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.18,
            "axes.axisbelow": True,
        }
    )
