"""Consistent plots; notebook cells still state the question and interpretation."""

import matplotlib.pyplot as plt

COLORS = ["#246b5c", "#cd8844", "#567897", "#885f84", "#607264", "#bc5c50"]


MODEL_LABELS = {
    "latest_known": "Latest available month",
    "seasonal_naive": "Same month last year",
    "ridge_raw_seasonal_a10": "Ridge + season",
    "ridge_raw_lags_a10": "Ridge, past values only",
    "ridge_log_seasonal_a10": "Log Ridge + season",
}
PRODUCT_LABELS = {"87113090": "380-500 cc", "87114000": "500-800 cc", "87115000": "Above 800 cc"}


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


def short_label(label):
    """Hide historical footnotes in chart labels; preserve source labels in the data."""
    return label.split(" (")[0]
