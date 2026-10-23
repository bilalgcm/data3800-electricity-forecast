"""Shared Matplotlib style so every figure in the report looks the same."""
import matplotlib

matplotlib.use("Agg")  # save to file, no window needed
import matplotlib.pyplot as plt

import config

# Colours: one fixed colour per meaning, used the same way in every figure.
ACTUAL = "#333333"      # measured load
BLUE = "#2a78d6"        # our data / our model (random forest)
ORANGE = "#eb6834"      # second series (ENTSO-E forecast, weekend)
TEXT = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.figsize": (9, 4.5),
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelcolor": TEXT,
    "axes.edgecolor": GRID,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "legend.frameon": False,
    "lines.linewidth": 2,
})


def save(fig, name):
    """Save a figure as PNG in figures/ and close it."""
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path.relative_to(config.ROOT)}")
