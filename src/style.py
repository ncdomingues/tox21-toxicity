"""Shared chart style: recessive axes and grid, a fixed categorical colour order."""
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#b9b8b2"

# Categorical slots, always assigned in this order (never cycled)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
# Diverging poles (below / above expectation), neutral midpoint grey
DIV_NEG, DIV_POS, DIV_MID = "#e34948", "#2a78d6", "#f0efec"


ACTIVITY_COLORS = {"active": ORANGE, "inactive": BLUE}


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.figsize": (8, 4.5), "figure.dpi": 100,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": NEUTRAL, "axes.labelcolor": TEXT_2, "text.color": TEXT,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "legend.frameon": False, "lines.linewidth": 2,
    })
