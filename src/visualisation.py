"""
src.visualization
-----------------
Utils and settings for plots.
"""

############################################################
# Imports
############################################################

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import PercentFormatter
from typing import Literal


############################################################
# Color Definitions
# Source: KU-farvepalet CMYK (designguide.ku.dk)
# Converted via R/G/B = 255 × (1−channel) × (1−K)
############################################################

# Primary KU brand colors
ku_red  = "#A8003B"   # KU-rød   CMYK 0/100/65/34
ku_gray = "#4D4D4D"   # KU-grå   CMYK 0/0/0/70

# Red family
dark_red   = "#A8003B"  # CMYK  0/100/65/34
red        = "#CE0C0C"  # CMYK 15/95/95/5
light_red  = "#F22626"  # CMYK  5/85/85/0

# Blue family
dark_blue  = "#001F5C"  # CMYK 100/80/40/40
blue       = "#2B578D"  # CMYK  80/60/35/15
light_blue = "#BFD9F2"  # CMYK  25/15/5/0

# Petroleum (teal) family
dark_petroleum  = "#136060"  # CMYK 90/50/50/25
petroleum       = "#309E9E"  # CMYK 80/35/35/5
light_petroleum = "#BFF2E6"  # CMYK 25/5/10/0

# Green family
dark_green  = "#2D6B00"  # CMYK 75/40/100/30
green       = "#45AC00"  # CMYK 70/25/100/10
light_green = "#BFE6A6"  # CMYK 25/10/35/0

# Gray family
dark_gray  = "#262626"  # CMYK 0/0/0/85
gray       = "#4D4D4D"  # CMYK 0/0/0/70
light_gray = "#D9D9D9"  # CMYK 0/0/0/15

# Neutrals
white      = "#FFFFFF"  # CMYK 0/0/0/0
champagne  = "#FFF2E6"  # CMYK 0/5/10/0
yellow     = "#FFB31A"  # CMYK 0/30/90/0


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Primary palette — 6 maximally distinct KU hues for categorical data
palette = [
    ku_red,
    dark_blue,
    petroleum,
    dark_green,
    yellow,
    dark_gray,
]

# Muted palette — desaturated darks, suitable for academic papers
muted = [
    dark_red,
    dark_blue,
    dark_petroleum,
    dark_green,
    dark_gray,
    champagne,
]

# Vibrant palette — mid-tone hues with high contrast, for presentations
vibrant = [
    red,
    blue,
    petroleum,
    green,
    yellow,
    light_red,
]

# Monochrome sequences
reds  = [dark_red, red, light_red]
blues = [dark_blue, blue, light_blue]


############################################################
# Custom colormaps
############################################################

class cmap:
    """Container of custom colormaps."""

    red_cmap = mcolors.LinearSegmentedColormap.from_list(
        "red_scale",
        [white, light_red, red, dark_red]
    )

    blue_cmap = mcolors.LinearSegmentedColormap.from_list(
        "blue_scale",
        [white, light_blue, blue, dark_blue]
    )

    green_cmap = mcolors.LinearSegmentedColormap.from_list(
        "green_scale",
        [white, light_green, green, dark_green]
    )

    petroleum_cmap = mcolors.LinearSegmentedColormap.from_list(
        "petroleum_scale",
        [white, light_petroleum, petroleum, dark_petroleum]
    )

    # Blue → white → red: good for correlation matrices and signed values
    diverging_cmap = mcolors.LinearSegmentedColormap.from_list(
        "diverging",
        [dark_blue, white, dark_red]
    )

    # Cool-to-warm heat: emphasises extremes at both ends
    heat_cmap = mcolors.LinearSegmentedColormap.from_list(
        "heat",
        [dark_blue, light_blue, white, light_red, dark_red]
    )

    categorical_cmap = mcolors.ListedColormap(palette)


############################################################
# Applied Global Settings
############################################################

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['CMU Serif'],
    'axes.titleweight': 'bold',
    'axes.prop_cycle': plt.cycler(color=palette),
})


############################################################
# Plotting
############################################################

def percent_axis(which: Literal['x', 'y']):
    """Formats an axis to have ticks in percent.

    Assumes values in [0,1].

    Parameters:
    -----------
    which : Literal['x', 'y']
        Which axis to format.
    """
    if which.lower() == 'x':
        plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
    elif which.lower() == 'y':
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
