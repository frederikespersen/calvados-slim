"""
src.visualization
------------------------------------------------------------------------
Utils and settings for plots.
"""


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import PercentFormatter
from typing import Literal


#······································································#
# Colors and Palettes
# Source: KU-farvepalet RGB (designguide.ku.dk)
#······································································#

# Primary KU brand colors
ku_red  = "#901A1E"   # Official UCPH Red (RGB 144/26/30)
ku_gray = "#666666"   # Official UCPH Gray (RGB 102/102/102)

# Red family
dark_red   = "#901A1E"  # Same as ku_red
red        = "#C73028"  
light_red  = "#DB3B0A"  

# Blue family
dark_blue  = "#122947"  
blue       = "#425570"  
light_blue = "#BAC7D9"  

# Petroleum (teal) family
dark_petroleum  = "#0A5963"  
petroleum       = "#197F8E"  
light_petroleum = "#B7D7DE"  

# Green family
dark_green  = "#39641C"  
green       = "#4B8325"  
light_green = "#BECAA8"  

# Gray family
dark_gray  = "#3D3D3D"  
gray       = "#666666"  # Same as ku_gray
light_gray = "#E1DFDF"  

# Neutrals
white      = "#FFFFFF"  
champagne  = "#FEFAF2"  
yellow     = "#FFBD38"


#······································································#

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



#······································································#
# Custom colormaps
#······································································#

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



#······································································#
# Applied Global Settings
#······································································#

plt.rcParams.update({
    # Fonts
    'font.family': 'serif',
    'font.serif': ['CMU Serif'],
    'mathtext.fontset': 'cm',

    # Text
    'axes.titleweight': 'bold',

    # Colour
    'axes.prop_cycle': plt.cycler(color=palette),

    # Resolution
    'figure.dpi': 300,

    # Splines
    'axes.spines.top': False,
    'axes.spines.right': False,

    # Series
    'lines.linewidth': 1,
    'lines.markersize': 1,
})



#······································································#
# Plotting
#······································································#

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
