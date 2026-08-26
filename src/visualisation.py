"""
src.visualisation
------------------------------------------------------------------------
Matplotlib styling, colour palettes, plotting helpers, and preparation of
trajectories for viewing.
"""


import warnings
from typing import Literal
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import PercentFormatter
import MDAnalysis as mda
from MDAnalysis import transformations
from MDAnalysis.core.groups import AtomGroup

from src import utils


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

# Yellow family — not an official KU hue, tuned to sit alongside champagne/red
dark_yellow  = "#B8860F"
yellow       = "#FFBD38"
light_yellow = "#FCE4B0"

# Purple family — not an official KU hue, matched to the petroleum/green
# triads' saturation and value so it reads as part of the same set
dark_purple  = "#4B2142"
purple       = "#7C3F76"
light_purple = "#D9C3D6"

# Orange family — not an official KU hue, a burnt-sienna companion to
# ku_red/yellow rather than a bright, off-brand orange
dark_orange  = "#8A4A1E"
orange       = "#C97A2E"
light_orange = "#F0D3A8"

#······································································#

# Primary palette — 10 maximally distinct hues for categorical data
palette = [
    ku_red,
    dark_blue,
    petroleum,
    dark_green,
    yellow,
    purple,
    orange,
    red,
    blue,
    green,
    dark_petroleum,
    light_red,
    light_blue,
    light_green,
    light_petroleum,
    dark_gray,
    ku_gray,
    light_gray,
]

# Muted palette — desaturated darks, suitable for academic papers
muted = [
    dark_red,
    dark_blue,
    dark_petroleum,
    dark_green,
    dark_purple,
    dark_orange,
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
    purple,
    orange,
    light_red,
]

# Monochrome sequences
reds    = [dark_red, red, light_red]
blues   = [dark_blue, blue, light_blue]
purples = [dark_purple, purple, light_purple]
oranges = [dark_orange, orange, light_orange]



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

    purple_cmap = mcolors.LinearSegmentedColormap.from_list(
        "purple_scale",
        [white, light_purple, purple, dark_purple]
    )

    orange_cmap = mcolors.LinearSegmentedColormap.from_list(
        "orange_scale",
        [white, light_orange, orange, dark_orange]
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
    'axes.unicode_minus': False,  # Use ASCII hyphen (CMU Serif lacks U+2212)

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

def format_percent_axis(which: Literal['x', 'y']) -> None:
    """Format the current axes' ticks as percentages.

    Assumes the plotted values lie in ``[0, 1]``. Operates on the current
    axes (``plt.gca()``).

    Parameters
    ----------
    which : Literal['x', 'y']
        Which axis to format.
    """
    if which.lower() == 'x':
        plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
    elif which.lower() == 'y':
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))



#······································································#
# Trajectories
#······································································#

# CALVADOS runs in a periodic box and writes raw coordinates: a chain that
# drifts over a box face comes back on the opposite side, so a trajectory
# opened as-is shows chains split in half and complexes that appear
# dissociated whenever the pair happens to straddle a boundary. On top of
# that, nothing stops the whole complex from tumbling, so a viewer shows the
# binding site rolling in and out of sight.
#
# Anchoring on a chosen group of beads fixes all of it. Every chain is first
# made whole again (``unwrap`` undoes the wrapping along the chain's own
# connectivity); the box is then rigid-body fitted so the anchor beads sit at
# the box centre in a fixed orientation (``fit_rot_trans``, i.e. the rotation
# and translation that best superimpose them on their first-frame reference)
# or merely translated there (``center_in_box``) when the rotation is not
# wanted; finally the remaining chains are re-wrapped as whole molecules, so
# each is placed in the periodic image nearest to that centre. What is left is
# a trajectory in which the anchor neither moves nor turns, and its binding
# partners stay next to it.
#
# The natural anchor for viewing a complex is the acceptor's contact patch
# (:func:`acceptor_contacts`): the binding site then stays put and centred, and
# everything that moves on screen, the rest of the acceptor included, moves
# relative to it.

def _add_sequential_bonds(universe: mda.Universe) -> None:
    """Bond consecutive beads within each chain of a coarse-grained topology.

    ``unwrap`` needs bonds to know which beads belong to the same molecule, but
    CALVADOS' topology PDB carries no ``CONECT`` records. With one bead per
    residue and beads written in sequence order, the connectivity is simply
    ``i -- i+1`` within each chain, which makes every chain exactly one
    fragment.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Coarse-grained universe to add a ``bonds`` topology attribute to,
        in place.

    Raises
    ------
    ValueError
        If any chain does not hold exactly one bead per residue, in which case
        consecutive atoms are not consecutive residues and the sequential
        connectivity assumed here would be wrong.
    """
    bonds = []
    for segment in universe.segments:
        atoms = segment.atoms
        if atoms.n_atoms != segment.residues.n_residues:
            raise ValueError(
                f"Chain '{segment.segid}' is not one bead per residue "
                f"({atoms.n_atoms} beads, {segment.residues.n_residues} residues), "
                "so sequential connectivity cannot be assumed. Provide a topology "
                "with explicit bonds instead."
            )
        indices = atoms.indices
        bonds.extend(zip(indices[:-1], indices[1:]))
    universe.add_TopologyAttr('bonds', bonds)

#······································································#

def acceptor_contacts(
        universe: mda.Universe,
        specific_interactions: str,
    ) -> AtomGroup:
    """Select the acceptor beads that carry a Gō contact.

    The beads listed on the left-hand side of ``specific_interactions.txt``,
    i.e. the acceptor's half of the binding interface. Following the run
    layout, the acceptor is the first chain of the topology and its bead
    indices in the file are 1-based within that chain.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Universe of the run, one bead per residue.
    specific_interactions : str
        Path to the run's ``specific_interactions.txt``.

    Returns
    -------
    MDAnalysis.AtomGroup
        The distinct contacting acceptor beads, in topology order.

    Raises
    ------
    ValueError
        If the acceptor chain is not one bead per residue, or if a listed bead
        index falls outside it.
    """
    contacts = utils.parse_specific_interactions(specific_interactions)
    acceptor = universe.segments[0].atoms
    if acceptor.n_atoms != universe.segments[0].residues.n_residues:
        raise ValueError(
            f"Acceptor chain '{universe.segments[0].segid}' is not one bead per "
            f"residue, so the bead indices in '{specific_interactions}' cannot be "
            "resolved."
        )
    residues = np.unique(contacts['Acceptor residue'].to_numpy(dtype=int))
    if residues.min() < 1 or residues.max() > acceptor.n_atoms:
        raise ValueError(
            f"'{specific_interactions}' lists acceptor beads "
            f"{residues.min()}-{residues.max()}, outside the acceptor chain's "
            f"1-{acceptor.n_atoms} beads. Are topology and contacts from the same run?"
        )
    return acceptor[residues - 1]

#······································································#

def _anchor(
        universe: mda.Universe,
        anchor: AtomGroup,
        orient: bool,
        center: Literal['geometry', 'mass'],
    ) -> mda.Universe:
    """Attach the unwrap/fit/wrap transformations that pin ``anchor`` in place.

    The transformations are registered on the trajectory rather than applied,
    so they cost nothing until frames are read and no coordinates are held in
    memory: every frame yielded afterwards (by iteration, indexing, an analysis
    class or a writer) comes out anchored.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Universe to transform, in place. Must carry bonds (see
        :func:`_add_sequential_bonds`).
    anchor : MDAnalysis.AtomGroup
        Beads to hold at the box centre.
    orient : bool
        If ``True``, also hold the anchor's orientation fixed by rigid-body
        fitting each frame onto the anchor's first-frame geometry; if
        ``False``, only translate.
    center : Literal['geometry', 'mass']
        Whether the anchor's geometric or mass centre is placed at the box
        centre (and, when ``orient``, weights the fit).

    Returns
    -------
    MDAnalysis.Universe
        The same universe, with the transformations attached and rewound.
    """
    if orient:
        # The orientation reference is the anchor as it stands in the first
        # frame, made whole and shifted to the box centre; fitting every frame
        # onto it both centres and de-rotates the anchor.
        universe.trajectory[0]
        universe.atoms.unwrap(compound='fragments', inplace=True)
        reference = mda.Merge(universe.atoms).atoms[anchor.indices]
        weights = reference.masses if center == 'mass' else None
        reference.universe.atoms.translate(
            universe.dimensions[:3] / 2 - reference.center(weights))
        fit = transformations.fit_rot_trans(anchor, reference, weights=weights)
    else:
        fit = transformations.center_in_box(anchor, center=center)

    universe.trajectory.add_transformations(
        transformations.unwrap(universe.atoms),
        fit,
        transformations.wrap(universe.atoms, compound='fragments'),
    )
    universe.trajectory[0]  # re-read frame 0 through the transformations
    return universe

#······································································#

def _load(topology: str, trajectory: str | list[str]) -> mda.Universe:
    """Load a run's topology and trajectory, with chain connectivity."""
    with warnings.catch_warnings():
        # The DCDReader emits a 3.0-deprecation notice about making independent
        # timesteps; irrelevant here, as each frame is transformed as it is read.
        warnings.filterwarnings(
            'ignore', message='DCDReader currently makes independent timesteps')
        universe = mda.Universe(topology, trajectory)
    if not hasattr(universe.atoms, 'bonds'):
        _add_sequential_bonds(universe)
    return universe

#······································································#

def center_on_contacts(
        topology: str,
        trajectory: str | list[str],
        specific_interactions: str,
        orient: bool = True,
        center: Literal['geometry', 'mass'] = 'geometry',
    ) -> mda.Universe:
    """Load a trajectory anchored on the acceptor's contact patch.

    The binding site (:func:`acceptor_contacts`) is held at the box centre and,
    with ``orient``, at a fixed orientation, so the view stays on the interface
    while the donor, and the acceptor's own flexible regions, move against it.
    Chains are kept whole across the periodic boundaries throughout.

    Parameters
    ----------
    topology : str
        Path to the topology PDB (e.g. a run's ``top.pdb``).
    trajectory : str or list[str]
        Path to the trajectory DCD, or several to read as one continuous
        trajectory.
    specific_interactions : str
        Path to the run's ``specific_interactions.txt``, defining the contacts.
    orient : bool, optional
        Hold the contact patch's orientation fixed as well as its position, by
        default ``True``. The rotation is the one that best superimposes the
        patch on its first-frame geometry, so a rigid patch (as a Gō-restrained
        interface is) pins all three axes and the acceptor no longer tumbles.
    center : Literal['geometry', 'mass'], optional
        Which patch centre to place at the box centre, by default
        ``"geometry"``. Prefer the default: CALVADOS writes every bead as a
        carbon ``CA``, so the masses an MDAnalysis topology parses from the PDB
        are uniform placeholders rather than residue masses.

    Returns
    -------
    MDAnalysis.Universe
        The universe, with the anchoring transformations attached.

    Raises
    ------
    ValueError
        If the topology and the contact file are inconsistent (see
        :func:`acceptor_contacts`) or the topology is not one bead per residue.

    Examples
    --------
    >>> run = 'sim/1dp-idp/mdm2-p53/eps_go=3.00/visualisation/rep1'
    >>> u = center_on_contacts(f'{run}/top.pdb', f'{run}/mdm2-p53.dcd',
    ...                        f'{run}/specific_interactions.txt')
    """
    universe = _load(topology, trajectory)
    return _anchor(universe, acceptor_contacts(universe, specific_interactions),
                   orient, center)

#······································································#

def center_on_chain(
        topology: str,
        trajectory: str | list[str],
        chain_id: str,
        orient: bool = False,
        center: Literal['geometry', 'mass'] = 'geometry',
    ) -> mda.Universe:
    """Load a trajectory anchored on one whole chain.

    As :func:`center_on_contacts`, but with the entire chain as the anchor --
    the coarser view, for when the interest is the chain as a whole rather than
    its binding site.

    Parameters
    ----------
    topology : str
        Path to the topology PDB (e.g. a run's ``top.pdb``).
    trajectory : str or list[str]
        Path to the trajectory DCD, or several to read as one continuous
        trajectory.
    chain_id : str
        Chain id to anchor on, as written in the topology (e.g. ``"A"``).
    orient : bool, optional
        Hold the chain's orientation fixed as well as its position, by default
        ``False``. A fit over a whole chain is dominated by its flexible parts,
        which makes the resulting orientation wander; anchor on the contact
        patch instead when a steady view of the interface is what is wanted.
    center : Literal['geometry', 'mass'], optional
        Which chain centre to place at the box centre, by default
        ``"geometry"`` (see :func:`center_on_contacts`).

    Returns
    -------
    MDAnalysis.Universe
        The universe, with the anchoring transformations attached.

    Raises
    ------
    KeyError
        If ``chain_id`` is not found in the topology.
    ValueError
        If bonds are absent and the topology is not one bead per residue (see
        :func:`_add_sequential_bonds`).
    """
    universe = _load(topology, trajectory)
    return _anchor(universe, utils.select_chain(universe, chain_id), orient, center)

#······································································#

def write_centered_trajectory(
        universe: mda.Universe,
        output: str,
        stride: int = 1,
    ) -> str:
    """Write a transformed universe's coordinates to a new trajectory file.

    Materialises the on-the-fly transformations of :func:`center_on_contacts` /
    :func:`center_on_chain` frame by frame, for viewers (ChimeraX, VMD, PyMOL)
    that read coordinates straight from the file and would otherwise show the
    raw, wrapped ones. Topologies are untouched by the transformations, so the
    run's ``top.pdb`` remains the file to open the output with.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Universe to write, typically one returned by :func:`center_on_contacts`
        or :func:`center_on_chain`.
    output : str
        Path of the trajectory to write. The format follows the extension
        (``.dcd``, ``.xtc``, ...).
    stride : int, optional
        Keep every ``stride``-th frame, by default ``1``. Applied at the reader
        level, so skipped frames are never transformed or written.

    Returns
    -------
    str
        The ``output`` path.
    """
    frames = universe.trajectory[::stride] if stride != 1 else universe.trajectory
    with mda.Writer(output, universe.atoms.n_atoms) as writer:
        for _ in tqdm(frames):
            writer.write(universe.atoms)
    return output

#······································································#

def chain_ids(topology: str) -> list[str]:
    """List the chain ids of a topology, in file order.

    Convenience for picking the ``chain_id`` to anchor on without opening the
    PDB by hand.

    Parameters
    ----------
    topology : str
        Path to the topology PDB.

    Returns
    -------
    list[str]
        The chain ids, one per chain.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return [segment.segid for segment in mda.Universe(topology).segments]
