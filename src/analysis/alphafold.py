"""
src.analysis.alphafold
------------------------------------------------------------------------
For analysis of AlphaFold3-generated structures.
"""

import os
import numpy as np
import json
from typing import Literal
from scipy.spatial import distance_matrix

from src import utils
from src.analysis import coarsegrain


#······································································#
# Inter-chain residue-level analysis
#······································································#

def inter_chain_analysis(
        structure_cif: os.PathLike,
        full_data_json: os.PathLike,
        chain1_id: str,
        chain2_id: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Takes the output of an AlphaFold invocation,
    returns the inter-residue PAE, P_{inc}, and minimum all atom distance matrices
    between two specified chains.

    Parameters
    ----------
    structure_cif : Path
        The path to an AlphaFold structure CIF-file
    full_data_json : Path
        The path to an AlphaFold "``*_full_data_*.json``"-file
    chain1_id : str
        The chain ID assigned to the first chain of interest in the AlphaFold structure
    chain2_id : str
        The chain ID assigned to the second chain of interest in the AlphaFold structure

    Returns
    -------
    tuple[np.ndarray]
        A tuple of the following inter-chain residue-level matrices:

        * PAE
        * P_inc
        * Minimum all-atom distance


    """
    return (
        inter_chain_pae(full_data_json, chain1_id, chain2_id),
        inter_chain_pinc(structure_cif, full_data_json, chain1_id, chain2_id),
        inter_chain_min_dist(structure_cif, chain1_id, chain2_id),
    )



#······································································#
# Individual analyses
#······································································#

def inter_chain_pae(
        full_data_json: os.PathLike,
        chain1_id: str,
        chain2_id: str,
        perspective: Literal['symmetric', 'chain1', 'chain2']='symmetric'
    ) -> np.ndarray:
    """Loads an AlphaFold PAE matrix from an AlphaFold "``*_full_data_*.json``"-file,
    returns the PAE-matrix
    between the residues of two specified chains by their chain ID.

    Parameters
    ----------
    full_data_json : Path
        The path to an AlphaFold "``*_full_data_*.json``"-file
    chain1_id : str
        The chain ID assigned to the first chain of interest in the AlphaFold structure
    chain2_id : str
        The chain ID assigned to the second chain of interest in the AlphaFold structure
    perspective : Literal['symmetric', 'chain1', 'chain2'] = 'symmetric'
        The PAE-matrix between two chains has two entries depending on the chain perspective;
        This parameter decides whether to return the value from the perspective of either chain
        ('chain1', 'chain2') or whether to return their average ('symmetric').

    Returns
    -------
    res_pae: np.ndarray
        A inter-chain PAE matrix between the specified chains.
    """

    # Loading PAE data
    with open(full_data_json, 'r') as file:
        full_data = json.load(file)
    res_pae = np.array(full_data['pae'])
    res_chain = np.array(full_data['token_chain_ids'])

    # Determining chain masks
    chain1_mask = res_chain == chain1_id
    chain2_mask = res_chain == chain2_id

    # Determining PAE perspective
    if perspective == 'symmetric':
        sym_res_pae = (res_pae + res_pae.T) / 2
        return sym_res_pae[chain1_mask, :][:, chain2_mask]
    elif perspective == 'chain1':
        return res_pae[chain1_mask, :][:, chain2_mask]
    elif perspective == 'chain2':
        return res_pae[chain2_mask, :][:, chain1_mask].T
    
#······································································#

def inter_chain_min_dist(
        structure_cif: os.PathLike,
        chain1_id: str,
        chain2_id: str,
    ) -> np.ndarray:
    """Loads an AlphaFold structure CIF-file,
    returns the a minimum all-atom inter-residue distance matrix
    between the residues of two specified chains by their chain ID.

    Parameters
    ----------
    structure_cif : Path
        The path to an AlphaFold structure CIF-file
    chain1_id : str
        The chain ID assigned to the first chain of interest in the AlphaFold structure
    chain2_id : str
        The chain ID assigned to the second chain of interest in the AlphaFold structure

    Returns
    -------
    res_min_dist: np.ndarray
        A minimum all-atom inter-residue distance matrix between the specified chains.
    """
    
    # Loading and subsetting chains
    universe = utils.load_structure(structure_cif)
    chain1 = utils.select_chain(universe, chain1_id).residues
    chain2 = utils.select_chain(universe, chain2_id).residues

    # Per-residue heavy-atom coordinates (hydrogens excluded)
    def _heavy_coords(residues):
        return [res.atoms.select_atoms('not name H*').positions for res in residues]
    res_coords_1 = _heavy_coords(chain1)
    res_coords_2 = _heavy_coords(chain2)

    res_min_dist = np.zeros((len(res_coords_1), len(res_coords_2)))
    for i, rc1 in enumerate(res_coords_1):
        for j, rc2 in enumerate(res_coords_2):
            res_min_dist[i, j] = np.min(distance_matrix(rc1, rc2))
    
    return res_min_dist

#······································································#

def inter_chain_pinc(
        structure_cif: os.PathLike,
        full_data_json: os.PathLike,
        chain1_id: str,
        chain2_id: str,
        r_c: float = 10.0,
        coordinate_mode: Literal['CA', 'COM'] = 'CA'
) -> np.ndarray:
    """Compute the inter-chain conditional contact-probability matrix (P_inc).
 
    For every ordered residue pair drawn from two chains of an AlphaFold model,
    this returns the probability that the pair is in contact (centre-to-centre
    distance below ``r_c``) given the positional uncertainty encoded in the
    Predicted Aligned Error (PAE). It implements the geometric sphere-overlap
    model of Pellegrini et al. [1]_, which underlies the Pinc interface score of
    Toth-Petroczy and Badonyi [2]_.
 
    Parameters
    ----------
    structure_cif : os.PathLike
        The path to an AlphaFold structure CIF-file
    full_data_json : os.PathLike
        The path to an AlphaFold "``*_full_data_*.json``"-file
    chain1_id : str
        The chain ID assigned to the first chain of interest in the AlphaFold structure
    chain2_id : str
        The chain ID assigned to the second chain of interest in the AlphaFold structure
    r_c : float, optional
        Radius of the contact sphere ``S_C`` in Angstrom, by default 10.0
        (Pellegrini et al. [1]_); The Pinc preprint [2]_ uses 12 Angstrom as its
        default contact radius
    coordinate_mode : Literal['CA', 'COM'], optional
        Per-residue representation: ``'CA'`` uses the alpha-carbon position
        (default; matches the original formulation), ``'COM'`` uses the residue
        centre of mass.
 
    Returns
    -------
    np.ndarray
        Float array of shape ``(N1, N2)`` with values of inter-residue Pinc-scores clipped to ``[0, 1]``,
        where ``N1`` and ``N2`` are the residue counts of ``chain1_id``/``chain2_id``.
        Entry ``[i, j]`` is the conditional probability that residue ``i`` of
        chain 1 and residue ``j`` of chain 2 are in contact.
    
    Model
    -----
    Within the PAE framework the *scored* residue is the one whose position is
    being evaluated and the *aligned* residue is the reference assumed to be
    correctly placed [2]_. Two spheres are defined per ordered pair:
 
    - ``S_C``: a contact sphere of radius ``r_c`` centred on the reference
      residue. The 10 Angstrom default approximates the spatial extent of a
      residue's side chain and thus a sensible proximity cutoff [1]_.
    - ``S_R``: an uncertainty sphere of radius equal to the PAE of the pair,
      centred on the scored residue.
 
    Assuming the true position of the scored residue is uniformly distributed
    over ``S_R``, the conditional contact probability is the fraction of ``S_R``
    that falls inside ``S_C``:
 
    .. math:: P_\\mathrm{inc} = \\frac{\\mathrm{vol}(S_R \\cap S_C)}{\\mathrm{vol}(S_R)}.
 
    The intersection volume is the standard two-sphere lens
 
    .. math::
        \\mathrm{vol}(S_R \\cap S_C) =
        \\frac{\\pi (r_c + r_R - d)^2
        \\,[\\,d^2 + 2d(r_c + r_R) - 3(r_c - r_R)^2\\,]}{12\\,d},
 
    evaluated only in the partial-overlap regime ``|r_c - r_R| < d < r_c + r_R``;
    full-containment and disjoint cases are handled in closed form. Because PAE
    is asymmetric, ``P_inc`` is asymmetric: ``P_inc[i, j] != P_inc[j, i]`` in
    general, and the orientation is fixed by ``perspective`` in the PAE lookup.
 
 
    References
    ----------
    .. [1] Pellegrini S., Dove-Estrella O., Muinos F., Lopez-Bigas N.,
       Gonzalez-Perez A. (2025) Oncodrive3D: fast and accurate detection of
       structural clusters of somatic mutations under positive selection.
       Nucleic Acids Research, 53(15), gkaf776.
       https://doi.org/10.1093/nar/gkaf776
    .. [2] Toth-Petroczy A., Badonyi M. (2026) Pinc: a simple probabilistic
       AlphaFold interaction score. bioRxiv 2026.03.02.708997.
       https://doi.org/10.64898/2026.03.02.708997
    """


    # Loading and subsetting chains
    universe = utils.load_structure(structure_cif)
    chain1 = utils.select_chain(universe, chain1_id).residues
    chain2 = utils.select_chain(universe, chain2_id).residues

    # Getting residue representations
    if coordinate_mode == 'CA':
        res_coords_1 = coarsegrain.get_ca_coords(chain1)
        res_coords_2 = coarsegrain.get_ca_coords(chain2)
    else:  # 'COM'
        res_coords_1 = coarsegrain.get_com_coords(chain1)
        res_coords_2 = coarsegrain.get_com_coords(chain2)
    
    # Loading PAE data
    res_pae = inter_chain_pae(full_data_json, chain1_id, chain2_id, perspective='chain1')
    
    # Asserting structure representation and PAE tokens match in length
    assert res_coords_1.shape[0] == res_pae.shape[0]
    assert res_coords_2.shape[0] == res_pae.shape[1]
    
    # Computing pairwise distances between residues
    diff = res_coords_1[:, None, :] - res_coords_2[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    
    # Mapping parameters to the intersection geometry model
    r_R = np.maximum(res_pae, 1e-6)  # Prevent division by zero
    r_C = np.full_like(d, r_c)
    
    # Calculating probabilities across geometric regimes
    # Case 1: No intersection at all
    res_pinc = np.zeros_like(d)
    
    # Case 2a: Uncertainty sphere is completely inside contact sphere
    submerged_R_in_C = (d <= (r_C - r_R)) & (r_R <= r_C)
    res_pinc[submerged_R_in_C] = 1.0
    
    # Case 2b: Contact sphere is completely inside uncertainty sphere
    submerged_C_in_R = (d <= (r_R - r_C)) & (r_R > r_C)
    res_pinc[submerged_C_in_R] = (r_C[submerged_C_in_R] ** 3) / (r_R[submerged_C_in_R] ** 3)
    
    # Case 3: Partial Intersection
    intersect_mask = (d > np.abs(r_C - r_R)) & (d < (r_C + r_R))
    d_i = d[intersect_mask]
    r_C_i = r_C[intersect_mask]
    r_R_i = r_R[intersect_mask]
    vol_intersect = (np.pi * (r_C_i + r_R_i - d_i) ** 2 * (d_i ** 2 + 2 * d_i * (r_C_i + r_R_i) - 3 * (r_C_i - r_R_i) ** 2)) / (12.0 * d_i)
    vol_V_R = (4.0 / 3.0) * np.pi * (r_R_i ** 3)
    res_pinc[intersect_mask] = vol_intersect / vol_V_R
    
    return np.clip(res_pinc, 0.0, 1.0)
