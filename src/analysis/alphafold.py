"""
src.analysis.alphafold
------------------------------------------------------------------------
For analysis of AlphaFold3-generated structures.
"""


from typing import Literal
import numpy as np
import json
from Bio.PDB import MMCIFParser
from scipy.spatial import distance_matrix



#······································································#
# Curated analysis
#······································································#

def inter_chain_analysis(
        structure_cif,
        full_data_json,
        chainid_1: str,
        chainid_2: str
    ):
    return (
        inter_pae(full_data_json, chainid_1, chainid_2),
        inter_pinc(structure_cif, full_data_json, chainid_1, chainid_2),
        min_inter_res_dist(structure_cif, chainid_1, chainid_2),
        avg_inter_res_plddt(structure_cif, chainid_1, chainid_2),
    )


#······································································#
# Individual analyses
#······································································#

def inter_pae(
        full_data_json,
        chainid_1: str,
        chainid_2: str,
        perspective: Literal['symmetric', 'chain1', 'chain2']='symmetric'
    ) -> np.ndarray:

    # Loading PAE data
    with open(full_data_json, 'r') as file:
        full_data = json.load(file)
    res_pae = np.array(full_data['pae'])
    res_chain = np.array(full_data['token_chain_ids'])

    # Determining chain masks
    chain1_mask = res_chain == chainid_1
    chain2_mask = res_chain == chainid_2

    # Determining PAE perspective
    if perspective == 'symmetric':
        sym_res_pae = (res_pae + res_pae.T) / 2
        return sym_res_pae[chain1_mask, :][:, chain2_mask]
    elif perspective == 'chain1':
        return res_pae[chain1_mask, :][:, chain2_mask]
    elif perspective == 'chain2':
        return res_pae[chain2_mask, :][:, chain1_mask].T
    
#······································································#
def min_inter_res_dist(
        structure_cif,
        chainid_1: str,
        chainid_2: str,
    ) -> np.ndarray:
    
    # Loading structure
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('afstruc', structure_cif)

    # Subsetting chains
    model = structure[0]
    chain_1 = model[chainid_1]
    chain_2 = model[chainid_2]

    # Getting residue heavy atom coordinates
    def get_res_coords(chain):
        res_coords = []
        for res in chain:
            atoms = [a for a in res if not a.get_name().startswith('H')]
            coords = np.array([a.get_coord() for a in atoms])
            res_coords.append(coords)
        return res_coords
    res_coords_1 = get_res_coords(chain_1)
    res_coords_2 = get_res_coords(chain_2)

    min_dist = np.zeros((len(res_coords_1), len(res_coords_2)))
    for i, rc1 in enumerate(res_coords_1):
        for j, rc2 in enumerate(res_coords_2):
            min_dist[i, j] = np.min(distance_matrix(rc1, rc2))
    
    return min_dist

#······································································#
def avg_inter_res_plddt(
        structure_cif,
        chainid_1: str,
        chainid_2: str,
    ) -> np.ndarray:
    
    # Loading structure
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('afstruc', structure_cif)

    # Subsetting chains
    model = structure[0]
    chain_1 = model[chainid_1]
    chain_2 = model[chainid_2]

    # Getting residue mean plddts
    def get_res_plddts(chain):
        res_plddts = []
        for res in chain:
            atoms = [a for a in res if not a.get_name().startswith('H')]
            avg_plddt = np.mean([a.get_bfactor() for a in atoms])
            res_plddts.append(avg_plddt)
        return res_plddts
    res_plddts_1 = np.array(get_res_plddts(chain_1))
    res_plddts_2 = np.array(get_res_plddts(chain_2))

    avg_plddts = (res_plddts_1[:,np.newaxis] + res_plddts_2[np.newaxis,:]) / 2
    
    return avg_plddts

#······································································#
def inter_pinc(
        structure_cif: str,
        full_data_json: str,
        chainid_1: str,
        chainid_2: str,
        r_c: float = 10.0,
        coordinate_mode: Literal['CA', 'COM'] = 'CA'
) -> np.ndarray:
    """
    Calculates the uncertainty-aware contact probability matrix between 
    residues of chain 1 (rows, R) and chain 2 (columns, C).
    
    Parameters:
        structure_cif: Path to the MMCIF structure file.
        full_data_json: Path to the AlphaFold3 summary JSON containing PAE data.
        chainid_1: String ID for the first chain (Residue R).
        chainid_2: String ID for the second chain (Center C).
        r_c: Radius threshold for physical contact in Angstroms (default: 10.0).
        coordinate_mode: Coordinate representation to use. Options are:
                         'CA'  : Use the C-alpha atom coordinate.
                         'COM' : Use the mass-weighted Center-of-Mass of the residue.
        
    Returns:
        prob_matrix: (N1, N2) array containing pairwise contact probabilities.
    """

    # 1. Parse Structure
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('afstruc', structure_cif)
    model = structure[0]
    
    # Internal helper to calculate standard IUPAC mass-weighted center of mass
    def _get_residue_com(residue) -> np.ndarray:
        # Standard atomic masses for macromolecular elements
        MASS_DICT = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'S': 32.060, 'P': 30.974}
        coords = []
        masses = []
        for atom in residue.get_atoms():
            element = atom.element.strip().upper()
            mass = MASS_DICT.get(element, 12.011)  # Fallback to Carbon mass if unmapped
            coords.append(atom.get_coord())
            masses.append(mass)
        if not coords:
            raise ValueError(f"Residue {residue.get_resname()} {residue.id} has no valid atoms.")
        return np.average(np.array(coords), axis=0, weights=np.array(masses))

    # Extract coordinates based on selected mode
    if coordinate_mode == 'CA':
        coords_1 = np.array([res['CA'].get_coord() for res in model[chainid_1] if 'CA' in res])
        coords_2 = np.array([res['CA'].get_coord() for res in model[chainid_2] if 'CA' in res])
    else:  # 'COM'
        coords_1 = np.array([_get_residue_com(res) for res in model[chainid_1]])
        coords_2 = np.array([_get_residue_com(res) for res in model[chainid_2]])
        
    # 2. Extract and mask the directional PAE submatrix
    with open(full_data_json, 'r') as file:
        full_data = json.load(file)
    res_pae = np.array(full_data['pae'])
    res_chain = np.array(full_data['token_chain_ids'])
    
    chain1_mask = res_chain == chainid_1
    chain2_mask = res_chain == chainid_2
    
    pae_submatrix = res_pae[chain1_mask, :][:, chain2_mask]
    
    # Methodological Safeguard: Verify alignment between structure data and JSON tokens
    if coords_1.shape[0] != pae_submatrix.shape[0] or coords_2.shape[0] != pae_submatrix.shape[1]:
        raise ValueError(
            f"Dimension mismatch between structural coordinates and PAE tokens.\n"
            f"Chain 1 ({chainid_1}) - Coords: {coords_1.shape[0]}, PAE Tokens: {pae_submatrix.shape[0]}\n"
            f"Chain 2 ({chainid_2}) - Coords: {coords_2.shape[0]}, PAE Tokens: {pae_submatrix.shape[1]}\n"
            f"Ensure structural residues exactly match the token mapping in full_data_json."
        )
    
    # 3. Compute pairwise distances between coordinate sets
    diff = coords_1[:, None, :] - coords_2[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    
    # 4. Map parameters to the intersection geometry model
    r_R = np.maximum(pae_submatrix, 1e-6)  # Prevent division by zero
    r_C = np.full_like(d, r_c)
    
    # 5. Calculate probabilities across geometric regimes
    prob_matrix = np.zeros_like(d)
    
    # Case 2a: Uncertainty sphere is completely inside contact sphere
    submerged_R_in_C = (d <= (r_C - r_R)) & (r_R <= r_C)
    prob_matrix[submerged_R_in_C] = 1.0
    
    # Case 2b: Contact sphere is completely inside uncertainty sphere
    submerged_C_in_R = (d <= (r_R - r_C)) & (r_R > r_C)
    prob_matrix[submerged_C_in_R] = (r_C[submerged_C_in_R] ** 3) / (r_R[submerged_C_in_R] ** 3)
    
    # Case 3: Partial Intersection
    intersect_mask = (d > np.abs(r_C - r_R)) & (d < (r_C + r_R))
    
    d_i = d[intersect_mask]
    r_C_i = r_C[intersect_mask]
    r_R_i = r_R[intersect_mask]
    
    vol_intersect = (np.pi * (r_C_i + r_R_i - d_i) ** 2 * (d_i ** 2 + 2 * d_i * (r_C_i + r_R_i) - 3 * (r_C_i - r_R_i) ** 2)) / (12.0 * d_i)
    
    vol_V_R = (4.0 / 3.0) * np.pi * (r_R_i ** 3)
    prob_matrix[intersect_mask] = vol_intersect / vol_V_R
    
    return np.clip(prob_matrix, 0.0, 1.0)