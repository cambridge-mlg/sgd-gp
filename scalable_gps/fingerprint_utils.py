"""Code for getting features from molecules."""
from __future__ import annotations
from typing import Optional

import numpy as np
from joblib import Parallel, delayed

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from typing import Dict

FP_Dict = Dict[int, int]


def _smiles_to_mols(
    smiles_list: list[str], n_jobs: Optional[int] = None
) -> list[Chem.Mol]:
    """
    Helper function to convert list of SMILES to list of mols,
    raising an error if any invalid SMILES are found.
    """

    # Define a separate function since rdkit functions cannot be pickled by joblib
    def mol_from_smiles(s):
        return Chem.MolFromSmiles(s)

    if n_jobs is None:
        mols = [mol_from_smiles(s) for s in smiles_list]
    else:
        mols = Parallel(n_jobs=n_jobs)(delayed(mol_from_smiles)(s) for s in smiles_list)

    assert not any(m is None for m in mols)
    return mols


def mol_to_fp_dict(
    mols: list[Chem.Mol],
    radius: int,
    use_counts: bool = True,
    n_jobs: Optional[int] = None,
) -> list[FP_Dict]:
    """Get Morgan fingerprint bit dict from a list of mols."""

    # Define a separate function since rdkit functions cannot be pickled by joblib
    def fp_func(mol):
        return rdMolDescriptors.GetMorganFingerprint(
            mol, radius=radius, useCounts=use_counts
        ).GetNonzeroElements()

    if n_jobs is None:
        fps = [fp_func(mol) for mol in mols]
    else:
        fps = Parallel(n_jobs=n_jobs)(delayed(fp_func)(mol) for mol in mols)

    return fps


def mol_to_radius_sep_fp_dict(
    mols: list[Chem.Mol], max_radius: int, **kwargs
) -> list[dict[int, FP_Dict]]:
    """
    For each mol, return a dict mapping a radius to a fingerprint dict of bits active
    just for that particular radius. Essentially allows the radius origin of each fingerprint to be identified.
    """
    assert max_radius >= 0  # needs to be non-negative

    # Get all fingerprints
    all_radii = range(max_radius + 1)
    fps = {
        radius: mol_to_fp_dict(mols, radius=radius, **kwargs) for radius in all_radii
    }

    # Initialize output to be 0 radius dict
    out = [{0: fps[0][i]} for i, _ in enumerate(mols)]

    # Add each radius > 0 by subtracting the previous dict
    for radius in all_radii:
        if radius == 0:
            continue
        for fp_r, fp_last_r, d_out in zip(fps[radius], fps[radius - 1], out):
            d_add: dict[int, int] = dict()
            for k, n in fp_r.items():
                n_new = n - fp_last_r.get(k, 0)
                if n_new > 0:
                    d_add[k] = n_new
            d_out[radius] = d_add

    return out


def fp_dicts_to_arr(
    fp_dicts: list[FP_Dict], nbits: int, binarize: bool = False
) -> np.ndarray:
    """Convert a list of fingerprint dicts to a numpy array."""

    # Fold fingerprints into array
    out = np.zeros((len(fp_dicts), nbits))
    for i, fp in enumerate(fp_dicts):
        for k, v in fp.items():
            out[i, k % nbits] += v

    # Potentially binarize
    if binarize:
        out = np.minimum(out, 1.0)
        assert set(np.unique(out)) <= {0.0, 1.0}

    return out


def mol_to_fingerprint_arr(
    mols: list[Chem.Mol], nbits: int, binarize: bool = False, **kwargs
) -> np.ndarray:
    """Returns a fingerprint mapped into a numpy array."""
    fp_dicts = mol_to_fp_dict(mols=mols, **kwargs)
    return fp_dicts_to_arr(fp_dicts, nbits=nbits, binarize=binarize)


def mol_to_radius_sep_fingerprint_arr(
    mols: list[Chem.Mol],
    max_radius: int,
    bits_per_radius: int = 256,
    binarize: bool = False,
    **kwargs,
) -> np.ndarray:
    """Returns an array representing the concatenation of fingerprint arrays for a series of radii."""

    # Create fingerprints
    fps_per_radius = mol_to_radius_sep_fp_dict(
        mols=mols, max_radius=max_radius, **kwargs
    )

    # Make a separate array for each radius
    all_arrs: list[np.ndarray] = list()
    for r in range(max_radius + 1):
        all_arrs.append(
            fp_dicts_to_arr(
                fp_dicts=[d[r] for d in fps_per_radius],
                nbits=bits_per_radius,
                binarize=binarize,
            )
        )

    # Return concatenation of arrays
    return np.concatenate(all_arrs, axis=1)


def smiles_to_fingerprint_arr(
    smiles_list: list[str], n_jobs: Optional[int] = None, **kwargs
) -> np.array:
    mol_list = _smiles_to_mols(smiles_list, n_jobs=n_jobs)
    return mol_to_fingerprint_arr(mols=mol_list, n_jobs=n_jobs, **kwargs)


def smiles_to_radius_sep_fingerprint_arr(
    smiles_list: list[str], n_jobs: Optional[int] = None, **kwargs
) -> np.array:
    mol_list = _smiles_to_mols(smiles_list, n_jobs=n_jobs)
    return mol_to_radius_sep_fingerprint_arr(mols=mol_list, n_jobs=n_jobs, **kwargs)