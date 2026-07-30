from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable


def nip_off_pept(peptide: str) -> str:
    """Extracts the core peptide sequence surrounded by `.` characters.

    Args:
        peptide (str): The peptide string containing flanking characters.

    Returns:
        str: The core peptide sequence without flanking characters.
    """
    pept_pattern = r"\.(.+)\."
    match = re.search(pept_pattern, peptide)
    if match is None:
        return peptide
    subpept = match.group(1)
    return subpept


def strip_peptide(peptide: str, nip_off: bool = True) -> str:
    """Removes non-alphabetic characters and optionally nips off flanking characters.

    Args:
        peptide (str): The peptide string to be cleaned.
        nip_off (bool, optional): Whether to nip off flanking characters. Defaults to True.

    Returns:
        str: The cleaned peptide string.
    """
    if nip_off:
        return re.sub(r"[^A-Za-z]+", "", nip_off_pept(peptide))
    else:
        return re.sub(r"[^A-Za-z]+", "", peptide)


def compute_peptide_coverage(
    pept_start_positions: Iterable[int],
    pept_end_positions: Iterable[int],
    sequence_length: int,
) -> np.ndarray:
    """Computes the coverage of peptides over a protein sequence.

    Args:
        pept_start_positions (Iterable[int]): Start positions of peptides.
        pept_end_positions (Iterable[int]): End positions of peptides.
        sequence_length (int): Length of the protein sequence.

    Returns:
        np.ndarray: An array representing the coverage of each position in the protein sequence.
    """
    coverage = np.zeros(sequence_length, dtype=int)
    for start, end in zip(pept_start_positions, pept_end_positions, strict=True):
        coverage[start:end] += 1
    return coverage
