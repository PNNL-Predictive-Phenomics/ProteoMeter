from __future__ import annotations

import re
from typing import Iterable


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


def relable_pept(peptide: str, label_pos: Iterable[int], ptm_label: str = "*") -> str:
    """Relabels the peptide with PTM labels at specified positions.

    Args:
        peptide (str): The original peptide string.
        label_pos (Iterable[int]): Positions where the PTM labels should be inserted.
        ptm_label (str, optional): The label to insert. Defaults to '*'.

    Returns:
        str: The peptide string with inserted PTM labels.
    """
    strip_pept = strip_peptide(peptide)
    for i, pos in enumerate(label_pos):
        strip_pept = (
            strip_pept[: (pos + i + 1)] + ptm_label + strip_pept[(pos + i + 1) :]
        )
    return peptide[:2] + strip_pept + peptide[-2:]
