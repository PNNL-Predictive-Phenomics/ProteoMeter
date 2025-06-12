from __future__ import annotations

import re


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
