from __future__ import annotations

import re
from typing import Iterable


def nip_off_pept(peptide: str) -> str:
    # pept_pattern = "\\.(.+)\\."
    # is equivalent to
    pept_pattern = r"\.(.+)\."
    match = re.search(pept_pattern, peptide)
    if match is None:
        return peptide
    subpept = match.group(1)
    return subpept


def strip_peptide(peptide: str, nip_off: bool = True) -> str:
    if nip_off:
        return re.sub(r"[^A-Za-z]+", "", nip_off_pept(peptide))
    else:
        return re.sub(r"[^A-Za-z]+", "", peptide)


def relable_pept(peptide: str, label_pos: Iterable[int], ptm_label: str = "*"):
    strip_pept = strip_peptide(peptide)
    for i, pos in enumerate(label_pos):
        strip_pept = (
            strip_pept[: (pos + i + 1)] + ptm_label + strip_pept[(pos + i + 1) :]
        )
    return peptide[:2] + strip_pept + peptide[-2:]
