# type: ignore

import re


def nip_off_pept(peptide):
    # pept_pattern = "\\.(.+)\\."
    # is equivalent to
    pept_pattern = r"\.(.+)\."
    subpept = re.search(pept_pattern, peptide).group(1)
    return subpept


def strip_peptide(peptide, nip_off=True):
    if nip_off:
        return re.sub(r"[^A-Za-z]+", "", nip_off_pept(peptide))
    else:
        return re.sub(r"[^A-Za-z]+", "", peptide)


def relable_pept(peptide, label_pos, ptm_label="*"):
    strip_pept = strip_peptide(peptide)
    for i, pos in enumerate(label_pos):
        strip_pept = (
            strip_pept[: (pos + i + 1)] + ptm_label + strip_pept[(pos + i + 1) :]
        )
    return peptide[:2] + strip_pept + peptide[-2:]
