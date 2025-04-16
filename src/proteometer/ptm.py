from __future__ import annotations

import re

import pandas as pd

from proteometer.params import Params
from proteometer.peptide import nip_off_pept
from proteometer.residue import (
    count_site_number,
    count_site_number_with_global_proteomics,
)


def get_ptm_pos_in_pept(
    peptide: str,
    ptm_label: str = "*",
    special_chars: str = r".]+-=@_!#$%^&*()<>?/\|}{~:[",
) -> list[int]:
    peptide = nip_off_pept(peptide)
    if ptm_label in special_chars:
        ptm_label = "\\" + ptm_label
    ptm_pos = [m.start() for m in re.finditer(ptm_label, peptide)]
    pos = sorted([val - i - 1 for i, val in enumerate(ptm_pos)])
    return pos


def get_yst(strip_pept: str, ptm_aa: str = "YSTyst") -> list[tuple[int, str]]:
    return [
        (i, letter.upper()) for i, letter in enumerate(strip_pept) if letter in ptm_aa
    ]


# Function not used
# def get_ptm_info(peptide:str, residue:str | None=None, prot_seq:str | None=None, ptm_label:str="*"):
#     if prot_seq is not None:
#         clean_pept = strip_peptide(peptide)
#         pept_pos = prot_seq.find(clean_pept)
#         all_yst = get_yst(clean_pept)
#         all_ptm = [[pept_pos + yst[0] + 1, yst[1], yst[0]] for yst in all_yst]
#         return all_ptm
#     if residue is not None:
#         subpept = nip_off_pept(peptide)
#         split_substr = subpept.split(ptm_label)
#         res_pos = sorted([int(res) for res in re.findall(r"\d+", residue)])
#         first_pos = res_pos[0]
#         res_pos.insert(0, first_pos - len(split_substr[0]))
#         pept_pos = 0
#         all_ptm = []
#         for i, res in enumerate(res_pos):
#             # print(i)
#             if i > 0:
#                 pept_pos += len(split_substr[i - 1])
#             yst_pos = get_yst(split_substr[i])
#             if len(yst_pos) > 0:
#                 for j in yst_pos:
#                     ptm = [j[0] + res + 1, j[1], pept_pos + j[0]]
#                     all_ptm.append(ptm)
#         return all_ptm


def get_phosphositeplus_pos(mod_rsd: str) -> list[int]:
    return [int(re.sub(r"[^0-9]+", "", mod)) for mod in mod_rsd]


def combine_multi_ptms(multi_proteomics: dict[str, pd.DataFrame], par: Params):
    proteomics_list: list[pd.DataFrame] = []
    for key, value in multi_proteomics.items():
        if key == "global":
            prot = value
            prot[par.type_col] = "global"
            prot[par.experiment_col] = "PTM"
            prot[par.residue_col] = "GLB"
            prot[par.site_col] = (
                prot[par.uniprot_col] + par.id_separator + prot[par.residue_col]
            )
            proteomics_list.append(prot)
        elif key in par.ptm_names:
            ptm_df = value
            ptm_df[par.type_col] = par.ptm_abbreviations[key]
            ptm_df[par.experiment_col] = "PTM"
            ptm_df = count_site_number(ptm_df, par.uniprot_col, par.site_number_col)
            proteomics_list.append(ptm_df)
        else:
            KeyError(
                f"The key {key} is not recognized. Please check the input data and config file."
            )

    all_ptms = (
        pd.concat(proteomics_list, axis=0, join="outer", ignore_index=True)
        .sort_values(by=[par.id_col, par.type_col, par.experiment_col, par.site_col])
        .reset_index(drop=True)
    )
    all_ptms = count_site_number_with_global_proteomics(
        all_ptms, par.uniprot_col, par.id_col, par.site_number_col
    )

    return all_ptms
