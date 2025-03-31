# type: ignore
import re

import pandas as pd

from proteometer.peptide import nip_off_pept, strip_peptide
from proteometer.residue import (
    count_site_number,
    count_site_number_with_global_proteomics,
)


def get_ptm_pos_in_pept(
    peptide, ptm_label="*", special_chars=r".]+-=@_!#$%^&*()<>?/\|}{~:["
):
    peptide = nip_off_pept(peptide)
    if ptm_label in special_chars:
        ptm_label = "\\" + ptm_label
    ptm_pos = [m.start() for m in re.finditer(ptm_label, peptide)]
    pos = sorted([val - i - 1 for i, val in enumerate(ptm_pos)])
    return pos


def get_yst(strip_pept, ptm_aa="YSTyst"):
    return [
        [i, letter.upper()] for i, letter in enumerate(strip_pept) if letter in ptm_aa
    ]


def get_ptm_info(peptide, residue=None, prot_seq=None, ptm_label="*"):
    if prot_seq is not None:
        clean_pept = strip_peptide(peptide)
        pept_pos = prot_seq.find(clean_pept)
        all_yst = get_yst(clean_pept)
        all_ptm = [[pept_pos + yst[0] + 1, yst[1], yst[0]] for yst in all_yst]
        return all_ptm
    if residue is not None:
        subpept = nip_off_pept(peptide)
        split_substr = subpept.split(ptm_label)
        res_pos = sorted([int(res) for res in re.findall(r"\d+", residue)])
        first_pos = res_pos[0]
        res_pos.insert(0, first_pos - len(split_substr[0]))
        pept_pos = 0
        all_ptm = []
        for i, res in enumerate(res_pos):
            # print(i)
            if i > 0:
                pept_pos += len(split_substr[i - 1])
            yst_pos = get_yst(split_substr[i])
            if len(yst_pos) > 0:
                for j in yst_pos:
                    ptm = [j[0] + res + 1, j[1], pept_pos + j[0]]
                    all_ptm.append(ptm)
        return all_ptm


def get_phosphositeplus_pos(mod_rsd):
    return [int(re.sub(r"[^0-9]+", "", mod)) for mod in mod_rsd]


def combine_multi_ptms(
    multi_proteomics,
    residue_col,
    uniprot_col,
    site_col,
    site_number_col,
    id_separator="@",
    id_col="id",
    type_col="Type",
    experiment_col="Experiment",
):
    proteomics_list = []
    for key, value in multi_proteomics.items():
        if key.lower() == "global":
            prot = value
            prot[type_col] = "global"
            prot[experiment_col] = "PTM"
            prot[residue_col] = "GLB"
            prot[site_col] = prot[uniprot_col] + id_separator + prot[residue_col]
            proteomics_list.append(prot)
        else:
            ptm_df = value
            ptm_df[type_col] = key
            ptm_df[experiment_col] = "PTM"
            ptm_df = count_site_number(ptm_df, uniprot_col, site_number_col)
            proteomics_list.append(ptm_df)

    all_ptms = (
        pd.concat(proteomics_list, axis=0, join="outer", ignore_index=True)
        .sort_values(by=[id_col, type_col, experiment_col, site_col])
        .reset_index(drop=True)
    )
    all_ptms = count_site_number_with_global_proteomics(
        all_ptms, uniprot_col, id_col, site_number_col
    )

    return all_ptms
