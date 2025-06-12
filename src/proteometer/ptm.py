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
    """Get the positions of PTM labels in a peptide.

    This function processes a peptide string to find the positions of
    post-translational modification (PTM) labels. It accounts for special
    characters and returns a list of positions adjusted to the stripped
    peptide sequence. Positions are 0-indexed from the start of the peptide.

    Args:
        peptide (str): The peptide string potentially containing PTM labels.
        ptm_label (str, optional): The label representing PTM. Defaults to '*'.
        special_chars (str, optional): A string of special characters that
            might need escaping in regex operations. Defaults to common
            special characters.

    Returns:
        list[int]: A sorted list of integer positions where the PTM labels
            occur in the peptide, adjusted for any modifications made during
            processing.
    """
    peptide = nip_off_pept(peptide)
    if ptm_label in special_chars:
        ptm_label = "\\" + ptm_label
    ptm_pos = [m.start() for m in re.finditer(ptm_label, peptide)]
    pos = sorted([val - i for i, val in enumerate(ptm_pos)])
    return pos


def get_yst(strip_pept: str, ptm_aa: str = "YSTyst") -> list[tuple[int, str]]:
    """Get YST positions in a peptide.

    This function takes a stripped peptide sequence and finds the positions of
    Y, S, and T residues.

    Args:
        strip_pept (str): The stripped peptide sequence.
        ptm_aa (str, optional): The residues letters for Y, S, and T residues. Defaults to 'YSTyst'.

    Returns:
        list[tuple[int, str]]: A list of tuples where the first element is the
            position of the label in the stripped peptide and the second
            element is the YST residue letter.
    """
    return [
        (i, letter.upper()) for i, letter in enumerate(strip_pept) if letter in ptm_aa
    ]


def get_phosphositeplus_pos(mod_rsd: str) -> list[int]:
    """Extracts numeric positions from a string of modified residues.

    Args:
        mod_rsd (str): A string of modified residues.

    Returns:
        list[int]: A list of numeric positions extracted from the input string.
    """
    return [int(re.sub(r"[^0-9]+", "", mod)) for mod in mod_rsd]


def combine_multi_ptms(
    multi_proteomics: dict[str, pd.DataFrame], par: Params
) -> pd.DataFrame:
    """Combines multiple proteomics dataframes into a single dataframe.

    This function processes and combines different types of proteomics
    data into a unified dataframe. It distinguishes between global
    proteomics data and post-translational modifications (PTM) data,
    assigning specific labels and counting site numbers accordingly.

    Args:
        multi_proteomics (dict[str, pd.DataFrame]): Dictionary of proteomics dataframes
            with keys indicating the type of proteomics ('global' or PTM types).
        par (Params): Configuration parameters containing column names and PTM details.

    Returns:
        pd.DataFrame: A combined dataframe containing all the input proteomics data,
            labeled and processed as per the specified parameters.
    """

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
