from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd


def get_res_names(residues: Iterable[str]) -> list[list[str]]:
    """Extracts residue names from an iterable of residue strings.

    Args:
        residues (Iterable[str]): An iterable of residue strings, each containing
            an uppercase letter followed by digits and optional lowercase letters or hyphens.

    Returns:
        list[list[str]]: A list of lists, where each inner list contains the extracted
            residue names from the corresponding input string. 
    """
    res_names = [
        [res for res in re.findall(r"[A-Z]\d+[a-z\-]*", residue)]
        for residue in residues
    ]
    return res_names


def get_res_pos(residues: Iterable[str]) -> list[list[int]]:
    """Extracts residue positions from an iterable of residue strings.

    Args:
        residues (Iterable[str]): An iterable of residue strings, each containing
            an uppercase letter followed by digits and optional lowercase letters or hyphens.

    Returns:
        list[list[int]]: A list of lists, where each inner list contains the extracted
            residue positions from the corresponding input string.
    """
    res_pos = [
        [int(res) for res in re.findall(r"\d+", residue)]
        for residue in residues
    ]
    return res_pos


def count_site_number(
    df: pd.DataFrame, uniprot_col: str, site_number_col: str = "site_number"
) -> pd.DataFrame:
    """Counts the number of sites per protein in a given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing protein and site information.
        uniprot_col (str): Column name of the protein identifier.
        site_number_col (str, optional): Name of the column to store the site number.
            Defaults to 'site_number'.

    Returns:
        pd.DataFrame: DataFrame with the site number added.
    """
    site_number = df.groupby(uniprot_col).size()
    site_number.name = site_number_col
    df = pd.merge(df, site_number, left_on=uniprot_col, right_index=True)
    return df


def count_site_number_with_global_proteomics(
    df: pd.DataFrame,  # index must match id_col
    uniprot_col: str,
    id_col: str,
    site_number_col: str = "site_number",
) -> pd.DataFrame:
    """Counts the number of sites per protein in a given DataFrame, with the global proteomics
    data used as the reference.

    Args:
        df (pd.DataFrame): DataFrame containing protein and site information. The index of
            this DataFrame must match `id_col`.
        uniprot_col (str): Column name of the protein identifier.
        id_col (str): Column name of the identifier that matches the index of the DataFrame.
        site_number_col (str, optional): Name of the column to store the site number.
            Defaults to 'site_number'.

    Returns:
        pd.DataFrame: DataFrame with the site number added.
    """
    site_number = df.groupby(uniprot_col).size() - 1  # type: ignore
    site_number.name = site_number_col
    for uniprot in site_number.index:  # type: ignore
        df.loc[df[id_col] == uniprot, site_number_col] = site_number[uniprot]
    return df
