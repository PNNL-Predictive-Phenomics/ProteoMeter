from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from proteometer.utils import expsum

if TYPE_CHECKING:
    from typing import Any, Callable, Literal

    import pandas as pd

    AggDictFloat = dict[str, Callable[[pd.Series[float]], float]]
    AggDictStr = dict[str, Callable[[pd.Series[str]], str]]
    AggDictAny = dict[str, Callable[[pd.Series[Any]], Any]]


def rollup_to_site(
    df_ori: pd.DataFrame,
    int_cols: list[str],
    uniprot_col: str,
    peptide_col: str,
    residue_col: str,
    residue_sep: str = ";",
    id_col: str = "id",
    id_separator: str = "@",
    site_col: str = "Site",
    multiply_rollup_counts: bool = True,
    ignore_NA: bool = True,
    rollup_func: Literal["median", "mean", "sum"] = "sum",
) -> pd.DataFrame:
    """Roll up peptide-level data to site-level data.

    Args:
        df_ori (pd.DataFrame): Original DataFrame containing peptide data.
        int_cols (list[str]): List of column names with intensity values to roll up.
        uniprot_col (str): Column name for UniProt identifiers.
        peptide_col (str): Column name for peptides.
        residue_col (str): Column name for residues.
        residue_sep (str, optional): Separator for residues in the residue column. Defaults to ";".
        id_col (str, optional): Column name for generated IDs. Defaults to "id".
        id_separator (str, optional): Separator for ID components. Defaults to "@".
        site_col (str, optional): Column name for site information. Defaults to "Site".
        multiply_rollup_counts (bool, optional): Whether to multiply rollup counts by the number of observations. Defaults to True.
        ignore_NA (bool, optional): Whether to ignore NA values during rollup. Defaults to True.
        rollup_func (Literal["median", "mean", "sum"], optional): Aggregation function to use. Defaults to "sum".

    Returns:
        pd.DataFrame: DataFrame with rolled-up site-level data.
    """

    df = df_ori.reset_index(drop=True)
    info_cols = [col for col in df.columns if col not in int_cols]

    df[residue_col] = df[residue_col].str.split(residue_sep)
    df = df.explode(residue_col)
    df[id_col] = df[uniprot_col] + id_separator + df[residue_col]

    info_cols_wo_peptide_col = [col for col in info_cols if col != peptide_col]
    agg_methods_0: AggDictStr = {peptide_col: lambda x: "; ".join(x)}
    agg_methods_1: AggDictAny = {
        i: lambda x: x.iloc[0] for i in info_cols_wo_peptide_col
    }
    if multiply_rollup_counts:
        if ignore_NA:
            if rollup_func.lower() == "median":
                agg_methods_2: AggDictFloat = {
                    i: lambda x: (np.log2(len(x)) + x.median() if (not x[x.notna()].empty) else np.nan) for i in int_cols
                }
            elif rollup_func.lower() == "mean":
                agg_methods_2: AggDictFloat = {
                    i: lambda x: (np.log2(len(x)) + x.mean() if (not x[x.notna()].empty) else np.nan) for i in int_cols
                }
            elif rollup_func.lower() == "sum":
                agg_methods_2: AggDictFloat = {i: expsum for i in int_cols}
            else:
                raise ValueError(
                    "The rollup function is not recognized. Please choose from the following: median, mean, sum"
                )
        else:
            if rollup_func.lower() == "median":
                agg_methods_2: AggDictFloat = {
                    i: lambda x: (np.log2(x.notna().sum()) + x.median() if (not x[x.notna()].empty) else np.nan) for i in int_cols
                }
            elif rollup_func.lower() == "mean":
                agg_methods_2: AggDictFloat = {
                    i: lambda x: (np.log2(x.notna().sum()) + x.mean() if (not x[x.notna()].empty) else np.nan) for i in int_cols
                }
            elif rollup_func.lower() == "sum":
                agg_methods_2: AggDictFloat = {i: expsum for i in int_cols}
            else:
                raise ValueError(
                    "The rollup function is not recognized. Please choose from the following: median, mean, sum"
                )
    else:
        if rollup_func.lower() == "median":
            agg_methods_2: AggDictFloat = {i: lambda x: (x.median() if (not x[x.notna()].empty) else np.nan) for i in int_cols}
        elif rollup_func.lower() == "mean":
            agg_methods_2: AggDictFloat = {i: lambda x: (x.mean() if (not x[x.notna()].empty) else np.nan) for i in int_cols}
        elif rollup_func.lower() == "sum":
            agg_methods_2: AggDictFloat = {i: expsum for i in int_cols}
        else:
            raise ValueError(
                "The rollup function is not recognized. Please choose from the following: median, mean, sum"
            )
    df = df.groupby(id_col, as_index=False).agg(
        {**agg_methods_0, **agg_methods_1, **agg_methods_2}
    )
    df[site_col] = df[id_col].to_list()
    df[int_cols] = df[int_cols].replace([np.inf, -np.inf], np.nan)
    df.index = df[id_col].to_list()  # type: ignore
    return df


