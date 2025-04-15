from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any, Callable, Literal

    import pandas as pd

    AggDictFloat = dict[str, Callable[[pd.Series[float]], float]]
    AggDictStr = dict[str, Callable[[pd.Series[str]], str]]
    AggDictAny = dict[str, Callable[[pd.Series[Any]], Any]]


# Rolling up the quantification by using log2 medians and plus the log2 of peptide number
def rollup(
    df_ori: pd.DataFrame,
    int_cols: list[str],
    rollup_col: str,
    id_col: str = "id",
    id_separator: str = "@",
    multiply_rollup_counts: bool = True,
    ignore_NA: bool = True,
    rollup_func: Literal["median", "mean", "sum"] = "median",
):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = df_ori.reset_index(drop=True)
    info_cols = [col for col in df.columns if col not in int_cols]

    df[id_col] = df[id_col] + id_separator + df[rollup_col]

    df[int_cols] = 2 ** df[int_cols]
    df[int_cols] = df[int_cols].fillna(0)

    agg_methods_1: AggDictAny = {i: lambda x: x.iloc[0] for i in info_cols}
    if multiply_rollup_counts:
        if ignore_NA:
            if rollup_func.lower() == "median":
                agg_methods_2: AggDictFloat = {
                    i: lambda x: np.log2(len(x)) + x.median() for i in int_cols
                }
            elif rollup_func.lower() == "mean":
                agg_methods_2: AggDictFloat = {
                    i: lambda x: np.log2(len(x)) + x.mean() for i in int_cols
                }
            elif rollup_func.lower() == "sum":
                agg_methods_2: AggDictFloat = {
                    i: lambda x: np.log2(np.nansum(2 ** (x.replace(0, np.nan))))
                    for i in int_cols
                }
            else:
                raise ValueError(
                    "The rollup function is not recognized. Please choose from the following: median, mean, sum"
                )
        else:
            if rollup_func.lower() == "median":
                agg_methods_2: AggDictFloat = {
                    i: lambda x: np.log2(x.notna().sum()) + x.median() for i in int_cols
                }
            elif rollup_func.lower() == "mean":
                agg_methods_2: AggDictFloat = {
                    i: lambda x: np.log2(x.notna().sum()) + x.mean() for i in int_cols
                }
            elif rollup_func.lower() == "sum":
                agg_methods_2: AggDictFloat = {
                    i: lambda x: np.log2(np.nansum(2 ** (x.replace(0, np.nan))))
                    for i in int_cols
                }
            else:
                raise ValueError(
                    "The rollup function is not recognized. Please choose from the following: median, mean, sum"
                )
    else:
        if rollup_func.lower() == "median":
            agg_methods_2: AggDictFloat = {i: lambda x: x.median() for i in int_cols}
        elif rollup_func.lower() == "mean":
            agg_methods_2: AggDictFloat = {i: lambda x: x.mean() for i in int_cols}
        elif rollup_func.lower() == "sum":
            agg_methods_2: AggDictFloat = {
                i: lambda x: np.log2(np.nansum(2 ** (x.replace(0, np.nan))))
                for i in int_cols
            }
        else:
            raise ValueError(
                "The rollup function is not recognized. Please choose from the following: median, mean, sum"
            )
    df = df.groupby(id_col, as_index=False).agg({**agg_methods_1, **agg_methods_2})
    df[int_cols] = np.log2(df[int_cols])
    df[int_cols] = df[int_cols].replace([np.inf, -np.inf], np.nan)
    df.index = df[id_col].to_list()  # type: ignore
    return df


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
):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
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
                    i: lambda x: np.log2(len(x)) + x.median() for i in int_cols
                }
            elif rollup_func.lower() == "mean":
                agg_methods_2: AggDictFloat = {
                    i: lambda x: np.log2(len(x)) + x.mean() for i in int_cols
                }
            elif rollup_func.lower() == "sum":
                agg_methods_2: AggDictFloat = {
                    i: lambda x: np.log2(np.nansum(2 ** (x.replace(0, np.nan))))
                    for i in int_cols
                }
            else:
                raise ValueError(
                    "The rollup function is not recognized. Please choose from the following: median, mean, sum"
                )
        else:
            if rollup_func.lower() == "median":
                agg_methods_2: AggDictFloat = {
                    i: lambda x: np.log2(x.notna().sum()) + x.median() for i in int_cols
                }
            elif rollup_func.lower() == "mean":
                agg_methods_2: AggDictFloat = {
                    i: lambda x: np.log2(x.notna().sum()) + x.mean() for i in int_cols
                }
            elif rollup_func.lower() == "sum":
                agg_methods_2: AggDictFloat = {
                    i: lambda x: np.log2(np.nansum(2 ** (x.replace(0, np.nan))))
                    for i in int_cols
                }
            else:
                raise ValueError(
                    "The rollup function is not recognized. Please choose from the following: median, mean, sum"
                )
    else:
        if rollup_func.lower() == "median":
            agg_methods_2: AggDictFloat = {i: lambda x: x.median() for i in int_cols}
        elif rollup_func.lower() == "mean":
            agg_methods_2: AggDictFloat = {i: lambda x: x.mean() for i in int_cols}
        elif rollup_func.lower() == "sum":
            agg_methods_2: AggDictFloat = {
                i: lambda x: np.log2(np.nansum(2 ** (x.replace(0, np.nan))))
                for i in int_cols
            }
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
