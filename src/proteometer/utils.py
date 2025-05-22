from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


def generate_index(
    df: pd.DataFrame,
    prot_col: str,
    level_col: str | None = None,
    id_separator: str = "@",
    id_col: str = "id",
) -> pd.DataFrame:
    """
    Generate a unique index for a DataFrame based on protein column identifier and optional level column identifier.

    Args:
        df (pd.DataFrame): Input DataFrame.
        prot_col (str): Column name for protein identifiers.
        level_col (str | None, optional): Column name for level identifiers. Defaults to None.
        id_separator (str, optional): Separator for combining protein and level identifiers. Defaults to "@".
        id_col (str, optional): Name of the new column for the generated index. Defaults to "id".

    Returns:
        pd.DataFrame: DataFrame with the generated index.
    """
    if level_col is None:
        df[id_col] = df[prot_col]
    else:
        df[id_col] = df[prot_col] + id_separator + df[level_col]

    # proper way to do this is
    # df.set_index(id_col, inplace=True)
    # but there is a bunch of reindexing going on,
    # so this would require fixing this elsewhere too.
    # In the short term, it is easiest to just ignore
    # this since it works.
    df.index = df[id_col].to_list()  # type: ignore
    return df


def check_missingness(
    df: pd.DataFrame, groups: Sequence[str], group_cols: Sequence[Sequence[str]]
) -> pd.DataFrame:
    """
    Calculate missingness for specified groups in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        groups (Sequence[str]): Names of the groups.
        group_cols (Sequence[Sequence[str]]): Columns corresponding to each group.

    Returns:
        pd.DataFrame: DataFrame with missingness information added.
    """
    df["Total missingness"] = 0
    for name, cols in zip(groups, group_cols):
        df[f"{name} missingness"] = df[cols].isna().sum(axis=1)
        df["Total missingness"] = df["Total missingness"] + df[f"{name} missingness"]
    return df


def filter_missingness(
    df: pd.DataFrame,
    groups: Sequence[str],
    group_cols: Sequence[Sequence[str]],
    min_replicates_qc: int = 2,
) -> pd.DataFrame:
    """
    Filter rows in a DataFrame based on missingness thresholds for specified groups.

    Args:
        df (pd.DataFrame): Input DataFrame.
        groups (Sequence[str]): Names of the groups.
        group_cols (Sequence[Sequence[str]]): Columns corresponding to each group.
        missing_thr (float, optional): Threshold for missingness. Defaults to 0.0.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    df = check_missingness(df, groups, group_cols)

    df["missing_check"] = 0
    for name, cols in zip(groups, group_cols):
        df["missing_check"] = df["missing_check"] + (
            (len(cols) - df[f"{name} missingness"]) < min_replicates_qc 
        ).astype(int)
    df_w = df[~(df["missing_check"] > 0)].copy()
    return df_w
