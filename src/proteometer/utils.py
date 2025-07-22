from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, List

    import pandas as pd


def flatten(s: List[Any]) -> List[Any]:
    """
    Flattens a nested list into a single list.
    Args:
        s (List[Any]): A list that may contain nested lists.
    Returns:
        List[Any]: A flattened list containing all elements from the input list.
    """
    result = []
    stack = [s]
    while stack:
        current = stack.pop()
        if isinstance(current, list):  # type: ignore
            stack.extend(reversed(current))  # Reverse to maintain order
        else:
            result.append(current)
    return result  # type: ignore


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
    method: Literal["all", "any"] = "any",
) -> pd.DataFrame:
    """
    Filter rows in a DataFrame based on missingness thresholds for specified groups.

    Args:
        df (pd.DataFrame): Input DataFrame.
        groups (Sequence[str]): Names of the groups.
        group_cols (Sequence[Sequence[str]]): Columns corresponding to each group.
        min_replicates_qc (float, optional): Threshold for minimal number of
            replicates that are not NA. Defaults to 2.
        method (str, optional): Method for filtering. Can be "all" or "any".
            Defaults to "all". If "all", all groups must meet the threshold.
            If "any", at least one group must meet the threshold.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    df = check_missingness(df, groups, group_cols)

    df["missing_check"] = 0
    for name, cols in zip(groups, group_cols):
        df["missing_check"] = df["missing_check"] + (
            (len(cols) - df[f"{name} missingness"]) < min_replicates_qc
        ).astype(int)
    if method == "all":
        df_w = df[df["missing_check"] == 0].copy()
    elif method == "any":
        df_w = df[df["missing_check"] < len(groups)].copy()
    else:
        raise ValueError(f"Unknown method: {method}. Use 'all' or 'any'.")
    return df_w


def expsum(x: pd.Series[float]) -> float:
    val = cast(float, np.nansum(2 ** (x.replace(0, np.nan))))
    if val == 0:
        return np.nan
    return np.log2(val)
