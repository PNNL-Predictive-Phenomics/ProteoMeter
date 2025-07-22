from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import pandas as pd

from proteometer.params import Params
from proteometer.stats import TTestGroup

from proteometer.utils import flatten

def group_columns(
    metadata: pd.DataFrame, par: Params
) -> tuple[list[list[str]], list[str]]:
    """Generate the control and treatment group columns from the metadata.

    Args:
        metadata (pd.DataFrame): The metadata DataFrame with column names specified in `par`.
        par (Params): The parameters object with the following attributes:
            - metadata_group_col: Column name for group.
            - metadata_sample_col: Column name for sample.
            - metadata_condition_col: Column name for condition.
            - metadata_control_condition: Control condition name.

    Returns:
        tuple[list[list[str]], list[str]]: A tuple with the control and treatment
            group information. The first element is a list of lists, where each
            inner list contains the sample columns for each group. The second element
            is a list of group names. Control groups are first, followed by treatment groups.
    """
    cond_column: pd.Series[str] = metadata[par.metadata_condition_col].astype(str)
    control_ind: pd.Series[bool] = cond_column == par.metadata_control_condition
    treat_ind = cond_column == par.metadata_treatment_condition
    control_groups: list[str] = list(
        metadata[control_ind][par.metadata_group_col].unique()  # type: ignore
    )

    control_group_cols: list[list[str]] = [
        metadata[metadata[par.metadata_group_col] == group][
            par.metadata_sample_col
        ].to_list()
        for group in control_groups
    ]
    treat_groups: list[str] = list(
        metadata[treat_ind][par.metadata_group_col].unique()  # type: ignore
    )
    treat_group_cols: list[list[str]] = [
        metadata[metadata[par.metadata_group_col] == group][
            par.metadata_sample_col
        ].to_list()
        for group in treat_groups
    ]
    return control_group_cols + treat_group_cols, control_groups + treat_groups


def int_columns(metadata: pd.DataFrame, par: Params) -> list[str]:
    """Return a list of all the intensity columns in the metadata.

    Args:
        metadata (pd.DataFrame): The metadata DataFrame with column names specified in `par`.
        par (Params): The parameters object with the following attributes:
            - metadata_sample_col: Column name for sample.

    Returns:
        list[str]: A list of all the intensity columns in the metadata.
    """

    ms: pd.Series[str] = metadata[par.metadata_sample_col].astype(str)
    return ms.to_list()


def anova_columns(metadata: pd.DataFrame, par: Params) -> list[str]:
    """Return a list of columns for ANOVA analysis excluding pooled channel samples.

    Args:
        metadata (pd.DataFrame): The metadata DataFrame with column names specified in `par`.
        par (Params): The parameters object with attributes:
            - metadata_condition_col: Column name for condition.
            - pooled_chanel_condition: Condition name for pooled channel.
            - metadata_group_col: Column name for group.
            - metadata_sample_col: Column name for sample.

    Returns:
        list[str]: A list of sample columns excluding those in pooled channel groups.
    """

    condition_column: pd.Series[str] = metadata[par.metadata_condition_col].astype(str)
    control_ind: pd.Series[bool] = condition_column == par.pooled_chanel_condition
    tt_groups = list(metadata[control_ind][par.metadata_group_col].unique())  # type: ignore

    tt_group_cols: list[list[str]] = [
        metadata[metadata[par.metadata_group_col] == group][
            par.metadata_sample_col
        ].to_list()
        for group in tt_groups
    ]

    anova_cols = [
        sample
        for sample in cast(Iterable[str], metadata[par.metadata_sample_col].values)
        if sample not in flatten(tt_group_cols)
    ]
    return anova_cols


def t_test_groups(metadata: pd.DataFrame, par: Params) -> list[TTestGroup]:
    """Generate the user-defined pairwise t-test groups from the metadata.

    Args:
        metadata (pd.DataFrame): The metadata DataFrame with column names specified in `par`.
        par (Params): The parameters object with the following attributes:
            - metadata_group_col: Column name for group.
            - metadata_sample_col: Column name for sample.
            - ttest_pairs: List of tuples containing the user-defined control and treatment group pairs.

    Returns:
        list[TTestGroup]: A list of TTestGroup objects with the user-defined pairwise t-test groups.
    """
    pairwise_ttest_groups: list[TTestGroup] = []
    for ctrl_group, treat_group in par.ttest_pairs:
        control_samples = metadata[metadata[par.metadata_group_col] == ctrl_group][  # type: ignore
            par.metadata_sample_col
        ].to_list()
        treat_samples = metadata[metadata[par.metadata_group_col] == treat_group][  # type: ignore
            par.metadata_sample_col
        ].to_list()
        t_test_group = TTestGroup(
            treat_group=treat_group,
            control_group=ctrl_group,
            treat_samples=cast(list[str], treat_samples),
            control_samples=cast(list[str], control_samples),
        )

        pairwise_ttest_groups.append(t_test_group)
    return pairwise_ttest_groups
