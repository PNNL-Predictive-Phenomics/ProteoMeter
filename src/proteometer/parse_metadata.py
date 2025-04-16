from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
import pandas as pd

from proteometer.params import Params
from proteometer.stats import TTestGroup


def group_columns(
    metadata: pd.DataFrame, par: Params
) -> tuple[list[list[str]], list[str]]:
    cond_column: pd.Series[str] = metadata[par.metadata_condition_col]
    control_ind: pd.Series[bool] = cond_column == par.metadata_control_condition
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
        metadata[control_ind][par.metadata_group_col].unique()  # type: ignore
    )
    treat_group_cols: list[list[str]] = [
        metadata[metadata[par.metadata_group_col] == group][
            par.metadata_sample_col
        ].to_list()
        for group in treat_groups
    ]
    return control_group_cols + treat_group_cols, control_groups + treat_groups


def int_columns(metadata: pd.DataFrame, par: Params) -> list[str]:
    ms: pd.Series[str] = metadata[par.metadata_sample_col]
    return ms.to_list()


def anova_columns(metadata: pd.DataFrame, par: Params) -> list[str]:
    condition_column: pd.Series[str] = metadata[par.metadata_condition_col]
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
        if sample not in np.ravel(tt_group_cols)
    ]
    return anova_cols


def t_test_groups(metadata: pd.DataFrame, par: Params) -> list[TTestGroup]:
    pairwise_pars: Iterable[str] = metadata[par.pairwise_factor].unique()  # type: ignore
    pairwise_ttest_groups: list[TTestGroup] = []

    cond_column: pd.Series[str] = metadata[par.metadata_condition_col]
    control_ind = cond_column == par.metadata_control_condition
    treat_ind = cond_column == par.metadata_treatment_condition

    pairwise_column: pd.Series[str] = metadata[par.pairwise_factor]

    for pairwise_par in pairwise_pars:
        pairwise_ind: pd.Series[bool] = pairwise_column == pairwise_par

        cgroups: Iterable[str] = metadata[control_ind & pairwise_ind][  # type: ignore
            par.metadata_group_col
        ].unique()

        for control_group in cgroups:
            tgroups: Iterable[str] = metadata[treat_ind & pairwise_ind][  # type: ignore
                par.metadata_group_col
            ].unique()

            for treat_group in tgroups:
                control_samples: list[float] = metadata[  # type: ignore
                    metadata[par.metadata_group_col] == control_group
                ][par.metadata_sample_col].to_list()
                treat_samples: list[float] = metadata[  # type: ignore
                    metadata[par.metadata_group_col] == treat_group
                ][par.metadata_sample_col].to_list()
                t_test_group = TTestGroup(
                    treat_group=treat_group,
                    control_group=control_group,
                    treat_samples=cast(list[float], treat_samples),
                    control_samples=cast(list[float], control_samples),
                )
                pairwise_ttest_groups.append(t_test_group)

    return pairwise_ttest_groups


def user_t_test_groups(metadata: pd.DataFrame, par: Params) -> list[TTestGroup]:
    user_pairwise_ttest_groups: list[TTestGroup] = []
    for user_ctrl_group, user_treat_group in par.user_ttest_pairs:
        control_samples = metadata[metadata[par.metadata_group_col] == user_ctrl_group][  # type: ignore
            par.metadata_sample_col
        ].to_list()
        treat_samples = metadata[metadata[par.metadata_group_col] == user_treat_group][  # type: ignore
            par.metadata_sample_col
        ].to_list()
        t_test_group = TTestGroup(
            treat_group=user_treat_group,
            control_group=user_ctrl_group,
            treat_samples=cast(list[float], treat_samples),
            control_samples=cast(list[float], control_samples),
        )

        user_pairwise_ttest_groups.append(t_test_group)
    return user_pairwise_ttest_groups
