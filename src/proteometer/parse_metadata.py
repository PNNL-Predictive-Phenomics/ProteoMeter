# type: ignore
import numpy as np
import pandas as pd

from proteometer.params import Params


def group_columns(metadata: pd.DataFrame, par: Params):
    control_groups = list(
        metadata[
            metadata[par.metadata_condition_col] == par.metadata_control_condition
        ][par.metadata_group_col].unique()
    )
    control_group_cols = [
        metadata[metadata[par.metadata_group_col] == group][
            par.metadata_sample_col
        ].to_list()
        for group in control_groups
    ]
    treat_groups = list(
        metadata[
            metadata[par.metadata_condition_col] == par.metadata_treatment_condition
        ][par.metadata_group_col].unique()
    )
    treat_group_cols = [
        metadata[metadata[par.metadata_group_col] == group][
            par.metadata_sample_col
        ].to_list()
        for group in treat_groups
    ]
    return control_group_cols + treat_group_cols, control_groups + treat_groups


def int_columns(metadata: pd.DataFrame, par: Params):
    return metadata[par.metadata_sample_col].to_list()


def anova_columns(metadata: pd.DataFrame, par: Params):
    tt_groups = list(
        metadata[metadata[par.metadata_condition_col] == par.pooled_chanel_condition][
            par.metadata_group_col
        ].unique()
    )
    tt_group_cols = [
        metadata[metadata[par.metadata_group_col] == group][
            par.metadata_sample_col
        ].to_list()
        for group in tt_groups
    ]
    anova_cols = [
        sample
        for sample in metadata[par.metadata_sample_col].values
        if sample not in np.ravel(tt_group_cols)
    ]
    return anova_cols


# TODO: get better types for t-test groups
def t_test_groups(metadata: pd.DataFrame, par: Params):
    pairwise_pars = metadata[par.pairwise_factor].unique()
    pairwise_ttest_groups = []
    for pairwise_par in pairwise_pars:
        for control_group in list(
            set(
                metadata[
                    (
                        metadata[par.metadata_condition_col]
                        == par.metadata_control_condition
                    )
                    & (metadata[par.pairwise_factor] == pairwise_par)
                ][par.metadata_group_col]
            )
        ):
            for treat_group in list(
                set(
                    metadata[
                        (
                            metadata[par.metadata_condition_col]
                            == par.metadata_treatment_condition
                        )
                        & (metadata[par.pairwise_factor] == pairwise_par)
                    ][par.metadata_group_col]
                )
            ):
                pairwise_ttest_groups.append(
                    [
                        f"{treat_group}/{control_group}",
                        control_group,
                        treat_group,
                        metadata[metadata[par.metadata_group_col] == control_group][
                            par.metadata_sample_col
                        ].to_list(),
                        metadata[metadata[par.metadata_group_col] == treat_group][
                            par.metadata_sample_col
                        ].to_list(),
                    ]
                )

    return pairwise_ttest_groups


def user_t_test_groups(metadata: pd.DataFrame, par: Params):
    user_pairwise_ttest_groups = []
    for user_test_pair in par.user_ttest_pairs:
        user_ctrl_group = user_test_pair[0]
        user_treat_group = user_test_pair[1]
        user_pairwise_ttest_groups.append(
            [
                f"{user_treat_group}/{user_ctrl_group}",
                user_ctrl_group,
                user_treat_group,
                metadata[metadata[par.metadata_group_col] == user_ctrl_group][
                    par.metadata_sample_col
                ].to_list(),
                metadata[metadata[par.metadata_group_col] == user_treat_group][
                    par.metadata_sample_col
                ].to_list(),
            ]
        )
    return user_pairwise_ttest_groups
