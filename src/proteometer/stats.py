from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pingouin as pg
import scipy as sp


@dataclass
class TTestGroup:
    treat_group: str
    control_group: str
    treat_samples: Sequence[str]
    control_samples: Sequence[str]

    def label(self):
        return f"{self.treat_group}/{self.control_group}"


def log2_transformation(
    df2transform: pd.DataFrame, int_cols: Sequence[str]
) -> pd.DataFrame:
    """
    Applies log2 transformation to specified intensity columns in a DataFrame.

    Args:
        df2transform (pd.DataFrame): DataFrame containing the data to transform.
        int_cols (Sequence[str]): List of intensity column names to apply the transformation.

    Returns:
        pd.DataFrame: DataFrame with log2-transformed intensity columns.
    """
    df2transform[int_cols] = np.log2(df2transform[int_cols].replace(0, np.nan))
    return df2transform


def anova(
    df: pd.DataFrame,
    anova_cols: list[str],
    metadata: pd.DataFrame,
    anova_factors: Sequence[str],
    sample_col: str,
) -> pd.DataFrame:
    """
    Performs ANOVA on specified columns of a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data for analysis.
        anova_cols (list[str]): List of column names to analyze.
        metadata (pd.DataFrame): Metadata containing sample information.
        anova_factors (Sequence[str], optional): Factors for ANOVA analysis.
        sample_col (str, optional): Column name for sample identifiers.

    Returns:
        pd.DataFrame: DataFrame with ANOVA p-values and adjusted p-values.
    """
    if len(anova_factors) < 1:
        return df

    anova_factor_names = [
        f"{anova_factors[i]} * {anova_factors[j]}" if i != j else f"{anova_factors[i]}"
        for i in range(len(anova_factors))
        for j in range(i, len(anova_factors))
    ]

    df_w = df[anova_cols].copy()
    f_stats_factors: Sequence[pd.DataFrame] = []
    for df_id, row in df_w.iterrows():  # type: ignore
        df_f = pd.merge(
            row,
            metadata[metadata[sample_col].isin(anova_cols)],
            left_index=True,
            right_on=sample_col,
        )

        try:
            aov_f = pg.anova(data=df_f, dv=df_id, between=anova_factors, detailed=True)  # type: ignore
            if not isinstance(aov_f, pd.DataFrame):
                raise TypeError
            if "p-unc" in aov_f.columns:
                p_vals = {
                    f"ANOVA_[{anova_factor_name}]_pval": aov_f[
                        aov_f["Source"] == anova_factor_name
                    ]["p-unc"].values[0]
                    for anova_factor_name in anova_factor_names
                }
            else:
                p_vals = {
                    f"ANOVA_[{anova_factor_name}]_pval": np.nan
                    for anova_factor_name in anova_factor_names
                }

        except (
            TypeError,
            AssertionError,
            ValueError,
        ) as e:  # pg.anova can throw assertion error or value error if not enough data
            Warning(f"ANOVA failed for {df_id}: {e}")
            p_vals = {
                f"ANOVA_[{anova_factor_name}]_pval": np.nan
                for anova_factor_name in anova_factor_names
            }
        f_stats_factors.append(pd.DataFrame({"id": [df_id]} | p_vals))

    f_stats_factors_df = pd.concat(f_stats_factors).reset_index(drop=True)
    for anova_factor_name in anova_factor_names:
        f_stats_factors_df[f"ANOVA_[{anova_factor_name}]_adj-p"] = (
            sp.stats.false_discovery_control(
                f_stats_factors_df[f"ANOVA_[{anova_factor_name}]_pval"]
                .fillna(1.0)
                .astype(float)
            )
        )
        f_stats_factors_df.loc[
            f_stats_factors_df[f"ANOVA_[{anova_factor_name}]_pval"].isna(),
            f"ANOVA_[{anova_factor_name}]_adj-p",
        ] = np.nan
    f_stats_factors_df.set_index("id", inplace=True)
    df = pd.merge(df, f_stats_factors_df, left_index=True, right_index=True)

    return df


# Here is the function to do the t-test This is same for both protide and
# protein as well as rolled up protein data. Hopefully this is also the same for
# PTM data
def pairwise_ttest(
    df: pd.DataFrame, pairwise_ttest_groups: Iterable[TTestGroup]
) -> pd.DataFrame:
    """
    Performs pairwise t-tests for specified treatment and control groups.

    Args:
        df (pd.DataFrame): DataFrame containing the data for analysis.
        pairwise_ttest_groups (Iterable[TTestGroup]): Iterable of TTestGroup objects defining the groups.

    Returns:
        pd.DataFrame: DataFrame with t-test results, including p-values and adjusted p-values.
    """
    for pairwise_ttest_group in pairwise_ttest_groups:
        label = pairwise_ttest_group.label()
        df[label] = df[pairwise_ttest_group.treat_samples].mean(axis=1) - df[
            pairwise_ttest_group.control_samples
        ].mean(axis=1)
        df[f"{label}_pval"] = sp.stats.ttest_ind(
            df[pairwise_ttest_group.treat_samples],
            df[pairwise_ttest_group.control_samples],
            axis=1,
            nan_policy="omit",
        ).pvalue
        df[f"{label}_adj-p"] = sp.stats.false_discovery_control(
            df[f"{label}_pval"].fillna(1.0).astype(float)
        )
        df.loc[
            df[f"{label}_pval"].isna(),
            f"{label}_adj-p",
        ] = np.nan
    return df


# calculating the FC and p-values for protein abundances. See `abundance.py`
def calculate_pairwise_scalars(
    prot: pd.DataFrame,
    pairwise_ttest_name: str | None = None,
    sig_type: str = "pval",
    sig_thr: float = 0.05,
) -> pd.DataFrame:
    """
    Calculates pairwise scalars based on significance thresholds.

    Args:
        prot (pd.DataFrame): DataFrame containing pairwise t-test results.
        pairwise_ttest_name (str | None, optional): Name of the pairwise t-test column. Defaults to None.
        sig_type (str, optional): Type of significance metric (e.g., "pval"). Defaults to "pval".
        sig_thr (float, optional): Significance threshold. Defaults to 0.05.

    Returns:
        pd.DataFrame: DataFrame with calculated scalars.
    """
    if prot[f"{pairwise_ttest_name}_{sig_type}"].dtype != float:
        raise TypeError(f"{pairwise_ttest_name}_{sig_type} must be float")
    prot[f"{pairwise_ttest_name}_scalar"] = [
        prot[pairwise_ttest_name][i] if p < sig_thr else 0
        for i, p in enumerate(prot[f"{pairwise_ttest_name}_{sig_type}"])  # type: ignore
    ]
    return prot


def calculate_all_pairwise_scalars(
    prot: pd.DataFrame,
    pairwise_ttest_groups: Iterable[TTestGroup],
    sig_type: str = "pval",
    sig_thr: float = 0.05,
) -> pd.DataFrame:
    """
    Calculates pairwise scalars for all specified t-test groups.

    Args:
        prot (pd.DataFrame): DataFrame containing pairwise t-test results.
        pairwise_ttest_groups (Iterable[TTestGroup]): Iterable of TTestGroup objects defining the groups.
        sig_type (str, optional): Type of significance metric (e.g., "pval"). Defaults to "pval".
        sig_thr (float, optional): Significance threshold. Defaults to 0.05.

    Returns:
        pd.DataFrame: DataFrame with calculated scalars for all groups.
    """
    for pairwise_ttest_group in pairwise_ttest_groups:
        prot = calculate_pairwise_scalars(
            prot, pairwise_ttest_group.label(), sig_type, sig_thr
        )
    return prot
