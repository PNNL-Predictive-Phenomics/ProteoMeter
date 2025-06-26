from __future__ import annotations

import warnings
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


def recalculate_adj_pval(df: pd.DataFrame, comparisons: list[str]):
    """
    Recalculates adjusted p-values for specified comparisons.

    Args:
        df (pd.DataFrame): DataFrame containing p-values and adjusted p-values.
        comparisons (list[str]): List of comparison names. Each comparison
            should have a p-value and adjusted p-value indicated by a "_pval" and
            "_adj-p" suffix, respectively.

    Returns:
        pd.DataFrame: DataFrame with recalculated adjusted p-values.
    """
    for comparison in comparisons:
        pcol = f"{comparison}_pval"
        apcol = f"{comparison}_adj-p"
        ind = ~df[pcol].isna()
        df.loc[ind, apcol] = sp.stats.false_discovery_control(
            df[ind][pcol].astype(float)
        )
        df.loc[
            df[pcol].isna(),
            apcol,
        ] = np.nan

    return df


def recalculate_adj_pval_proteinwise(
    df: pd.DataFrame, comparisons: list[str], protein_col: str = "Protein"
):
    """
    Recalculates adjusted p-values for specified comparisons, computed protein-wise.

    See:
    - Schopper et al. Nature Protocols, 12(11):2391-2410, October 2017.
    - Nagel et al. Cellular Proteomics, 24(4):100934, April 2025.

    Args:
        df (pd.DataFrame): DataFrame containing p-values and adjusted p-values.
        comparisons (list[str]): List of comparison names. Each comparison
            should have a p-value and adjusted p-value indicated by a "_pval" and
            "_adj-p" suffix, respectively.

    Returns:
        pd.DataFrame: DataFrame with recalculated adjusted p-values.
    """
    for comparison in comparisons:
        pcol = f"{comparison}_pval"
        apcol = f"{comparison}_adj-p"
        ind = ~df[pcol].isna()
        for protein in df[protein_col].unique():
            ind_prot = ind & (df[protein_col] == protein)
            df.loc[ind_prot, apcol] = sp.stats.false_discovery_control(
                df[ind_prot][pcol].astype(float)  # type: ignore
            )
        df.loc[
            df[pcol].isna(),
            apcol,
        ] = np.nan

    return df


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
    ret = df2transform.copy()
    ret[int_cols] = np.log2(ret[int_cols].replace(0, np.nan))
    return ret


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
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=Warning)
                aov_f = pg.anova(  # type: ignore
                    data=df_f, dv=df_id, between=anova_factors, detailed=True
                )
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
        ind = ~f_stats_factors_df[f"ANOVA_[{anova_factor_name}]_pval"].isna()
        f_stats_factors_df.loc[ind, f"ANOVA_[{anova_factor_name}]_adj-p"] = (
            sp.stats.false_discovery_control(
                f_stats_factors_df[ind][f"ANOVA_[{anova_factor_name}]_pval"].astype(
                    float
                )
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

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            df[f"{label}_pval"] = sp.stats.ttest_ind(
                df[pairwise_ttest_group.treat_samples],
                df[pairwise_ttest_group.control_samples],
                axis=1,
                nan_policy="omit",
            ).pvalue

            ind = ~df[f"{label}_pval"].isna()
            df.loc[ind, f"{label}_adj-p"] = sp.stats.false_discovery_control(
                df[ind][f"{label}_pval"].astype(float)
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
