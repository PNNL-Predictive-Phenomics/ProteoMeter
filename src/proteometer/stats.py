# type: ignore
import numpy as np
import pandas as pd
import pingouin as pg
import scipy as sp


def log2_transformation(df2transform, int_cols):
    """_summary_

    Args:
        df2transform (_type_): _description_
    """
    df2transform[int_cols] = np.log2(df2transform[int_cols].replace(0, np.nan))
    return df2transform


def anova(df, anova_cols, metadata_ori, anova_factors=["Group"], sample_col="Sample"):
    """_summary_

    Args:
        df (_type_): _description_
        anova_cols (_type_): _description_
        metadata_ori (_type_): _description_
        anova_factors (list, optional): _description_. Defaults to ["Group"].

    Returns:
        _type_: _description_
    """
    if len(anova_factors) < 1:
        raise ValueError(
            "The anova_factors is empty. Please provide the factors for ANOVA analysis. The default factor is 'Group'."
        )

    metadata = metadata_ori[metadata_ori[sample_col].isin(anova_cols)].copy()

    anova_factor_names = [
        f"{anova_factors[i]} * {anova_factors[j]}" if i != j else f"{anova_factors[i]}"
        for i in range(len(anova_factors))
        for j in range(i, len(anova_factors))
    ]

    df_w = df[anova_cols].copy()
    f_stats_factors = []
    for row in df_w.iterrows():
        df_id = row[0]
        df_f = row[1]
        df_f = pd.DataFrame(df_f).loc[anova_cols].astype(float)
        df_f = pd.merge(df_f, metadata, left_index=True, right_on=sample_col)

        try:
            aov_f = pg.anova(data=df_f, dv=df_id, between=anova_factors, detailed=True)
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

        except Exception as e:
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
                f_stats_factors_df[f"ANOVA_[{anova_factor_name}]_pval"].fillna(1)
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
def pairwise_ttest(df, pairwise_ttest_groups):
    """_summary_

    Args:
        df (_type_): _description_
    """
    for pairwise_ttest_group in pairwise_ttest_groups:
        df[pairwise_ttest_group[0]] = (
            df[pairwise_ttest_group[4]].mean(axis=1)
            - df[pairwise_ttest_group[3]].mean(axis=1)
        ).fillna(0)
        df[f"{pairwise_ttest_group[0]}_pval"] = sp.stats.ttest_ind(
            df[pairwise_ttest_group[4]],
            df[pairwise_ttest_group[3]],
            axis=1,
            nan_policy="omit",
        ).pvalue
        df[f"{pairwise_ttest_group[0]}_adj-p"] = sp.stats.false_discovery_control(
            df[f"{pairwise_ttest_group[0]}_pval"].fillna(1)
        )
        df.loc[
            df[f"{pairwise_ttest_group[0]}_pval"].isna(),
            f"{pairwise_ttest_group[0]}_adj-p",
        ] = np.nan
    return df


# calculating the FC and p-values for protein abundances. See `abundance.py`
def calculate_pairwise_scalars(
    prot, pairwise_ttest_name=None, sig_type="pval", sig_thr=0.05
):
    """_summary_

    Args:
        prot (_type_): _description_
    """
    prot[f"{pairwise_ttest_name}_scalar"] = [
        prot[pairwise_ttest_name][i] if p < sig_thr else 0
        for i, p in enumerate(prot[f"{pairwise_ttest_name}_{sig_type}"])
    ]
    return prot


def calculate_all_pairwise_scalars(
    prot, pairwise_ttest_groups, sig_type="pval", sig_thr=0.05
):
    """_summary_

    Args:
        prot (_type_): _description_
        pairwise_ttest_groups (_type_): _description_
        sig_type (str, optional): _description_. Defaults to "pval".
        sig_thr (float, optional): _description_. Defaults to 0.05.

    Returns:
        _type_: _description_
    """
    for pairwise_ttest_group in pairwise_ttest_groups:
        prot = calculate_pairwise_scalars(
            prot, pairwise_ttest_group[0], sig_type, sig_thr
        )
    return prot
