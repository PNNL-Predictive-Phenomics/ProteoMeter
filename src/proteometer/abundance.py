# type: ignore
import pandas as pd
from stats import calculate_pairwise_scalars


def get_prot_abund_scalars(
    prot, pairwise_ttest_name=None, sig_type="pval", sig_thr=0.05
):
    """_summary_

    Args:
        prot (_type_): _description_
        pairwise_ttest_name (_type_, optional): _description_. Defaults to None.
        sig_type (str, optional): _description_. Defaults to "pval".
        sig_thr (float, optional): _description_. Defaults to 0.05.

    Returns:
        _type_: _description_
    """
    prot = calculate_pairwise_scalars(prot, pairwise_ttest_name, sig_type, sig_thr)
    scalar_dict = dict(zip(prot.index, prot[f"{pairwise_ttest_name}_scalar"]))
    return scalar_dict


# correct the PTM or LiP data using the protein abundance scalars with significantly changed proteins only
def prot_abund_correction_sig_only(
    df, prot, pairwise_ttest_groups, uniprot_col, sig_type="pval", sig_thr=0.05
):
    """_summary_

    Args:
        df (_type_): _description_
        prot (_type_): _description_
        pairwise_ttest_groups (_type_): _description_
        uniprot_col (_type_): _description_
        sig_type (str, optional): _description_. Defaults to "pval".
        sig_thr (float, optional): _description_. Defaults to 0.05.

    Returns:
        _type_: _description_
    """
    for pairwise_ttest_group in pairwise_ttest_groups:
        if pairwise_ttest_group[0] not in prot.columns:
            scalar_dict = get_prot_abund_scalars(
                prot, pairwise_ttest_group[0], sig_type, sig_thr
            )
        else:
            scalar_dict = dict(
                zip(prot.index, prot[f"{pairwise_ttest_group[0]}_scalar"])
            )
        df[f"{pairwise_ttest_group[0]}_scalar"] = [
            scalar_dict.get(uniprot_id, 0) for uniprot_id in df[uniprot_col]
        ]
        df[pairwise_ttest_group[4]] = df[pairwise_ttest_group[4]].subtract(
            df[f"{pairwise_ttest_group[0]}_scalar"], axis=0
        )
    return df


# correct the PTM or LiP data using all protein abundance recommended approach
def prot_abund_correction(pept, prot, cols2correct, uniprot_col, non_tt_cols=None):
    """_summary_

    Args:
        pept (_type_): _description_
        prot (_type_): _description_
        cols2correct (_type_): _description_
        uniprot_col (_type_): _description_

    Returns:
        _type_: _description_
    """
    pept_new = []
    if non_tt_cols is None:
        non_tt_cols = cols2correct
    for uniprot_id in pept[uniprot_col].unique():
        pept_sub = pept[pept[uniprot_col] == uniprot_id].copy()
        if uniprot_id in prot[uniprot_col].unique():
            prot_abund_row = prot.loc[uniprot_id, cols2correct]
            prot_abund = prot_abund_row.fillna(0)
            prot_abund_median = prot_abund_row[non_tt_cols].median()
            if prot_abund_median:
                prot_abund_scale = (
                    prot_abund_row.div(prot_abund_row).fillna(0) * prot_abund_median
                )
            else:
                prot_abund_scale = prot_abund_row.div(prot_abund_row).fillna(0) * 0
            pept_sub[cols2correct] = (
                pept_sub[cols2correct]
                .sub(prot_abund, axis=1)
                .add(prot_abund_scale, axis=1)
            )
        pept_new.append(pept_sub)
    pept_new = pd.concat(pept_new)

    return pept_new


# Alias the function for PTM data
def prot_abund_correction_TMT(pept, prot, cols2correct, uniprot_col, non_tt_cols=None):
    return prot_abund_correction(pept, prot, cols2correct, uniprot_col, non_tt_cols)
