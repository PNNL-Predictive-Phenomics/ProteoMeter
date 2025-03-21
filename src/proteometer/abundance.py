# type: ignore
import pandas as pd
from stats import calculate_pairwise_scalars

import proteometer.normalization as normalization
import proteometer.stats as stats
from proteometer.params import Params


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


def prot_abund_correction(
    pept,
    prot,
    par: Params,
    columns_to_correct=None,
    pairwise_ttest_groups=None,
    non_tt_cols=None,
):
    if par.abundance_correction_paired_samples:
        if columns_to_correct is None:
            raise ValueError(
                "`columns_to_correct` is required for paired sample abundance correction."
            )
        return prot_abund_correction_matched(
            pept,
            prot,
            columns_to_correct,
            par.uniprot_col,
            non_tt_cols,
        )
    else:
        if pairwise_ttest_groups is None:
            raise ValueError(
                "`pairwise_ttest_groups` is required for unpaired sample abundance correction."
            )
        return prot_abund_correction_sig_only(
            pept,
            prot,
            pairwise_ttest_groups,
            par.uniprot_col,
            sig_thr=par.abudnance_unpaired_sig_thr,
        )


# correct the PTM or LiP data using the protein abundance scalars with
# significantly changed proteins only
def prot_abund_correction_sig_only(
    pept, prot, pairwise_ttest_groups, uniprot_col, sig_type="pval", sig_thr=0.05
):
    """_summary_

    Args:
        pept (_type_): _description_
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
        pept[f"{pairwise_ttest_group[0]}_scalar"] = [
            scalar_dict.get(uniprot_id, 0) for uniprot_id in pept[uniprot_col]
        ]
        pept[pairwise_ttest_group[4]] = pept[pairwise_ttest_group[4]].subtract(
            pept[f"{pairwise_ttest_group[0]}_scalar"], axis=0
        )
    return pept


# correct the PTM or LiP data using all protein abundance recommended approach
def prot_abund_correction_matched(
    pept, prot, columns_to_correct, uniprot_col, non_tt_cols=None
):
    """_summary_

    Args:
        pept (_type_): _description_
        prot (_type_): _description_
        columns_to_correct (_type_): _description_
        uniprot_col (_type_): _description_

    Returns:
        _type_: _description_
    """
    pept_new = []
    if non_tt_cols is None:
        non_tt_cols = columns_to_correct
    for uniprot_id in pept[uniprot_col].unique():
        pept_sub = pept[pept[uniprot_col] == uniprot_id].copy()
        if uniprot_id in prot[uniprot_col].unique():
            prot_abund_row = prot.loc[uniprot_id, columns_to_correct]
            prot_abund = prot_abund_row.fillna(0)
            prot_abund_median = prot_abund_row[non_tt_cols].median()
            if prot_abund_median:
                prot_abund_scale = (
                    prot_abund_row.div(prot_abund_row).fillna(0) * prot_abund_median
                )
            else:
                prot_abund_scale = prot_abund_row.div(prot_abund_row).fillna(0) * 0
            pept_sub[columns_to_correct] = (
                pept_sub[columns_to_correct]
                .sub(prot_abund, axis=1)
                .add(prot_abund_scale, axis=1)
            )
        pept_new.append(pept_sub)
    pept_new = pd.concat(pept_new)

    return pept_new


def global_prot_normalization_and_stats(
    global_prot: pd.DataFrame,
    int_cols: list[str],
    anova_cols: list[str],
    pairwise_ttest_groups,
    user_pairwise_ttest_groups,
    metadata: pd.DataFrame,
    par: Params,
):
    if not par.batch_correction:
        global_prot = normalization.median_normalization(global_prot, int_cols)
    else:
        # NB: median normalization is only for global proteomics data, PTM data
        # need to be normalized by global proteomics data
        global_prot = normalization.median_normalization(
            global_prot,
            int_cols,
            metadata,
            par.batch_correct_samples,
            batch_col=par.metadata_batch_col,
            sample_col=par.metadata_sample_col,
        )
        # Batch correction
        global_prot = normalization.batch_correction(
            global_prot,
            metadata,
            par.batch_correct_samples,
            batch_col=par.metadata_batch_col,
            sample_col=par.metadata_sample_col,
        )
    if len(par.groups) > 2:
        global_prot = stats.anova(global_prot, anova_cols, metadata)
        global_prot = stats.anova(global_prot, anova_cols, metadata, par.anova_factors)
    global_prot = stats.pairwise_ttest(global_prot, pairwise_ttest_groups)
    global_prot = stats.pairwise_ttest(global_prot, user_pairwise_ttest_groups)

    return global_prot
