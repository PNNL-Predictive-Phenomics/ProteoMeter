# type: ignore
import re

import numpy as np
import pandas as pd

from proteometer.peptide import nip_off_pept, strip_peptide
from proteometer.residue import (
    count_site_number,
    count_site_number_with_global_proteomics,
)


def get_ptm_pos_in_pept(
    peptide, ptm_label="*", special_chars=r".]+-=@_!#$%^&*()<>?/\|}{~:["
):
    peptide = nip_off_pept(peptide)
    if ptm_label in special_chars:
        ptm_label = "\\" + ptm_label
    ptm_pos = [m.start() for m in re.finditer(ptm_label, peptide)]
    pos = sorted([val - i - 1 for i, val in enumerate(ptm_pos)])
    return pos


def get_yst(strip_pept, ptm_aa="YSTyst"):
    return [
        [i, letter.upper()] for i, letter in enumerate(strip_pept) if letter in ptm_aa
    ]


def get_ptm_info(peptide, residue=None, prot_seq=None, ptm_label="*"):
    if prot_seq is not None:
        clean_pept = strip_peptide(peptide)
        pept_pos = prot_seq.find(clean_pept)
        all_yst = get_yst(clean_pept)
        all_ptm = [[pept_pos + yst[0] + 1, yst[1], yst[0]] for yst in all_yst]
        return all_ptm
    if residue is not None:
        subpept = nip_off_pept(peptide)
        split_substr = subpept.split(ptm_label)
        res_pos = sorted([int(res) for res in re.findall(r"\d+", residue)])
        first_pos = res_pos[0]
        res_pos.insert(0, first_pos - len(split_substr[0]))
        pept_pos = 0
        all_ptm = []
        for i, res in enumerate(res_pos):
            # print(i)
            if i > 0:
                pept_pos += len(split_substr[i - 1])
            yst_pos = get_yst(split_substr[i])
            if len(yst_pos) > 0:
                for j in yst_pos:
                    ptm = [j[0] + res + 1, j[1], pept_pos + j[0]]
                    all_ptm.append(ptm)
        return all_ptm


def get_phosphositeplus_pos(mod_rsd):
    return [int(re.sub(r"[^0-9]+", "", mod)) for mod in mod_rsd]


# To normalize the PTM data by the global protein medians
def ptm_TMT_normalization(df2transform, global_pept, int_cols):
    """_summary_

    Args:
        df2transform (_type_): _description_
    """
    raise DeprecationWarning("Deprecated")
    global_filtered = global_pept[global_pept[int_cols].isna().sum(axis=1) == 0].copy()
    global_medians = global_filtered[int_cols].median(axis=0, skipna=True)
    df_transformed = df2transform.copy()
    df_filtered = df2transform[df2transform[int_cols].isna().sum(axis=1) == 0].copy()
    df_medians = df_filtered[int_cols].median(axis=0, skipna=True).fillna(0)
    df_transformed[int_cols] = (
        df_transformed[int_cols].sub(global_medians, axis=1) + df_medians.mean()
    )
    return df_transformed


# Batch correction for PTM data
def ptm_batch_correction(
    df4batcor,
    metadata_ori,
    batch_correct_samples=None,
    batch_col="Batch",
    sample_col="Sample",
    **kwargs,
):
    raise DeprecationWarning("Deprecated")
    df = df4batcor.copy()
    metadata = metadata_ori.copy()
    if batch_correct_samples is None:
        batch_correct_samples = metadata[sample_col].to_list()
    batch_means = {}
    for batch in metadata[metadata[sample_col].isin(batch_correct_samples)][
        batch_col
    ].unique():
        df_batch = df[
            metadata[
                (metadata[batch_col] == batch)
                & (metadata[sample_col].isin(batch_correct_samples))
            ][sample_col]
        ].copy()
        # df_batch = df_batch[df_batch.isna().sum(axis=1) <= 0].copy()
        df_batch_means = df_batch.mean(axis=1).fillna(0)
        # print(f"Batch {batch} means: {df_batch_means}")
        # print(f"Batch {batch} mean: {df_batch_means.mean()}")
        batch_means.update({batch: df_batch_means})
    batch_means = pd.DataFrame(batch_means)
    batch_means_diffs = batch_means.sub(batch_means.mean(axis=1), axis=0)
    metadata.index = metadata["Sample"].to_list()
    for batch in metadata[metadata[sample_col].isin(batch_correct_samples)][
        batch_col
    ].unique():
        int_cols_per_batch = metadata[(metadata[batch_col] == batch)][sample_col]
        df[int_cols_per_batch] = df[int_cols_per_batch].sub(
            batch_means_diffs[batch], axis=0
        )
    # df = df.replace([np.inf, -np.inf], np.nan)
    return df


# Specific for PTM data. This is to roll up the PTM data to the site level
def ptm_rollup_to_site(
    df_ori,
    int_cols,
    uniprot_col,
    peptide_col,
    residue_col,
    residue_sep=";",
    id_col="id",
    id_separator="@",
    site_col="Site",
    multiply_rollup_counts=True,
    ignore_NA=True,
    rollup_func="median",
):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    raise DeprecationWarning("Deprecated")
    df = df_ori.reset_index(drop=True)
    info_cols = [col for col in df.columns if col not in int_cols]

    df[residue_col] = df[residue_col].str.split(residue_sep)
    df = df.explode(residue_col)
    df[id_col] = df[uniprot_col] + id_separator + df[residue_col]

    info_cols_wo_peptide_col = [col for col in info_cols if col != peptide_col]
    agg_methods_0 = {peptide_col: lambda x: "; ".join(x)}
    agg_methods_1 = {i: lambda x: x.iloc[0] for i in info_cols_wo_peptide_col}
    if multiply_rollup_counts:
        if ignore_NA:
            if rollup_func.lower() == "median":
                agg_methods_2 = {
                    i: lambda x: np.log2(len(x)) + x.median() for i in int_cols
                }
            elif rollup_func.lower() == "mean":
                agg_methods_2 = {
                    i: lambda x: np.log2(len(x)) + x.mean() for i in int_cols
                }
            elif rollup_func.lower() == "sum":
                agg_methods_2 = {
                    i: lambda x: np.log2(np.nansum(2 ** (x.replace(0, np.nan))))
                    for i in int_cols
                }
            else:
                ValueError(
                    "The rollup function is not recognized. Please choose from the following: median, mean, sum"
                )
        else:
            if rollup_func.lower() == "median":
                agg_methods_2 = {
                    i: lambda x: np.log2(x.notna().sum()) + x.median() for i in int_cols
                }
            elif rollup_func.lower() == "mean":
                agg_methods_2 = {
                    i: lambda x: np.log2(x.notna().sum()) + x.mean() for i in int_cols
                }
            elif rollup_func.lower() == "sum":
                agg_methods_2 = {
                    i: lambda x: np.log2(np.nansum(2 ** (x.replace(0, np.nan))))
                    for i in int_cols
                }
            else:
                ValueError(
                    "The rollup function is not recognized. Please choose from the following: median, mean, sum"
                )
    else:
        if rollup_func.lower() == "median":
            agg_methods_2 = {i: lambda x: x.median() for i in int_cols}
        elif rollup_func.lower() == "mean":
            agg_methods_2 = {i: lambda x: x.mean() for i in int_cols}
        elif rollup_func.lower() == "sum":
            agg_methods_2 = {
                i: lambda x: np.log2(np.nansum(2 ** (x.replace(0, np.nan))))
                for i in int_cols
            }
        else:
            ValueError(
                "The rollup function is not recognized. Please choose from the following: median, mean, sum"
            )
    df = df.groupby(id_col, as_index=False).agg(
        {**agg_methods_0, **agg_methods_1, **agg_methods_2}
    )
    df[site_col] = df[id_col].to_list()
    df[int_cols] = df[int_cols].replace([np.inf, -np.inf], np.nan)
    df.index = df[id_col].to_list()
    return df


# Specific for PTM data. This is to roll up the PTM data to the site level
def ptm_median_rollup_to_site(
    df_ori,
    int_cols,
    uniprot_col,
    peptide_col,
    residue_col,
    residue_sep=";",
    id_col="id",
    id_separator="@",
    site_col="Site",
    multiply_rollup_counts=True,
    ignore_NA=True,
):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    raise DeprecationWarning("Deprecated")
    df = df_ori.reset_index(drop=True)
    info_cols = [col for col in df.columns if col not in int_cols]

    df[residue_col] = df[residue_col].str.split(residue_sep)
    df = df.explode(residue_col)
    df[id_col] = df[uniprot_col] + id_separator + df[residue_col]

    info_cols_wo_peptide_col = [col for col in info_cols if col != peptide_col]
    agg_methods_0 = {peptide_col: lambda x: "; ".join(x)}
    agg_methods_1 = {i: lambda x: x.iloc[0] for i in info_cols_wo_peptide_col}
    if multiply_rollup_counts:
        if ignore_NA:
            agg_methods_2 = {
                i: lambda x: np.log2(len(x)) + x.median() for i in int_cols
            }
        else:
            agg_methods_2 = {
                i: lambda x: np.log2(x.notna().sum()) + x.median() for i in int_cols
            }
    else:
        agg_methods_2 = {i: lambda x: x.median() for i in int_cols}
    df = df.groupby(id_col, as_index=False).agg(
        {**agg_methods_0, **agg_methods_1, **agg_methods_2}
    )
    df[site_col] = df[id_col].to_list()
    df[int_cols] = df[int_cols].replace([np.inf, -np.inf], np.nan)
    df.index = df[id_col].to_list()
    return df


# Specific for PTM data. This is to roll up the PTM data to the site level
def ptm_mean_rollup_to_site(
    df_ori,
    int_cols,
    uniprot_col,
    peptide_col,
    residue_col,
    residue_sep=";",
    id_col="id",
    id_separator="@",
    site_col="Site",
    multiply_rollup_counts=True,
    ignore_NA=True,
):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    raise DeprecationWarning("Deprecated")
    df = df_ori.reset_index(drop=True)
    info_cols = [col for col in df.columns if col not in int_cols]

    df[residue_col] = df[residue_col].str.split(residue_sep)
    df = df.explode(residue_col)
    df[id_col] = df[uniprot_col] + id_separator + df[residue_col]

    info_cols_wo_peptide_col = [col for col in info_cols if col != peptide_col]
    agg_methods_0 = {peptide_col: lambda x: "; ".join(x)}
    agg_methods_1 = {i: lambda x: x.iloc[0] for i in info_cols_wo_peptide_col}
    if multiply_rollup_counts:
        if ignore_NA:
            agg_methods_2 = {i: lambda x: np.log2(len(x)) + x.mean() for i in int_cols}
        else:
            agg_methods_2 = {
                i: lambda x: np.log2(x.notna().sum()) + x.mean() for i in int_cols
            }
    else:
        agg_methods_2 = {i: lambda x: x.mean() for i in int_cols}
    df = df.groupby(id_col, as_index=False).agg(
        {**agg_methods_0, **agg_methods_1, **agg_methods_2}
    )
    df[site_col] = df[id_col].to_list()
    df[int_cols] = df[int_cols].replace([np.inf, -np.inf], np.nan)
    df.index = df[id_col].to_list()
    return df


def combine_multi_ptms(
    multi_proteomics,
    residue_col,
    uniprot_col,
    site_col,
    site_number_col,
    id_separator="@",
    id_col="id",
    type_col="Type",
    experiment_col="Experiment",
):
    proteomics_list = []
    for key, value in multi_proteomics.items():
        if key.lower() == "global":
            prot = value
            prot[type_col] = "Global"
            prot[experiment_col] = "PTM"
            prot[residue_col] = "GLB"
            prot[site_col] = prot[uniprot_col] + id_separator + prot[residue_col]
            proteomics_list.append(prot)
        elif key.lower() == "redox":
            redox = value
            redox[type_col] = "Ox"
            redox[experiment_col] = "PTM"
            redox = count_site_number(redox, uniprot_col, site_number_col)
            proteomics_list.append(redox)
        elif key.lower() == "phospho":
            phospho = value
            phospho[type_col] = "Ph"
            phospho[experiment_col] = "PTM"
            phospho = count_site_number(phospho, uniprot_col, site_number_col)
            proteomics_list.append(phospho)
        elif key.lower() == "acetyl":
            acetyl = value
            acetyl[type_col] = "Ac"
            acetyl[experiment_col] = "PTM"
            acetyl = count_site_number(acetyl, uniprot_col, site_number_col)
            proteomics_list.append(acetyl)
        else:
            KeyError(f"The key {key} is not recognized. Please check the input data.")

    all_ptms = (
        pd.concat(proteomics_list, axis=0, join="outer", ignore_index=True)
        .sort_values(by=[id_col, type_col, experiment_col, site_col])
        .reset_index(drop=True)
    )
    all_ptms = count_site_number_with_global_proteomics(
        all_ptms, uniprot_col, id_col, site_number_col
    )

    return all_ptms
