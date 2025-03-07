# %%
# from anyio import value
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from Bio import SeqIO
import pingouin as pg

import os
import re
import math
from pathlib import Path
HOME = str(Path.home())

# %%
############################################
# General functions
############################################

colors = {
    'blue':    '#377eb8', 
    'orange':  '#ff7f00',
    'green':   '#4daf4a',
    'pink':    '#f781bf',
    'brown':   '#a65628',
    'purple':  '#984ea3',
    'gray':    '#999999',
    'red':     '#e41a1c',
    'yellow':  '#dede00'
}


# %%
def IsDefined(x):
    try:
        x
    except NameError:
        return False
    else:
        return True


# %%
def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])


# %%
def nip_off_pept(peptide):
    # pept_pattern = "\\.(.+)\\."
    # is equivalent to
    pept_pattern = r"\.(.+)\."
    subpept = re.search(pept_pattern, peptide).group(1)
    return subpept


# %%
def strip_peptide(peptide, nip_off=True):
    if nip_off:
        return re.sub(r"[^A-Za-z]+", '', nip_off_pept(peptide))
    else:
        return re.sub(r"[^A-Za-z]+", '', peptide)


# %%
def get_ptm_pos_in_pept(
    peptide, ptm_label='*', special_chars=r'.]+-=@_!#$%^&*()<>?/\|}{~:['
):
    peptide = nip_off_pept(peptide)
    if ptm_label in special_chars:
        ptm_label = '\\' + ptm_label
    ptm_pos = [m.start() for m in re.finditer(ptm_label, peptide)]
    pos = sorted([val - i - 1 for i, val in enumerate(ptm_pos)])
    return pos


# %%
def get_yst(strip_pept, ptm_aa="YSTyst"):
    return [
        [i, letter.upper()] for i, letter in enumerate(strip_pept) if letter in ptm_aa
    ]


# %%
def get_ptm_info(peptide, residue=None, prot_seq=None, ptm_label='*'):
    if prot_seq != None:
        clean_pept = strip_peptide(peptide)
        pept_pos = prot_seq.find(clean_pept)
        all_yst = get_yst(clean_pept)
        all_ptm = [[pept_pos + yst[0] + 1, yst[1], yst[0]] for yst in all_yst]
        return all_ptm
    if residue != None:
        subpept = nip_off_pept(peptide)
        split_substr = subpept.split(ptm_label)
        res_pos = sorted([int(res) for res in re.findall(r'\d+', residue)])
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
                    ptm = [j[0] + res_pos[i] + 1, j[1], pept_pos + j[0]]
                    all_ptm.append(ptm)
        return all_ptm


# %%
def relable_pept(peptide, label_pos, ptm_label='*'):
    strip_pept = strip_peptide(peptide)
    for i, pos in enumerate(label_pos):
        strip_pept = (
            strip_pept[: (pos + i + 1)] + ptm_label + strip_pept[(pos + i + 1):]
        )
    return peptide[:2] + strip_pept + peptide[-2:]


# %%
def get_phosphositeplus_pos(mod_rsd):
    return [int(re.sub(r"[^0-9]+", '', mod)) for mod in mod_rsd]


# %%
def get_res_names(residues):
    res_names = [
        [res for res in re.findall(r'[A-Z]\d+[a-z\-]+', residue)]
        if residue[0] != 'P'
        else [residue]
        for residue in residues
    ]
    return res_names


# %%
def get_res_pos(residues):
    res_pos = [
        [int(res) for res in re.findall(r'\d+', residue)] if residue[0] != 'P' else [0]
        for residue in residues
    ]
    return res_pos


# %%
def get_sequences_from_fasta(fasta_file):
    prot_seq_obj = SeqIO.parse(fasta_file, "fasta")
    prot_seqs = [seq_item for seq_item in prot_seq_obj]
    return prot_seqs


# %%
def plot_barcode(pal, ticklabel=None, barcode_name=None, ax=None, size=(10, 2)):
    """Plot the values in a color palette as a horizontal array.
    Parameters
    ----------
    pal : sequence of matplotlib colors
        colors, i.e. as returned by seaborn.color_palette()
    size :
        figure size of plot
    ax :
        an existing axes to use
    """
    n = len(pal)
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=size)
    ax.imshow(
        np.arange(n).reshape(1, n),
        cmap=mpl.colors.ListedColormap(list(pal)),
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_yticks([0])
    ax.set_yticklabels([barcode_name])
    # The proper way to set no ticks
    # ax.yaxis.set_major_locator(ticker.NullLocator())
    # ax.set_xticks(np.arange(n) - .5)
    # ax.set_xticks(np.arange(n))
    ax.set_xticks(np.arange(0, n, np.ceil(n / len(ticklabel)).astype("int")))
    # Ensure nice border between colors
    # ax.set_xticklabels(["" for _ in range(n)])
    ax.set_xticklabels(ticklabel)
    # return ax


# %%
def get_barcode(fc_bar, color_levels=20, fc_bar_max=None):
    # fc_bar = copy.deepcopy(res_fc_diff[["FC_DIFF", "FC_TYPE", "Res"]])
    both_pal_vals = sns.color_palette("Greens", color_levels)
    up_pal_vals = sns.color_palette("Reds", color_levels)
    down_pal_vals = sns.color_palette("Blues", color_levels)
    insig_pal_vals = sns.color_palette("Greys", color_levels)
    if fc_bar_max == None:
        fc_bar_max = fc_bar["FC_DIFF"].abs().max()
    bar_code = []
    for i in range(fc_bar.shape[0]):
        if fc_bar.iloc[i, 1] == "both":
            bar_code.append(
                both_pal_vals[
                    np.ceil(abs(fc_bar.iloc[i, 0]) / fc_bar_max * color_levels).astype(
                        "int"
                    )
                    - 1
                ]
            )
        elif fc_bar.iloc[i, 1] == "up":
            bar_code.append(
                up_pal_vals[
                    np.ceil(abs(fc_bar.iloc[i, 0]) / fc_bar_max * color_levels).astype(
                        "int"
                    )
                    - 1
                ]
            )
        elif fc_bar.iloc[i, 1] == "down":
            bar_code.append(
                down_pal_vals[
                    np.ceil(abs(fc_bar.iloc[i, 0]) / fc_bar_max * color_levels).astype(
                        "int"
                    )
                    - 1
                ]
            )
        elif fc_bar.iloc[i, 1] == "insig":
            bar_code.append(
                insig_pal_vals[
                    np.ceil(abs(fc_bar.iloc[i, 0]) / fc_bar_max * color_levels).astype(
                        "int"
                    )
                    - 1
                ]
            )
        else:
            bar_code.append((0, 0, 0))
    return bar_code


# %%
def get_protein_res(proteome, uniprot_id, prot_seqs):
    protein = proteome[proteome["uniprot_id"] == uniprot_id]
    protein.reset_index(drop=True, inplace=True)
    prot_seq_search = [seq for seq in prot_seqs if seq.id == uniprot_id]
    prot_seq = prot_seq_search[0]
    sequence = str(prot_seq.seq)
    clean_pepts = [strip_peptide(pept) for pept in protein["peptide"].to_list()]
    protein["clean_pept"] = clean_pepts
    pept_start = [sequence.find(clean_pept) for clean_pept in clean_pepts]
    pept_end = [
        sequence.find(clean_pept) + len(clean_pept) for clean_pept in clean_pepts
    ]
    protein["pept_start"] = pept_start
    protein["pept_end"] = pept_end
    protein["residue"] = [
        [res + str(sequence.find(clean_pept) + i) for i, res in enumerate(clean_pept)]
        for clean_pept in clean_pepts
    ]
    protein_res = protein.explode("residue")
    protein_res.reset_index(drop=True, inplace=True)
    return protein_res


# %%
def adjusted_p_value(pd_series, ignore_na=True, filling_val=1):
    output = pd_series.copy()
    if pd_series.isna().sum() > 0:
        # print("NAs present in pd_series.")
        if ignore_na:
            print("Ignoring NAs.")
            # pd_series =
        else:
            # print("Filling NAs with " + str(filling_val))
            output = sp.stats.false_discovery_control(pd_series.fillna(filling_val))
    else:
        # print("No NAs present in pd_series.")
        output = sp.stats.false_discovery_control(pd_series)
    return output


# %%
############################################
# For both PTM and LiP
############################################


# %%
# This part is to generate the index of dataframes
def generate_index(df, prot_col, level_col=None, id_separator='@', id_col="id"):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    if level_col is None:
        df[id_col] = df[prot_col]
    else:
        df[id_col] = df[prot_col] + id_separator + df[level_col]
    df.index = df[id_col].to_list()
    return df


# %%
# this is the function to do the log2 transformation
def log2_transformation(df2transform, int_cols):
    """_summary_

    Args:
        df2transform (_type_): _description_
    """
    df2transform[int_cols] = np.log2(df2transform[int_cols].replace(0, np.nan))
    return df2transform


# %%
# Rolling up the quantification by using log2 medians and plus the log2 of peptide number
def rollup_to_peptide(df_ori, int_cols, rollup_col, id_col="id", id_separator='@', multiply_rollup_counts=True, ignore_NA=True, rollup_func="sum"):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = df_ori.reset_index(drop=True)
    info_cols = [col for col in df.columns if col not in int_cols]

    df[id_col] = df[id_col] + id_separator + df[rollup_col]

    df[int_cols] = 2**df[int_cols]
    df[int_cols] = df[int_cols].fillna(0)

    agg_methods_1 = {i: lambda x: x.iloc[0] for i in info_cols}
    if multiply_rollup_counts:
        if ignore_NA:
            if rollup_func.lower() == "median":
                agg_methods_2 = {i: lambda x: np.log2(len(x)) + x.median() for i in int_cols}
            elif rollup_func.lower() == "mean":
                agg_methods_2 = {i: lambda x: np.log2(len(x)) + x.mean() for i in int_cols}
            elif rollup_func.lower() == "sum":
                agg_methods_2 = {i: lambda x: np.log2(np.nansum(2**(x.replace(0, np.nan)))) for i in int_cols}
            else:
                ValueError("The rollup function is not recognized. Please choose from the following: median, mean, sum")
        else:
            if rollup_func.lower() == "median":
                agg_methods_2 = {i: lambda x: np.log2(x.notna().sum()) + x.median() for i in int_cols}
            elif rollup_func.lower() == "mean":
                agg_methods_2 = {i: lambda x: np.log2(x.notna().sum()) + x.mean() for i in int_cols}
            elif rollup_func.lower() == "sum":
                agg_methods_2 = {i: lambda x: np.log2(np.nansum(2**(x.replace(0, np.nan)))) for i in int_cols}
            else:
                ValueError("The rollup function is not recognized. Please choose from the following: median, mean, sum")
    else:
        if rollup_func.lower() == "median":
            agg_methods_2 = {i: lambda x: x.median() for i in int_cols}
        elif rollup_func.lower() == "mean":
            agg_methods_2 = {i: lambda x: x.mean() for i in int_cols}
        elif rollup_func.lower() == "sum":
            agg_methods_2 = {i: lambda x: np.log2(np.nansum(2**(x.replace(0, np.nan)))) for i in int_cols}
        else:
            ValueError("The rollup function is not recognized. Please choose from the following: median, mean, sum")
    df = df.groupby(id_col, as_index=False).agg({**agg_methods_1, **agg_methods_2})
    df[int_cols] = np.log2(df[int_cols])
    df[int_cols] = df[int_cols].replace([np.inf, -np.inf], np.nan)
    df.index = df[id_col].to_list()
    return df


# %%
# Here is the function to normalize the data
def median_normalization(df2transform, int_cols, metadata_ori=None, batch_correct_samples=None, batch_col=None, sample_col="Sample", skipna=True, zero_center=False):
    """_summary_

    Args:
        df2transform (_type_): _description_
        int_cols (_type_): _description_

    Returns:
        _type_: _description_
    """
    df_transformed = df2transform.copy()
    if batch_col is None or metadata_ori is None:
        if skipna:
            df_filtered = df_transformed[df_transformed[int_cols].isna().sum(axis=1) == 0].copy()
        else:
            df_filtered = df_transformed.copy()

        if zero_center:
            median_correction_T = df_filtered[int_cols].median(axis=0, skipna=True).fillna(0)
        else:
            median_correction_T = df_filtered[int_cols].median(axis=0, skipna=True).fillna(0) - df_filtered[int_cols].median(axis=0, skipna=True).fillna(0).mean()
        df_transformed[int_cols] = df_transformed[int_cols].sub(median_correction_T, axis=1)
        # df_transformed = df_transformed.replace([np.inf, -np.inf], np.nan)
        return df_transformed
    
    metadata = metadata_ori.copy()
    if batch_correct_samples is None:
        batch_correct_samples = metadata[sample_col].to_list()
    for batch in metadata[metadata[sample_col].isin(batch_correct_samples)][batch_col].unique():
        int_cols_per_batch = metadata[(metadata[batch_col] == batch)][sample_col] 
        if skipna:
            df_filtered = df_transformed[df_transformed[int_cols_per_batch].isna().sum(axis=1) == 0].copy()
        else:
            df_filtered = df_transformed.copy()

        if zero_center:
            median_correction_T = df_filtered[int_cols_per_batch].median(axis=0, skipna=True).fillna(0)
        else:
            median_correction_T = df_filtered[int_cols_per_batch].median(axis=0, skipna=True).fillna(0) - df_filtered[int_cols_per_batch].median(axis=0, skipna=True).fillna(0).mean()
        df_transformed[int_cols_per_batch] = df_transformed[int_cols_per_batch].sub(median_correction_T, axis=1)
    # df_transformed = df_transformed.replace([np.inf, -np.inf], np.nan)
    return df_transformed


# %%
# To normalize the TMT peptide data by the global peptide medians
def TMT_normalization(df2transform, global_pept, int_cols):
    """_summary_

    Args:
        df2transform (_type_): _description_
    """
    global_filtered = global_pept[global_pept[int_cols].isna().sum(axis=1) == 0].copy()
    global_medians = global_filtered[int_cols].median(axis=0, skipna=True)
    df_transformed = df2transform.copy()
    df_filtered = df2transform[df2transform[int_cols].isna().sum(axis=1) == 0].copy()
    df_medians = df_filtered[int_cols].median(axis=0, skipna=True).fillna(0)
    df_transformed[int_cols] = df_transformed[int_cols].sub(global_medians, axis=1) + df_medians.mean()
    return df_transformed


# %%
# Batch correction for PTM data
def batch_correction(df4batcor, metadata_ori, batch_correct_samples=None, batch_col="Batch", sample_col="Sample", **kwargs):
    df = df4batcor.copy()
    metadata = metadata_ori.copy()
    if batch_correct_samples is None:
        batch_correct_samples = metadata[sample_col].to_list()
    batch_means = {}
    for batch in metadata[metadata[sample_col].isin(batch_correct_samples)][batch_col].unique():
        df_batch = df[metadata[(metadata[batch_col] == batch) & (metadata[sample_col].isin(batch_correct_samples))][sample_col]].copy()
        # df_batch = df_batch[df_batch.isna().sum(axis=1) <= 0].copy()
        df_batch_means = df_batch.mean(axis=1).fillna(0)
        # print(f"Batch {batch} means: {df_batch_means}")
        # print(f"Batch {batch} mean: {df_batch_means.mean()}")
        batch_means.update({batch:  df_batch_means})
    batch_means = pd.DataFrame(batch_means)
    batch_means_diffs = batch_means.sub(batch_means.mean(axis=1), axis=0)
    metadata.index = metadata[sample_col].to_list()
    for batch in metadata[metadata[sample_col].isin(batch_correct_samples)][batch_col].unique():
        int_cols_per_batch = metadata[(metadata[batch_col] == batch)][sample_col] 
        df[int_cols_per_batch] = df[int_cols_per_batch].sub(batch_means_diffs[batch], axis=0)
    # df = df.replace([np.inf, -np.inf], np.nan)
    return df


# %%
# Check the missingness by groups
def check_missingness(df, groups, group_cols):
    """_summary_

    Args:
        df (_type_): _description_
    """
    df["Total missingness"] = 0
    for name, cols in zip(groups, group_cols):
        df[f"{name} missingness"] = df[cols].isna().sum(axis=1)
        df["Total missingness"] = df["Total missingness"] + df[f"{name} missingness"]
    return df


# %%
# Here is the function to filter the missingness
def filter_missingness(df, groups, group_cols, missing_thr=0.0):
    """_summary_

    Args:
        df (_type_): _description_
        groups (_type_): _description_
        group_cols (_type_): _description_
        missing_thr (float, optional): _description_. Defaults to 0.0.

    Returns:
        _type_: _description_
    """
    df = check_missingness(df, groups, group_cols)

    df["missing_check"] = 0
    for name, cols in zip(groups, group_cols):
        df["missing_check"] = df["missing_check"] + (df[f"{name} missingness"] > missing_thr * len(cols)).astype(int)
    df_w = df[~(df["missing_check"] > 0)].copy()
    return df_w


# %%
# Specific for PTM data. This is to roll up the PTM data to the site level
def rollup_to_site(df_ori, int_cols, uniprot_col, peptide_col, residue_col, residue_sep=';', id_col="id", id_separator='@', site_col="Site", multiply_rollup_counts=True, ignore_NA=True, rollup_func="sum"):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = df_ori.reset_index(drop=True)
    info_cols = [col for col in df.columns if col not in int_cols]

    df[residue_col] = df[residue_col].str.split(residue_sep)
    df = df.explode(residue_col)
    df[id_col] = df[uniprot_col] + id_separator + df[residue_col]

    info_cols_wo_peptide_col = [col for col in info_cols if col != peptide_col]
    agg_methods_0 = {peptide_col: lambda x: '; '.join(x)}
    agg_methods_1 = {i: lambda x: x.iloc[0] for i in info_cols_wo_peptide_col}
    if multiply_rollup_counts:
        if ignore_NA:
            if rollup_func.lower() == "median":
                agg_methods_2 = {i: lambda x: np.log2(len(x)) + x.median() for i in int_cols}
            elif rollup_func.lower() == "mean":
                agg_methods_2 = {i: lambda x: np.log2(len(x)) + x.mean() for i in int_cols}
            elif rollup_func.lower() == "sum":
                agg_methods_2 = {i: lambda x: np.log2(np.nansum(2**(x.replace(0, np.nan)))) for i in int_cols}
            else:
                ValueError("The rollup function is not recognized. Please choose from the following: median, mean, sum")
        else:
            if rollup_func.lower() == "median":
                agg_methods_2 = {i: lambda x: np.log2(x.notna().sum()) + x.median() for i in int_cols}
            elif rollup_func.lower() == "mean":
                agg_methods_2 = {i: lambda x: np.log2(x.notna().sum()) + x.mean() for i in int_cols}
            elif rollup_func.lower() == "sum":
                agg_methods_2 = {i: lambda x: np.log2(np.nansum(2**(x.replace(0, np.nan)))) for i in int_cols}
            else:
                ValueError("The rollup function is not recognized. Please choose from the following: median, mean, sum")
    else:
        if rollup_func.lower() == "median":
            agg_methods_2 = {i: lambda x: x.median() for i in int_cols}
        elif rollup_func.lower() == "mean":
            agg_methods_2 = {i: lambda x: x.mean() for i in int_cols}
        elif rollup_func.lower() == "sum":
            agg_methods_2 = {i: lambda x: np.log2(np.nansum(2**(x.replace(0, np.nan)))) for i in int_cols}
        else:
            ValueError("The rollup function is not recognized. Please choose from the following: median, mean, sum")
    df = df.groupby(id_col, as_index=False).agg({**agg_methods_0, **agg_methods_1, **agg_methods_2})
    df[site_col] = df[id_col].to_list()
    df[int_cols] = df[int_cols].replace([np.inf, -np.inf], np.nan)
    df.index = df[id_col].to_list()
    return df


# %%
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
    metadata = metadata_ori[metadata_ori[sample_col].isin(anova_cols)].copy()

    # df = df.drop(columns=["ANOVA_[one-way]_pval", "ANOVA_[one-way]_adj-p"], errors='ignore')

    if len(anova_factors) < 1:
        print("The anova_factors is empty. Please provide the factors for ANOVA analysis. The default factor is 'Group'.")
        anova_factors = ["Group"]
    anova_factor_names = [f"{anova_factors[i]} * {anova_factors[j]}" if i != j else f"{anova_factors[i]}" for i in range(len(anova_factors)) for j in range(i, len(anova_factors))]

    df_w = df[anova_cols].copy()
    # f_stats = []
    f_stats_factors = []
    for row in df_w.iterrows():
        df_id = row[0]
        df_f = row[1]
        df_f = pd.DataFrame(df_f).loc[anova_cols].astype(float)
        df_f = pd.merge(df_f, metadata, left_index=True, right_on=sample_col)

        # aov = pg.anova(data=df_f, dv=df_id, between=oneway_factor, detailed=True)
        # if "p-unc" in aov.columns:
        #     p_val = aov[aov["Source"] == oneway_factor]["p-unc"].values[0]
        # else:
        #     p_val = np.nan
        # f_stats.append(pd.DataFrame({"id": [df_id], f"ANOVA_[{oneway_factor}]_pval": [p_val]}))
        try:
            aov_f = pg.anova(data=df_f, dv=df_id, between=anova_factors, detailed=True)
            if "p-unc" in aov_f.columns:
                p_vals = {f"ANOVA_[{anova_factor_name}]_pval": aov_f[aov_f["Source"] == anova_factor_name]["p-unc"].values[0] for anova_factor_name in anova_factor_names}
            else:
                p_vals = {f"ANOVA_[{anova_factor_name}]_pval": np.nan for anova_factor_name in anova_factor_names}
        # except AssertionError as e:
        except Exception as e:
            Warning(f"ANOVA failed for {df_id}: {e}")
            p_vals = {f"ANOVA_[{anova_factor_name}]_pval": np.nan for anova_factor_name in anova_factor_names}
        f_stats_factors.append(pd.DataFrame({"id": [df_id]} | p_vals))

    # f_stats_df = pd.concat(f_stats).reset_index(drop=True)
    # f_stats_df[f"ANOVA_[{oneway_factor}]_adj-p"] = sp.stats.false_discovery_control(f_stats_df[f"ANOVA_[{oneway_factor}]_pval"].fillna(1))
    # f_stats_df.loc[f_stats_df[f"ANOVA_{oneway_factor}_pval"].isna(), f"ANOVA_{oneway_factor}_adj-p"] = np.nan
    # f_stats_df.set_index("id", inplace=True)

    f_stats_factors_df = pd.concat(f_stats_factors).reset_index(drop=True)
    for anova_factor_name in anova_factor_names:
        f_stats_factors_df[f"ANOVA_[{anova_factor_name}]_adj-p"] = sp.stats.false_discovery_control(f_stats_factors_df[f"ANOVA_[{anova_factor_name}]_pval"].fillna(1))
        f_stats_factors_df.loc[f_stats_factors_df[f"ANOVA_[{anova_factor_name}]_pval"].isna(), f"ANOVA_[{anova_factor_name}]_adj-p"] = np.nan
    f_stats_factors_df.set_index("id", inplace=True)
    # f_stats_factors_df.index = f_stats_factors_df["id"].to_list()

    # df = pd.merge(df, f_stats_df, left_index=True, right_index=True)
    df = pd.merge(df, f_stats_factors_df, left_index=True, right_index=True)

    return df


# %%
# Here is the function to do the t-test
# This is same for both protide and protein as well as rolled up protein data. Hopefully this is also the same for PTM data
def pairwise_ttest(df, pairwise_ttest_groups):
    """_summary_

    Args:
        df (_type_): _description_
    """
    for pairwise_ttest_group in pairwise_ttest_groups:
        df[pairwise_ttest_group[0]] = (df[pairwise_ttest_group[4]].mean(axis=1) - df[pairwise_ttest_group[3]].mean(axis=1)).fillna(0)
        df[f"{pairwise_ttest_group[0]}_pval"] = sp.stats.ttest_ind(df[pairwise_ttest_group[4]], df[pairwise_ttest_group[3]], axis=1, nan_policy='omit').pvalue
        df[f"{pairwise_ttest_group[0]}_adj-p"] = sp.stats.false_discovery_control(df[f"{pairwise_ttest_group[0]}_pval"].fillna(1))
        df.loc[df[f"{pairwise_ttest_group[0]}_pval"].isna(), f"{pairwise_ttest_group[0]}_adj-p"] = np.nan
    return df


# %%
# calculating the FC and p-values for protein abundances.
def calculate_pairwise_scalars(prot, pairwise_ttest_name=None, sig_type="pval", sig_thr=0.05):
    """_summary_

    Args:
        prot (_type_): _description_
    """
    prot[f"{pairwise_ttest_name}_scalar"] = [prot[pairwise_ttest_name][i] if p < sig_thr else 0 for i, p in enumerate(prot[f"{pairwise_ttest_name}_{sig_type}"])]
    return prot


# %%
def get_prot_abund_scalars(prot, pairwise_ttest_name=None, sig_type="pval", sig_thr=0.05):
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


# %%
def calculate_all_pairwise_scalars(prot, pairwise_ttest_groups, sig_type="pval", sig_thr=0.05):
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
        prot = calculate_pairwise_scalars(prot, pairwise_ttest_group[0], sig_type, sig_thr)
    return prot


# %%
# correct the PTM or LiP data using the protein abundance scalars with significantly changed proteins only
def prot_abund_correction_sig_only(df, prot, pairwise_ttest_groups, uniprot_col, sig_type="pval", sig_thr=0.05):
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
            scalar_dict = get_prot_abund_scalars(prot, pairwise_ttest_group[0], sig_type, sig_thr)
        else:
            scalar_dict = dict(zip(prot.index, prot[f"{pairwise_ttest_group[0]}_scalar"]))
        df[f"{pairwise_ttest_group[0]}_scalar"] = [scalar_dict.get(uniprot_id, 0) for uniprot_id in df[uniprot_col]]
        df[pairwise_ttest_group[4]] = df[pairwise_ttest_group[4]].subtract(df[f"{pairwise_ttest_group[0]}_scalar"], axis=0)
    return df


# %%
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
                prot_abund_scale = prot_abund_row.div(prot_abund_row).fillna(0) * prot_abund_median
            else:
                prot_abund_scale = prot_abund_row.div(prot_abund_row).fillna(0) * 0
            pept_sub[cols2correct] = pept_sub[cols2correct].sub(prot_abund, axis=1).add(prot_abund_scale, axis=1)
        pept_new.append(pept_sub)
    pept_new = pd.concat(pept_new)

    return pept_new


# Alias the function for PTM data
def prot_abund_correction_TMT(pept, prot, cols2correct, uniprot_col, non_tt_cols=None):
    return prot_abund_correction(pept, prot, cols2correct, uniprot_col, non_tt_cols)


# %%
def count_site_number(df, uniprot_col, site_number_col="site_number"):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    site_number = df.groupby(uniprot_col).size()
    site_number.name = site_number_col
    df = pd.merge(df, site_number, left_on=uniprot_col, right_index=True)
    return df


# %%
def count_site_number_with_global_proteomics(df, uniprot_col, id_col, site_number_col="site_number"):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    site_number = df.groupby(uniprot_col).size() - 1
    site_number.name = site_number_col
    for uniprot in site_number.index:
        df.loc[df[id_col] == uniprot, site_number_col] = site_number[uniprot]
    return df


# %%
###########################################
# Specific for PTM
############################################


# %%
# To remove blank columns from TMT tables
def remove_blank_cols(df, blank_cols=None):
    """_summary_

    Args:
        df (_type_): _description_
        blank_cols (_type_): _description_

    Returns:
        _type_: _description_
    """
    if blank_cols is None:
        blank_cols = [col for col in df.columns if 'blank' in col.lower()]
    return df.drop(columns=blank_cols, errors='ignore')


def combine_multi_PTMs(multi_proteomics, residue_col, uniprot_col, site_col, site_number_col, id_separator='@', id_col="id", type_col="Type", experiment_col="Experiment"):
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

    all_ptms = pd.concat(proteomics_list, axis=0, join='outer', ignore_index=True).sort_values(by=[id_col, type_col, experiment_col, site_col]).reset_index(drop=True)
    all_ptms = count_site_number_with_global_proteomics(all_ptms, uniprot_col, id_col, site_number_col)

    return all_ptms


# %%
############################################
# Specific to LiP
############################################


# %%
# This part is filtering all the contaminants and reverse hits
def filter_contaminants_reverse_pept(df, search_tool, ProteinID_col_pept, uniprot_col):
    """_summary_

    Args:
        df (_type_): _description_
    """
    if search_tool.lower() == "maxquant":
        df = df[(df["Reverse"].isna()) & (df["Potential contaminant"].isna()) & (~df[ProteinID_col_pept].str.contains("(?i)Contaminant")) & (~df[ProteinID_col_pept].str.contains("(?i)REV__")) & (~df[ProteinID_col_pept].str.contains("(?i)CON__"))].copy()
        df[uniprot_col] = df[ProteinID_col_pept]
    elif search_tool.lower() == "msfragger" or search_tool.lower() == "fragpipe":
        df = df[(~df[ProteinID_col_pept].str.contains("(?i)contam_"))].copy()
        df[uniprot_col] = df[ProteinID_col_pept]
    else:
        print("The search tool is not specified or not supported yet. The user should provide the tables that have been filtered and without the contaminants and reverse hits.")

    return df


# %%
def filter_contaminants_reverse_prot(df, search_tool, ProteinID_col_prot, uniprot_col):
    """_summary_

    Args:
        df (_type_): _description_
    """
    if search_tool.lower() == "maxquant":
        df = df[(df["Only identified by site"].isna()) & (df["Reverse"].isna()) & (df["Potential contaminant"].isna()) & (~df[ProteinID_col_prot].str.contains("(?i)Contaminant")) & (~df[ProteinID_col_prot].str.contains("(?i)REV__")) & (~df[ProteinID_col_prot].str.contains("CON__"))].copy()
        df[uniprot_col] = [ids.split(';')[0] for ids in df[ProteinID_col_prot]]
    elif search_tool.lower() == "msfragger" or search_tool.lower() == "fragpipe":
        df = df[(~df[ProteinID_col_prot].str.contains("(?i)contam_"))].copy()
        df[uniprot_col] = df[ProteinID_col_prot]
    else:
        print("The search tool is not specified or not supported yet. The user should provide the tables that have been filtered and without the contaminants and reverse hits.")

    return df


# %%
# Filtering out the protein groups with less than 2 peptides
def filtering_protein_based_on_peptide_number(df2filter, PeptCounts_col, search_tool, min_pept_count=2):
    """_summary_

    Args:
        df2filter (_type_): _description_
    """
    if search_tool.lower() == "maxquant":
        df2filter["Pept count"] = [int(count.split(';')[0]) for count in df2filter[PeptCounts_col]]
    elif search_tool.lower() == "msfragger" or search_tool.lower() == "fragpipe":
        df2filter["Pept count"] = df2filter[PeptCounts_col]
    else:
        print("The search tool is not specified or not supported yet. The user should provide the tables that have been filtered and without the contaminants and reverse hits.")
    df2filter = df2filter[df2filter["Pept count"] >= min_pept_count].copy()
    return df2filter


# %%
# This function analyze the trypic pattern of the peptides in pept dataframe
def get_clean_peptides(pept_df, peptide_col, clean_pept_col = "clean_pept"):
    """_summary_

    Args:
        pept_df (_type_): _description_
        peptide_col (_type_): _description_

    Returns:
        _type_: _description_
    """
    clean_pepts = [strip_peptide(pept, nip_off=False) for pept in pept_df[peptide_col].to_list()]
    pept_df[clean_pept_col] = clean_pepts
    return pept_df


# %%
# This function analyze the trypic pattern of the peptides in pept dataframe
def get_tryptic_types(pept_df, prot_seq, peptide_col, clean_pept_col = "clean_pept"):
    seq_len = len(prot_seq)
    # pept_df.reset_index(drop=True, inplace=True)
    if pept_df.shape[0] == 0:
        print("The peptide dataframe is empty. Please check the input dataframe.")
        return
    else:
        if clean_pept_col not in pept_df.columns:
            pept_df = get_clean_peptides(pept_df, peptide_col, clean_pept_col)
        pept_start = [prot_seq.find(clean_pept) + 1 for clean_pept in pept_df[clean_pept_col]]
        pept_end = [prot_seq.find(clean_pept) + len(clean_pept) for clean_pept in pept_df[clean_pept_col]]
        pept_df["pept_start"] = pept_start
        pept_df["pept_end"] = pept_end
        pept_df["pept_type"] = ["Not-matched" if i == 0 else "Tryptic" if ((prot_seq[i-2] in "KR" or i == 1) and (prot_seq[j-1] in "KR" or j == seq_len)) else "Semi-tryptic" if (prot_seq[i-2] in "KR" or prot_seq[j-1] in "KR") else "Non-tryptic" for i,j in zip(pept_start, pept_end)]
    return pept_df


# %%
# Select peptides based on the digestion pattern, depending on the type of peptides (all, any-tryptic, tryptic, semi-tryptic, or non-tryptic)
def select_tryptic_pattern(pept_df, prot_seq, tryptic_pattern="all", peptide_col = None, clean_pept_col = "clean_pept"):
    """_summary_

    Args:
        pept_df (_type_): _description_
        prot_seq (_type_): _description_
        pept_type (str, optional): _description_. Defaults to "all".
        peptide_col (str, optional): _description_. Defaults to "Sequence".

    Raises:
        ValueError: _description_
    """

    if "pept_type" not in pept_df.columns:
        pept_df = get_tryptic_types(pept_df, prot_seq, peptide_col, clean_pept_col)

    if tryptic_pattern.lower() == "all":
        protein = pept_df[pept_df["pept_type"] != "Not-matched"].copy()
    elif tryptic_pattern.lower() == "any-tryptic":
        protein = pept_df[(pept_df["pept_type"] != "Non-tryptic") & (pept_df["pept_type"] != "Not-matched")].copy()
    elif tryptic_pattern.lower() == "tryptic":
        protein = pept_df[pept_df["pept_type"] == "Tryptic"].copy()
    elif tryptic_pattern.lower() == "semi-tryptic":
        protein = pept_df[pept_df["pept_type"] == "Semi-tryptic"].copy()
    elif tryptic_pattern.lower() == "non-tryptic":
        protein = pept_df[pept_df["pept_type"] == "Non-tryptic"].copy()
    else:
        raise ValueError("The peptide type is not recognized. Please choose from the following: all, any-tryptic, tryptic, semi-tryptic, non-tryptic")
    return protein


# %%
# #####!!!!!!!!! NEED to work on it!!!!!!!!!!!!!!#####
# This function is to get the peptides in LiP pept dataframe
def get_df_for_pept_alignment_plot(pept_df, prot_seq, pairwise_ttest_name, tryptic_pattern="all", peptide_col=None, clean_pept_col="clean_pept", max_vis_fc=3, id_separator='@',):
    """_summary_

    Args:
        pept_df (_type_): _description_
        prot_seq (_type_): _description_
        pept_type (str, optional): _description_. Defaults to "all".
        peptide_col (str, optional): _description_. Defaults to "Sequence".
        max_vis_fc (int, optional): _description_. Defaults to 3.
    """
    seq_len = len(prot_seq)
    protein = select_tryptic_pattern(pept_df, prot_seq, tryptic_pattern=tryptic_pattern, peptide_col=peptide_col, clean_pept_col=clean_pept_col)
    if protein.shape[0] <= 0:
        print(f"The {tryptic_pattern} peptide dataframe is empty. Please check the input dataframe.")
        return None
    else:
        # protein.reset_index(drop=True, inplace=True)
        protein["pept_id"] = [str(protein["pept_start"].to_list()[i]).zfill(4) + '-' + str(protein["pept_end"].to_list()[i]).zfill(4) + id_separator + pept for i, pept in enumerate(protein[peptide_col].to_list())]
        # protein.index = protein["pept_id"]
        ceiled_fc = [max_vis_fc if i > max_vis_fc else -max_vis_fc if i < -max_vis_fc else i for i in protein[pairwise_ttest_name].to_list()]
        foldchanges = np.zeros((protein.shape[0], seq_len))
        for i in range(len(foldchanges)):
            foldchanges[i, (protein["pept_start"].to_list()[i] - 1):(protein["pept_end"].to_list()[i] - 1)] = ceiled_fc[i]
        fc_df = pd.DataFrame(foldchanges, index=protein["pept_id"], columns=[aa + str(i + 1) for i, aa in enumerate(list(prot_seq))]).sort_index().replace({0:np.nan})
        return fc_df


# %%
# Plot the peptide alignment with the fold changes
def plot_pept_alignment(pept_df, prot_seq, pairwise_ttest_name, save2file = None, tryptic_pattern="all", peptide_col=None, clean_pept_col="clean_pept", max_vis_fc=3, color_map="coolwarm"):
    """_summary_

    Args:
        pept_df (_type_): _description_
        prot_seq (_type_): _description_
        save2file (_type_, optional): _description_. Defaults to None.
        pept_type (str, optional): _description_. Defaults to "all".
        peptide_col (str, optional): _description_. Defaults to "Sequence".
        color_map (str, optional): _description_. Defaults to "coolwarm".
        max_vis_fc (int, optional): _description_. Defaults to 3.
    """
    seq_len = len(prot_seq)
    fc_df = get_df_for_pept_alignment_plot(pept_df, prot_seq, pairwise_ttest_name, tryptic_pattern=tryptic_pattern, peptide_col=peptide_col, clean_pept_col=clean_pept_col, max_vis_fc = max_vis_fc)
    if fc_df is not None:
        plt.figure(figsize=(min(max(math.floor(seq_len/3), 5), 10), min(max(math.floor(pept_df.shape[0]/5), 3),6)))
        sns.heatmap(fc_df,center=0, cmap=color_map)
        plt.tight_layout()
        if save2file is not None:
            plt.savefig(f"{save2file}_{tryptic_pattern}_pept_alignments_with_FC.pdf")
            plt.close()
            return None
        else:
            return plt
    else:
        print(f"The {tryptic_pattern} peptide dataframe is empty. Please check the input dataframe.")
        return None


# %%
# This function is to analyze the trypic pattern of the peptides in LiP pept dataframe
def analyze_tryptic_pattern(protein, sequence, pairwise_ttest_groups, groups, description="", peptide_col=None, clean_pept_col="clean_pept", anova_type="[Group]", keep_non_tryptic=True, id_separator="@", sig_type="pval", sig_thr=0.05):
    """_summary_

    Args:
        protein (_type_): _description_
        sequence (_type_): _description_
        description (str, optional): _description_. Defaults to "".
    """
    # protein.reset_index(drop=True, inplace=True)
    seq_len = len(sequence)
    protein["Protein description"] = description
    protein["Protein length"] = seq_len
    protein = get_tryptic_types(protein, sequence, peptide_col, clean_pept_col)
    protein["Tryp Pept num"] = protein[(protein["pept_type"] == "Tryptic")].copy().shape[0]
    protein["Semi Pept num"] = protein[(protein["pept_type"] == "Semi-tryptic")].copy().shape[0]

    if len(groups) > 2:
        sig_semi_pepts = protein[(protein["pept_type"] == "Semi-tryptic") & (protein[f"ANOVA_{anova_type}_{sig_type}"] < sig_thr)].copy()
        protein[f"ANOVA_{anova_type} Sig Semi Pept num"] = sig_semi_pepts.shape[0]

    pairwise_ttest_names = [pairwise_ttest_group[0] for pairwise_ttest_group in pairwise_ttest_groups]
    if sig_semi_pepts.shape[0] != 0:
        protein[f"Max absFC of All ANOVA_{anova_type} Sig Semi Pept"] = np.nanmax(sig_semi_pepts[pairwise_ttest_names].abs().values)
        protein[f"Sum absFC of All ANOVA_{anova_type} Sig Semi Pept"] = np.nansum(sig_semi_pepts[pairwise_ttest_names].abs().values)
        protein[f"Median absFC of All ANOVA_{anova_type} Sig Semi Pept"] = np.nanmedian(sig_semi_pepts[pairwise_ttest_names].abs().values)
    else:
        protein[f"Max absFC of All ANOVA_{anova_type} Sig Semi Pept"] = np.nan
        protein[f"Sum absFC of All ANOVA_{anova_type} Sig Semi Pept"] = np.nan
        protein[f"Median absFC of All ANOVA_{anova_type} Sig Semi Pept"] = np.nan

    for pairwise_ttest_name in pairwise_ttest_names:
        sig_semi_pepts = protein[(protein["pept_type"] == "Semi-tryptic") & (protein[f"{pairwise_ttest_name}_{sig_type}"] < sig_thr)].copy()
        protein[f"{pairwise_ttest_name} Sig Semi Pept num"] = sig_semi_pepts.shape[0]
        if sig_semi_pepts.shape[0] != 0:
            protein[f"Max absFC of All {pairwise_ttest_name} Sig Semi Pept"] = np.nanmax(sig_semi_pepts[pairwise_ttest_name].abs().values)
            protein[f"Sum absFC of All {pairwise_ttest_name} Sig Semi Pept"] = np.nansum(sig_semi_pepts[pairwise_ttest_name].abs().values)
            protein[f"Median absFC of All {pairwise_ttest_name} Sig Semi Pept"] = np.nanmedian(sig_semi_pepts[pairwise_ttest_name].abs().values)
        else:
            protein[f"Max absFC of All {pairwise_ttest_name} Sig Semi Pept"] = np.nan
            protein[f"Sum absFC of All {pairwise_ttest_name} Sig Semi Pept"] = np.nan
            protein[f"Median absFC of All {pairwise_ttest_name} Sig Semi Pept"] = np.nan

    protein["pept_id"] = [str(protein["pept_start"].to_list()[i]).zfill(4) + '-' + str(protein["pept_end"].to_list()[i]).zfill(4) + id_separator + pept for i, pept in enumerate(protein[peptide_col].to_list())]
    # protein.index = protein["pept_id"]
    not_matched = protein[protein["pept_type"] == "Not-matched"].copy().sort_index()
    not_matched["lytic_group"] = 0
    tryptic = protein[protein["pept_type"] == "Tryptic"].copy().sort_index()
    tryptic["lytic_group"] = 0
    semitryptic = protein[protein["pept_type"] == "Semi-tryptic"].copy().sort_index()
    semitryptic["lytic_group"] = 0
    if keep_non_tryptic:
        nontryptic = protein[protein["pept_type"] == "Non-tryptic"].copy().sort_index()
        nontryptic["lytic_group"] = 0
    for i, idx in enumerate(tryptic.index.to_list()):
        tryptic.loc[idx, "lytic_group"] = i+1
        semitryptic.loc[((semitryptic["pept_start"] == tryptic.loc[idx, "pept_start"]) | (semitryptic["pept_end"] == tryptic.loc[idx, "pept_end"])), "lytic_group"] = i+1
        if keep_non_tryptic:
            nontryptic.loc[((nontryptic["pept_start"].astype(int) > int(tryptic.loc[idx, "pept_start"])) & (nontryptic["pept_start"].astype(int) < int(tryptic.loc[idx, "pept_end"]))), "lytic_group"] = i+1
    if keep_non_tryptic:
        protein_any_tryptic = pd.concat([not_matched, tryptic, semitryptic, nontryptic]).copy()
    else:
        protein_any_tryptic = pd.concat([not_matched, tryptic, semitryptic]).copy()

    return protein_any_tryptic


# %%
# Rollup to site level, NB: this is for individual proteins, because the protein sequence is needed
# This function is to roll up the LiP pept data to the site level with median values
def LiP_rollup_to_site(pept, int_cols, sequence, uniprot_col, residue_col="Residue", uniprot_id="Protein ID (provided by user)", peptide_col="Sequence", clean_pept_col="clean_pept", id_col="id", id_separator="@", pept_type_col="pept_type", site_col="Site", pos_col="Pos", multiply_rollup_counts=True, ignore_NA=True, rollup_func="median"):
    """_summary_

    Args:
        pept (_type_): _description_
        sequence (_type_): _description_
        uniprot_id (str, optional): _description_. Defaults to "".

    Raises:
        ValueError: _description_
    """
    # seq_len = len(sequence)
    if clean_pept_col not in pept.columns.to_list():
        pept = get_tryptic_types(pept, sequence, peptide_col, clean_pept_col)
    if pept.shape[0] > 0:
        pept = get_clean_peptides(pept, peptide_col, clean_pept_col)
        pept[residue_col] = [[res + str(sequence.find(clean_pept)+i+1) for i, res in enumerate(clean_pept)] for clean_pept in pept[clean_pept_col]]
        info_cols = [col for col in pept.columns if col not in int_cols]
        pept = pept.explode(residue_col)
        pept[id_col] = uniprot_id + id_separator + pept[residue_col] + id_separator + pept[pept_type_col]
        # pept[id_col] = uniprot_id + id_separator + pept[residue_col]
        # pept[int_cols] = 2 ** (pept[int_cols])
        # pept_grouped = pept[int_cols].groupby(pept.index).sum(min_count=1)
        # pept_grouped = log2_transformation(pept_grouped)
        # # Lisa Bramer and Kelly Straton suggested to use median of log2 scale values rathen than summing up the intenisty values at linear scale
        info_cols_wo_peptide_col = [col for col in info_cols if col != peptide_col]
        agg_methods_0 = {peptide_col: lambda x: '; '.join(x)}
        agg_methods_1 = {i: lambda x: x.iloc[0] for i in info_cols_wo_peptide_col}
        if multiply_rollup_counts:
            if ignore_NA:
                if rollup_func.lower() == "median":
                    agg_methods_2 = {i: lambda x: np.log2(len(x)) + x.median() for i in int_cols}
                elif rollup_func.lower() == "mean":
                    agg_methods_2 = {i: lambda x: np.log2(len(x)) + x.mean() for i in int_cols}
                elif rollup_func.lower() == "sum":
                    agg_methods_2 = {i: lambda x: np.log2(np.nansum(2**(x.replace(0, np.nan)))) for i in int_cols}
                else:
                    ValueError("The rollup function is not recognized. Please choose from the following: median, mean, sum")
            else:
                if rollup_func.lower() == "median":
                    agg_methods_2 = {i: lambda x: np.log2(x.notna().sum()) + x.median() for i in int_cols}
                elif rollup_func.lower() == "mean":
                    agg_methods_2 = {i: lambda x: np.log2(x.notna().sum()) + x.mean() for i in int_cols}
                elif rollup_func.lower() == "sum":
                    agg_methods_2 = {i: lambda x: np.log2(np.nansum(2**(x.replace(0, np.nan)))) for i in int_cols}
                else:
                    ValueError("The rollup function is not recognized. Please choose from the following: median, mean, sum")
        else:
            if rollup_func.lower() == "median":
                agg_methods_2 = {i: lambda x: x.median() for i in int_cols}
            elif rollup_func.lower() == "mean":
                agg_methods_2 = {i: lambda x: x.mean() for i in int_cols}
            elif rollup_func.lower() == "sum":
                agg_methods_2 = {i: lambda x: np.log2(np.nansum(2**(x.replace(0, np.nan)))) for i in int_cols}
            else:
                ValueError("The rollup function is not recognized. Please choose from the following: median, mean, sum")
        pept_grouped = pept.groupby(id_col, as_index=False).agg({**agg_methods_0, **agg_methods_1, **agg_methods_2})
        pept_grouped[uniprot_col] = uniprot_id
        pept_grouped[site_col] = [site.split(id_separator)[1] for site in pept_grouped[id_col]]
        pept_grouped[pos_col] = [int(re.sub(r"\D", "", site)) for site in pept_grouped[site_col]]
        pept_grouped.sort_values(by=[pos_col], inplace=True)
        pept_grouped[pept_type_col] = [site.split(id_separator)[-1] for site in pept_grouped[id_col]]
        # pept_grouped.index = uniprot_id + id_separator + pept_grouped["Site"]
        pept_grouped.index = pept_grouped[id_col].to_list()
        return pept_grouped
    else:
        raise ValueError("The pept dataframe is empty. Please check the input dataframe.")


# %%
# This function is to plot the barcode of a protein with fold changes at single site level
def plot_pept_barcode(pept_df, pairwise_ttest_name, sequence, save2file=None, uniprot_id="Protein ID (provided by user)", max_vis_fc=3, color_levels=20, sig_type="pval", sig_thr=0.05,):
    """_summary_

    Args:
        pept_df (_type_): _description_
        sequence (_type_): _description_
        save2file (_type_, optional): _description_. Defaults to None.
        max_vis_fc (int, optional): _description_. Defaults to 3.
        color_levels (int, optional): _description_. Defaults to 20.
    """
    seq_len = len(sequence)
    tryptic = pept_df[pept_df["pept_type"] == "Tryptic"].copy()
    semi = pept_df[pept_df["pept_type"] == "Semi-tryptic"].copy()
    if semi.shape[0] > 0 or tryptic.shape[0] > 0:
        # both_pal_vals = sns.color_palette("Greens", color_levels)
        up_pal_vals = sns.color_palette("Reds", color_levels)
        down_pal_vals = sns.color_palette("Blues", color_levels)
        insig_pal_vals = sns.color_palette("Greys", color_levels)

        fc_diff_names = [aa + str(i+1) for i, aa in enumerate(list(sequence))]
        fc_diff_max = pept_df[pairwise_ttest_name].abs().max()
        tryptic_bar_code = [(1.0, 1.0, 1.0)] * len(fc_diff_names)
        semi_bar_code = [(1.0, 1.0, 1.0)] * len(fc_diff_names)
        if tryptic.shape[0] > 0:
            tryptic_fc_diff = tryptic[["Site", "Pos", pairwise_ttest_name, f"{pairwise_ttest_name}_{sig_type}"]].copy()
            tryptic_fc_diff.index = tryptic_fc_diff["Site"].to_list()
            for i in range(tryptic_fc_diff.shape[0]):
                if tryptic_fc_diff.iloc[i, 2] > 0:
                    if tryptic_fc_diff.iloc[i, 3] < sig_thr:
                        tryptic_bar_code[tryptic_fc_diff.iloc[i, 1] - 1] = up_pal_vals[np.ceil(min(abs(tryptic_fc_diff.iloc[i, 2]),max_vis_fc + 0.1)/fc_diff_max * color_levels).astype("int") - 1]
                    else:
                        tryptic_bar_code[tryptic_fc_diff.iloc[i, 1] - 1] = insig_pal_vals[np.ceil(min(abs(tryptic_fc_diff.iloc[i, 2]),max_vis_fc + 0.1)/fc_diff_max * color_levels).astype("int") - 1]
                else:
                    if tryptic_fc_diff.iloc[i, 3] < sig_thr:
                        tryptic_bar_code[tryptic_fc_diff.iloc[i, 1] - 1] = down_pal_vals[np.ceil(min(abs(tryptic_fc_diff.iloc[i, 2]),max_vis_fc + 0.1)/fc_diff_max * color_levels).astype("int") - 1]
                    else:
                        tryptic_bar_code[tryptic_fc_diff.iloc[i, 1] - 1] = insig_pal_vals[np.ceil(min(abs(tryptic_fc_diff.iloc[i, 2]),max_vis_fc + 0.1)/fc_diff_max * color_levels).astype("int") - 1]
        if semi.shape[0] > 0:
            semi_fc_diff = semi[["Site", "Pos", pairwise_ttest_name, f"{pairwise_ttest_name}_{sig_type}"]].copy()
            semi_fc_diff.index = semi_fc_diff["Site"].to_list()
            for i in range(semi_fc_diff.shape[0]):
                if semi_fc_diff.iloc[i, 2] > 0:
                    if semi_fc_diff.iloc[i, 3] < sig_thr:
                        semi_bar_code[semi_fc_diff.iloc[i, 1] - 1] = up_pal_vals[np.ceil(min(abs(semi_fc_diff.iloc[i, 2]),max_vis_fc + 0.1)/fc_diff_max * color_levels).astype("int") - 1]
                    else:
                        semi_bar_code[semi_fc_diff.iloc[i, 1] - 1] = insig_pal_vals[np.ceil(min(abs(semi_fc_diff.iloc[i, 2]),max_vis_fc + 0.1)/fc_diff_max * color_levels).astype("int") - 1]
                else:
                    if semi_fc_diff.iloc[i, 3] < sig_thr:
                        semi_bar_code[semi_fc_diff.iloc[i, 1] - 1] = down_pal_vals[np.ceil(min(abs(semi_fc_diff.iloc[i, 2]),max_vis_fc + 0.1)/fc_diff_max * color_levels).astype("int") - 1]
                    else:
                        semi_bar_code[semi_fc_diff.iloc[i, 1] - 1] = insig_pal_vals[np.ceil(min(abs(semi_fc_diff.iloc[i, 2]),max_vis_fc + 0.1)/fc_diff_max * color_levels).astype("int") - 1]

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(2, 1, 1)
        plot_barcode(tryptic_bar_code, barcode_name=uniprot_id + "_tryptic", ticklabel=[fc_diff_names[j] for j in np.arange(0,seq_len,np.ceil(seq_len/10).astype("int"))], ax=ax)
        ax = fig.add_subplot(2, 1, 2)
        plot_barcode(semi_bar_code, barcode_name=uniprot_id + "_semi-tryptic", ticklabel=[fc_diff_names[j] for j in np.arange(0,seq_len,np.ceil(seq_len/10).astype("int"))], ax=ax)
        plt.tight_layout()
        if save2file is not None:
            plt.savefig(f"{save2file}_{uniprot_id}_any_tryptic_barcodes.pdf")
            plt.close()
            return None
        else:
            return plt
    else:
        print("The peptide dataframe is empty with either tryptic or semi-tryptic peptides. Please check the input dataframe.")
        return None


# %%
# This function is to analyze the digestion site pattern of the peptides in LiP pept dataframe
def LiP_rollup_to_lytic_site(df, int_cols, uniprot_col, sequence, residue_col="Residue", description="", tryptic_pattern="all", peptide_col="Sequence", clean_pept_col="clean_pept", id_separator="@", id_col="id", pept_type_col="pept_type", site_col="Site", pos_col="Pos", multiply_rollup_counts=True, ignore_NA=True, alternative_protease="ProK", rollup_func="median"):
    """_summary_

    Args:
        protein (_type_): _description_
        sequence (_type_): _description_
        description (str, optional): _description_. Defaults to "".
    """
    protein = df.copy()
    # protein.reset_index(drop=True, inplace=True)
    seq_len = len(sequence)
    uniprot_id = protein[uniprot_col].unique()[0]
    clean_pepts = [strip_peptide(pept, nip_off=False) for pept in protein[peptide_col].to_list()]
    protein["Protein description"] = description
    protein["Protein length"] = seq_len
    protein = get_tryptic_types(protein, sequence, peptide_col, clean_pept_col)

    lyticsites = []
    for clean_pept in clean_pepts:
        start_lytic_pos = sequence.find(clean_pept)
        end_lytic_pos = start_lytic_pos + len(clean_pept)
        start_lytic_site = sequence[start_lytic_pos - 1]
        end_lytic_site = sequence[end_lytic_pos - 1]
        lyticsites.append([start_lytic_site + str(start_lytic_pos), end_lytic_site + str(end_lytic_pos)])

    protein[residue_col] = lyticsites
    pept_num = len(clean_pepts)
    semi_num = protein[pept_type_col].to_list().count("Semi-tryptic")
    non_num = protein[pept_type_col].to_list().count("Non-tryptic")
    prok_num = protein[pept_type_col].to_list().count("Semi-tryptic") + protein[pept_type_col].to_list().count("Non-tryptic")
    protein2explode = select_tryptic_pattern(protein, sequence, tryptic_pattern=tryptic_pattern, peptide_col=peptide_col, clean_pept_col=clean_pept_col)

    if protein2explode.shape[0] > 0:
        protein_lys = protein2explode.explode(residue_col)
        info_cols = [col for col in protein_lys.columns if col not in int_cols]
        protein_lys[id_col] = uniprot_id + id_separator + protein_lys[residue_col]
        info_cols_wo_peptide_col = [col for col in info_cols if col != peptide_col]
        agg_methods_0 = {peptide_col: lambda x: '; '.join(x)}
        agg_methods_1 = {i: lambda x: x.iloc[0] for i in info_cols_wo_peptide_col}
        if multiply_rollup_counts:
            if ignore_NA:
                if rollup_func.lower() == "median":
                    agg_methods_2 = {i: lambda x: np.log2(len(x)) + x.median() for i in int_cols}
                elif rollup_func.lower() == "mean":
                    agg_methods_2 = {i: lambda x: np.log2(len(x)) + x.mean() for i in int_cols}
                elif rollup_func.lower() == "sum":
                    agg_methods_2 = {i: lambda x: np.log2(np.nansum(2**(x.replace(0, np.nan)))) for i in int_cols}
                else:
                    ValueError("The rollup function is not recognized. Please choose from the following: median, mean, sum")
            else:
                if rollup_func.lower() == "median":
                    agg_methods_2 = {i: lambda x: np.log2(x.notna().sum()) + x.median() for i in int_cols}
                elif rollup_func.lower() == "mean":
                    agg_methods_2 = {i: lambda x: np.log2(x.notna().sum()) + x.mean() for i in int_cols}
                elif rollup_func.lower() == "sum":
                    agg_methods_2 = {i: lambda x: np.log2(np.nansum(2**(x.replace(0, np.nan)))) for i in int_cols}
                else:
                    ValueError("The rollup function is not recognized. Please choose from the following: median, mean, sum")
        else:
            if rollup_func.lower() == "median":
                agg_methods_2 = {i: lambda x: x.median() for i in int_cols}
            elif rollup_func.lower() == "mean":
                agg_methods_2 = {i: lambda x: x.mean() for i in int_cols}
            elif rollup_func.lower() == "sum":
                agg_methods_2 = {i: lambda x: np.log2(np.nansum(2**(x.replace(0, np.nan)))) for i in int_cols}
            else:
                ValueError("The rollup function is not recognized. Please choose from the following: median, mean, sum")
        protein_lys_grouped = protein_lys.groupby(id_col, as_index=False).agg({**agg_methods_0, **agg_methods_1, **agg_methods_2})

        protein_lys_grouped[uniprot_col] = uniprot_id
        protein_lys_grouped[site_col] = [site.split(id_separator)[1] for site in protein_lys_grouped[id_col]]
        protein_lys_grouped[pos_col] = [int(re.sub(r"\D", "", site)) for site in protein_lys_grouped[site_col]]
        protein_lys_grouped["Lytic site type"] = ["trypsin" if ('K' in i) or ('R' in i) else alternative_protease for i in protein_lys_grouped[site_col]]
        protein_lys_grouped["All pept num"] = pept_num
        protein_lys_grouped["Semi-tryptic pept num"] = semi_num
        protein_lys_grouped["Non-tryptic pept num"] = non_num
        protein_lys_grouped[f"{alternative_protease} pept num"] = prok_num
        protein_lys_grouped[f"{alternative_protease} site num"] = protein_lys_grouped["Lytic site type"].to_list().count(alternative_protease)
        protein_lys_grouped["Tryp site num"] = protein_lys_grouped["Lytic site type"].to_list().count("trypsin")
        
        protein_lys_grouped.sort_values(by=[pos_col], inplace=True)
        # protein_lys_grouped_sig.sort_values(by=[pos_col], inplace=True)
        protein_lys_grouped.index = protein_lys_grouped[id_col].to_list()
        return protein_lys_grouped
    else:
        # raise ValueError("The pept dataframe is empty. Please check the input dataframe.")
        print(f"The resulted dataframe of digestion site in {uniprot_id} is empty. Please check the input dataframe.")
        return None


# %%
# Select digestion site type, depending on the type of digestion site (all or both, trypsin, prok)
def select_lytic_sites(site_df, site_type="prok", site_type_col="Lytic site type"):
    """_summary_

    Args:
        site_df (_type_): _description_
        site_type (str, optional): _description_. Defaults to "prok".
        site_type_col (str, optional): _description_. Defaults to "Lytic site type".
    """
    site_df_out = site_df[site_df[site_type_col] == site_type].copy()
    return site_df_out


# %%
# This function is to plot the barcode of a protein with fold changes at lytic site level
def plot_site_barcode(site_df, sequence, pairwise_ttest_name, save2file=None, uniprot_id="Protein ID (provided by user)", max_vis_fc=3, color_levels=20, site_type_col="Lytic site type", sig_type="pval", sig_thr=0.05):
    seq_len = len(sequence)
    trypsin = select_lytic_sites(site_df, "trypsin", site_type_col)
    prok = select_lytic_sites(site_df, "prok", site_type_col)
    if prok.shape[0] > 0 or trypsin.shape[0] > 0:
        # both_pal_vals = sns.color_palette("Greens", color_levels)
        up_pal_vals = sns.color_palette("Reds", color_levels)
        down_pal_vals = sns.color_palette("Blues", color_levels)
        insig_pal_vals = sns.color_palette("Greys", color_levels)

        fc_diff_names = [aa + str(i+1) for i, aa in enumerate(list(sequence))]
        fc_diff_max = site_df[pairwise_ttest_name].abs().max()
        trypsin_bar_code = [(1.0,1.0,1.0)] * len(fc_diff_names)
        prok_bar_code = [(1.0,1.0,1.0)] * len(fc_diff_names)
        if trypsin.shape[0] > 0:
            trypsin_fc_diff = trypsin[["Site", "Pos", pairwise_ttest_name, f"{pairwise_ttest_name}_{sig_type}"]].copy()
            trypsin_fc_diff.index = trypsin_fc_diff["Site"].to_list()
            for i in range(trypsin_fc_diff.shape[0]):
                if trypsin_fc_diff.iloc[i, 2] > 0:
                    if trypsin_fc_diff.iloc[i, 3] < sig_thr:
                        trypsin_bar_code[trypsin_fc_diff.iloc[i, 1] - 1] = up_pal_vals[np.ceil(min(abs(trypsin_fc_diff.iloc[i, 2]),max_vis_fc + 0.1)/fc_diff_max * color_levels).astype("int") - 1]
                    else:
                        trypsin_bar_code[trypsin_fc_diff.iloc[i, 1] - 1] = insig_pal_vals[np.ceil(min(abs(trypsin_fc_diff.iloc[i, 2]),max_vis_fc + 0.1)/fc_diff_max * color_levels).astype("int") - 1]
                else:
                    if trypsin_fc_diff.iloc[i, 3] < sig_thr:
                        trypsin_bar_code[trypsin_fc_diff.iloc[i, 1] - 1] = down_pal_vals[np.ceil(min(abs(trypsin_fc_diff.iloc[i, 2]),max_vis_fc + 0.1)/fc_diff_max * color_levels).astype("int") - 1]
                    else:
                        trypsin_bar_code[trypsin_fc_diff.iloc[i, 1] - 1] = insig_pal_vals[np.ceil(min(abs(trypsin_fc_diff.iloc[i, 2]),max_vis_fc + 0.1)/fc_diff_max * color_levels).astype("int") - 1]
        if prok.shape[0] > 0:
            prok_fc_diff = prok[["Site", "Pos", pairwise_ttest_name, f"{pairwise_ttest_name}_{sig_type}"]].copy()
            prok_fc_diff.index = prok_fc_diff["Site"].to_list()
            for i in range(prok_fc_diff.shape[0]):
                if prok_fc_diff.iloc[i, 2] > 0:
                    if prok_fc_diff.iloc[i, 3] < sig_thr:
                        prok_bar_code[prok_fc_diff.iloc[i, 1] - 1] = up_pal_vals[np.ceil(min(abs(prok_fc_diff.iloc[i, 2]),max_vis_fc + 0.1)/fc_diff_max * color_levels).astype("int") - 1]
                    else:
                        prok_bar_code[prok_fc_diff.iloc[i, 1] - 1] = insig_pal_vals[np.ceil(min(abs(prok_fc_diff.iloc[i, 2]),max_vis_fc + 0.1)/fc_diff_max * color_levels).astype("int") - 1]
                else:
                    if prok_fc_diff.iloc[i, 3] < sig_thr:
                        prok_bar_code[prok_fc_diff.iloc[i, 1] - 1] = down_pal_vals[np.ceil(min(abs(prok_fc_diff.iloc[i, 2]),max_vis_fc + 0.1)/fc_diff_max * color_levels).astype("int") - 1]
                    else:
                        prok_bar_code[prok_fc_diff.iloc[i, 1] - 1] = insig_pal_vals[np.ceil(min(abs(prok_fc_diff.iloc[i, 2]),max_vis_fc + 0.1)/fc_diff_max * color_levels).astype("int") - 1]

        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(2, 1, 1)
        plot_barcode(trypsin_bar_code, barcode_name = uniprot_id + "_trypsin_site", ticklabel=[fc_diff_names[j] for j in np.arange(0,seq_len,np.ceil(seq_len/10).astype("int"))], ax=ax)
        ax = fig.add_subplot(2, 1, 2)
        plot_barcode(prok_bar_code, barcode_name = uniprot_id + "_prok_site", ticklabel=[fc_diff_names[j] for j in np.arange(0,seq_len,np.ceil(seq_len/10).astype("int"))], ax=ax)
        plt.tight_layout()
        if save2file is not None:
            plt.savefig(f"{save2file}_{uniprot_id}_digestion_site_barcodes.pdf")
            plt.close()
            return None
        else:
            return plt
    else:
        print("The digestion site dataframe is empty with either trypsin or prok sites. Please check the input dataframe.")
        return None
