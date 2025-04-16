from __future__ import annotations

import re
from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import numpy as np
import pandas as pd

from proteometer.peptide import strip_peptide

if TYPE_CHECKING:
    from proteometer.stats import TTestGroup

    AggDictFloat = dict[str, Callable[[pd.Series[float]], float]]
    AggDictStr = dict[str, Callable[[pd.Series[str]], str]]
    AggDictAny = dict[str, Callable[[pd.Series[Any]], Any]]


# This part is filtering all the contaminants and reverse hits
def filter_contaminants_reverse_pept(
    df: pd.DataFrame,
    search_tool: Literal["maxquant", "msfragger", "fragpipe"],
    protein_id_col_pept: str,
    uniprot_col: str,
) -> pd.DataFrame:
    """_summary_

    Args:
        df (_type_): _description_
    """
    if search_tool.lower() == "maxquant":
        df = df[
            (df["Reverse"].isna())
            & (df["Potential contaminant"].isna())
            & (~df[protein_id_col_pept].str.contains("(?i)Contaminant"))
            & (~df[protein_id_col_pept].str.contains("(?i)REV__"))
            & (~df[protein_id_col_pept].str.contains("(?i)CON__"))
        ].copy()
        df[uniprot_col] = df[protein_id_col_pept]
    elif search_tool.lower() == "msfragger" or search_tool.lower() == "fragpipe":
        df = df[(~df[protein_id_col_pept].str.contains("(?i)contam_"))].copy()
        df[uniprot_col] = df[protein_id_col_pept]
    else:
        print(
            "The search tool is not specified or not supported yet. "
            "The user should provide the tables that have been filtered "
            "and without the contaminants and reverse hits."
        )

    return df


def filter_contaminants_reverse_prot(
    df: pd.DataFrame,
    search_tool: Literal["maxquant", "msfragger", "fragpipe"],
    protein_id_col_prot: str,
    uniprot_col: str,
) -> pd.DataFrame:
    """_summary_

    Args:
        df (_type_): _description_
    """
    if search_tool.lower() == "maxquant":
        df = df[
            (df["Only identified by site"].isna())
            & (df["Reverse"].isna())
            & (df["Potential contaminant"].isna())
            & (~df[protein_id_col_prot].str.contains("(?i)Contaminant"))
            & (~df[protein_id_col_prot].str.contains("(?i)REV__"))
            & (~df[protein_id_col_prot].str.contains("CON__"))
        ].copy()
        df[uniprot_col] = [
            ids.split(";")[0] for ids in cast("pd.Series[str]", df[protein_id_col_prot])
        ]
    elif search_tool.lower() == "msfragger" or search_tool.lower() == "fragpipe":
        df = df[(~df[protein_id_col_prot].str.contains("(?i)contam_"))].copy()
        df[uniprot_col] = df[protein_id_col_prot]
    else:
        print(
            "The search tool is not specified or not supported yet. "
            "The user should provide the tables that have been filtered "
            "and without the contaminants and reverse hits."
        )

    return df


# Filtering out the protein groups with less than 2 peptides
def filtering_protein_based_on_peptide_number(
    df2filter: pd.DataFrame,
    peptide_counts_col: str,
    search_tool: Literal["maxquant", "msfragger", "fragpipe"],
    min_pept_count: int = 2,
) -> pd.DataFrame:
    """_summary_

    Args:
        df2filter (_type_): _description_
    """
    if search_tool.lower() == "maxquant":
        df2filter["Pept count"] = [
            int(count.split(";")[0])
            for count in cast("pd.Series[str]", df2filter[peptide_counts_col])
        ]
    elif search_tool.lower() == "msfragger" or search_tool.lower() == "fragpipe":
        df2filter["Pept count"] = df2filter[peptide_counts_col]
    else:
        print(
            "The search tool is not specified or not supported yet. "
            "The user should provide the tables that have been filtered "
            "and without the contaminants and reverse hits."
        )
    df2filter = df2filter[df2filter["Pept count"] >= min_pept_count].copy()
    return df2filter


# This function analyze the trypic pattern of the peptides in pept dataframe
def get_clean_peptides(
    pept_df: pd.DataFrame, peptide_col: str, clean_pept_col: str = "clean_pept"
) -> pd.DataFrame:
    """_summary_

    Args:
        pept_df (_type_): _description_
        peptide_col (_type_): _description_

    Returns:
        _type_: _description_
    """
    clean_pepts = [
        strip_peptide(pept, nip_off=False)
        for pept in cast("list[str]", pept_df[peptide_col].to_list())
    ]
    pept_df[clean_pept_col] = clean_pepts
    return pept_df


# This function analyze the trypic pattern of the peptides in pept dataframe
def get_tryptic_types(
    pept_df: pd.DataFrame,
    prot_seq: str,
    peptide_col: str,
    clean_pept_col: str = "clean_pept",
) -> pd.DataFrame:
    seq_len = len(prot_seq)
    # pept_df.reset_index(drop=True, inplace=True)
    if pept_df.shape[0] == 0:
        raise ValueError(
            "The peptide dataframe is empty. Please check the input dataframe."
        )

    else:
        if clean_pept_col not in pept_df.columns:
            pept_df = get_clean_peptides(pept_df, peptide_col, clean_pept_col)
        pept_start = [
            prot_seq.find(clean_pept) + 1
            for clean_pept in cast("pd.Series[str]", pept_df[clean_pept_col])
        ]
        pept_end = [
            prot_seq.find(clean_pept) + len(clean_pept)
            for clean_pept in cast("pd.Series[str]", pept_df[clean_pept_col])
        ]
        pept_df["pept_start"] = pept_start
        pept_df["pept_end"] = pept_end
        pept_df["pept_type"] = [
            "Not-matched"
            if i == 0
            else "Tryptic"
            if (
                (prot_seq[i - 2] in "KR" or i == 1)
                and (prot_seq[j - 1] in "KR" or j == seq_len)
            )
            else "Semi-tryptic"
            if (prot_seq[i - 2] in "KR" or prot_seq[j - 1] in "KR")
            else "Non-tryptic"
            for i, j in zip(pept_start, pept_end)
        ]
    return pept_df


# Select peptides based on the digestion pattern, depending on the type of
# peptides (all, any-tryptic, tryptic, semi-tryptic, or non-tryptic)
def select_tryptic_pattern(
    pept_df: pd.DataFrame,
    prot_seq: str,
    tryptic_pattern: str = "all",
    peptide_col: str = "Sequence",
    clean_pept_col: str = "clean_pept",
) -> pd.DataFrame:
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
        protein = pept_df[
            (pept_df["pept_type"] != "Non-tryptic")
            & (pept_df["pept_type"] != "Not-matched")
        ].copy()
    elif tryptic_pattern.lower() == "tryptic":
        protein = pept_df[pept_df["pept_type"] == "Tryptic"].copy()
    elif tryptic_pattern.lower() == "semi-tryptic":
        protein = pept_df[pept_df["pept_type"] == "Semi-tryptic"].copy()
    elif tryptic_pattern.lower() == "non-tryptic":
        protein = pept_df[pept_df["pept_type"] == "Non-tryptic"].copy()
    else:
        raise ValueError(
            "The peptide type is not recognized. Please choose from the following: "
            "all, any-tryptic, tryptic, semi-tryptic, non-tryptic"
        )
    return protein


def analyze_tryptic_pattern(
    protein: pd.DataFrame,
    sequence: str,
    pairwise_ttest_groups: Iterable[TTestGroup],
    groups: Collection[str],
    peptide_col: str,
    description: str = "",
    clean_pept_col: str = "clean_pept",
    anova_type: str = "[Group]",
    keep_non_tryptic: bool = True,
    id_separator: str = "@",
    sig_type: str = "pval",
    sig_thr: float = 0.05,
) -> pd.DataFrame:
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
    protein["Tryp Pept num"] = (
        protein[(protein["pept_type"] == "Tryptic")].copy().shape[0]
    )
    protein["Semi Pept num"] = (
        protein[(protein["pept_type"] == "Semi-tryptic")].copy().shape[0]
    )

    pairwise_ttest_names = [
        pairwise_ttest_group.label() for pairwise_ttest_group in pairwise_ttest_groups
    ]
    if len(groups) > 2:
        sig_semi_pepts = protein[
            (protein["pept_type"] == "Semi-tryptic")
            & (protein[f"ANOVA_{anova_type}_{sig_type}"] < sig_thr)
        ].copy()
        protein[f"ANOVA_{anova_type} Sig Semi Pept num"] = sig_semi_pepts.shape[0]

        if sig_semi_pepts.shape[0] != 0:
            protein[f"Max absFC of All ANOVA_{anova_type} Sig Semi Pept"] = np.nanmax(
                cast(
                    "pd.Series[float]",
                    sig_semi_pepts[pairwise_ttest_names].abs().values,
                )
            )
            protein[f"Sum absFC of All ANOVA_{anova_type} Sig Semi Pept"] = np.nansum(
                cast(
                    "pd.Series[float]",
                    sig_semi_pepts[pairwise_ttest_names].abs().values,
                )
            )
            protein[f"Median absFC of All ANOVA_{anova_type} Sig Semi Pept"] = (
                np.nanmedian(
                    cast(
                        "pd.Series[float]",
                        sig_semi_pepts[pairwise_ttest_names].abs().values,
                    )
                )
            )
        else:
            protein[f"Max absFC of All ANOVA_{anova_type} Sig Semi Pept"] = np.nan
            protein[f"Sum absFC of All ANOVA_{anova_type} Sig Semi Pept"] = np.nan
            protein[f"Median absFC of All ANOVA_{anova_type} Sig Semi Pept"] = np.nan

    for pairwise_ttest_name in pairwise_ttest_names:
        sig_semi_pepts = protein[
            (protein["pept_type"] == "Semi-tryptic")
            & (protein[f"{pairwise_ttest_name}_{sig_type}"] < sig_thr)
        ].copy()
        protein[f"{pairwise_ttest_name} Sig Semi Pept num"] = sig_semi_pepts.shape[0]
        if sig_semi_pepts.shape[0] != 0:
            protein[f"Max absFC of All {pairwise_ttest_name} Sig Semi Pept"] = (
                np.nanmax(
                    cast(
                        "pd.Series[float]",
                        sig_semi_pepts[pairwise_ttest_name].abs().values,
                    )
                )
            )
            protein[f"Sum absFC of All {pairwise_ttest_name} Sig Semi Pept"] = (
                np.nansum(
                    cast(
                        "pd.Series[float]",
                        sig_semi_pepts[pairwise_ttest_name].abs().values,
                    )
                )
            )
            protein[f"Median absFC of All {pairwise_ttest_name} Sig Semi Pept"] = (
                np.nanmedian(
                    cast(
                        "pd.Series[float]",
                        sig_semi_pepts[pairwise_ttest_name].abs().values,
                    )
                )
            )
        else:
            protein[f"Max absFC of All {pairwise_ttest_name} Sig Semi Pept"] = np.nan
            protein[f"Sum absFC of All {pairwise_ttest_name} Sig Semi Pept"] = np.nan
            protein[f"Median absFC of All {pairwise_ttest_name} Sig Semi Pept"] = np.nan

    protein["pept_id"] = [
        str(cast("int", protein["pept_start"].to_list()[i])).zfill(4)
        + "-"
        + str(cast("int", protein["pept_end"].to_list()[i])).zfill(4)
        + id_separator
        + pept
        for i, pept in enumerate(cast("pd.Series[str]", protein[peptide_col].to_list()))
    ]
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
    else:
        nontryptic = None
    for i, idx in enumerate(cast("list[int]", tryptic.index.to_list())):
        tryptic.loc[idx, "lytic_group"] = i + 1
        semitryptic.loc[
            (
                (semitryptic["pept_start"] == tryptic.loc[idx, "pept_start"])
                | (semitryptic["pept_end"] == tryptic.loc[idx, "pept_end"])
            ),
            "lytic_group",
        ] = i + 1
        if keep_non_tryptic:
            assert nontryptic is not None
            nontryptic.loc[
                (
                    (
                        nontryptic["pept_start"].astype(int)
                        > int((tryptic.loc[idx, "pept_start"]))  # type: ignore
                    )
                    & (
                        nontryptic["pept_start"].astype(int)
                        < int((tryptic.loc[idx, "pept_end"]))  # type: ignore
                    )
                ),
                "lytic_group",
            ] = i + 1
    if keep_non_tryptic:
        protein_any_tryptic = pd.concat(
            [not_matched, tryptic, semitryptic, nontryptic]
        ).copy()
    else:
        protein_any_tryptic = pd.concat([not_matched, tryptic, semitryptic]).copy()

    return protein_any_tryptic


# This function is to analyze the digestion site pattern of the peptides in LiP pept dataframe
def rollup_to_lytic_site(
    df: pd.DataFrame,
    int_cols: Iterable[str],
    uniprot_col: str,
    sequence: str,
    residue_col: str = "Residue",
    description: str = "",
    tryptic_pattern: str = "all",
    peptide_col: str = "Sequence",
    clean_pept_col: str = "clean_pept",
    id_separator: str = "@",
    id_col: str = "id",
    pept_type_col: str = "pept_type",
    site_col: str = "Site",
    pos_col: str = "Pos",
    multiply_rollup_counts: bool = True,
    ignore_NA: bool = True,
    alternative_protease: str = "ProK",
    rollup_func: Literal["median", "mean", "sum"] = "median",
) -> pd.DataFrame:
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
    clean_pepts = [
        strip_peptide(pept, nip_off=False)
        for pept in cast("pd.Series[str]", protein[peptide_col].to_list())
    ]
    protein["Protein description"] = description
    protein["Protein length"] = seq_len
    protein = get_tryptic_types(protein, sequence, peptide_col, clean_pept_col)

    lyticsites = []
    for clean_pept in clean_pepts:
        start_lytic_pos = sequence.find(clean_pept)
        end_lytic_pos = start_lytic_pos + len(clean_pept)
        start_lytic_site = sequence[start_lytic_pos - 1]
        end_lytic_site = sequence[end_lytic_pos - 1]
        lyticsites.append(
            [
                start_lytic_site + str(start_lytic_pos),
                end_lytic_site + str(end_lytic_pos),
            ]
        )

    protein[residue_col] = lyticsites
    pept_num = len(clean_pepts)
    semi_num = protein[pept_type_col].to_list().count("Semi-tryptic")
    non_num = protein[pept_type_col].to_list().count("Non-tryptic")
    prok_num = protein[pept_type_col].to_list().count("Semi-tryptic") + protein[
        pept_type_col
    ].to_list().count("Non-tryptic")
    protein2explode = select_tryptic_pattern(
        protein,
        sequence,
        tryptic_pattern=tryptic_pattern,
        peptide_col=peptide_col,
        clean_pept_col=clean_pept_col,
    )

    if protein2explode.shape[0] == 0:
        raise ValueError(
            f"The resulted dataframe of digestion site in {uniprot_id} is empty. "
            "Please check the input dataframe."
        )

    protein_lys = protein2explode.explode(residue_col)
    info_cols = [col for col in protein_lys.columns if col not in int_cols]
    protein_lys[id_col] = uniprot_id + id_separator + protein_lys[residue_col]
    info_cols_wo_peptide_col = [col for col in info_cols if col != peptide_col]
    agg_methods_0: AggDictStr = {peptide_col: lambda x: "; ".join(x)}
    agg_methods_1: AggDictAny = {
        i: lambda x: x.iloc[0] for i in info_cols_wo_peptide_col
    }
    if multiply_rollup_counts:
        if ignore_NA:
            if rollup_func.lower() == "median":
                agg_methods_2: AggDictFloat = {
                    i: lambda x: np.log2(len(x)) + x.median() for i in int_cols
                }
            elif rollup_func.lower() == "mean":
                agg_methods_2: AggDictFloat = {
                    i: lambda x: np.log2(len(x)) + x.mean() for i in int_cols
                }
            elif rollup_func.lower() == "sum":
                agg_methods_2: AggDictFloat = {
                    i: lambda x: np.log2(np.nansum(2 ** (x.replace(0, np.nan))))
                    for i in int_cols
                }
            else:
                raise ValueError(
                    "The rollup function is not recognized. Please choose from the following: "
                    "median, mean, sum"
                )
        else:
            if rollup_func.lower() == "median":
                agg_methods_2: AggDictFloat = {
                    i: lambda x: np.log2(x.notna().sum()) + x.median() for i in int_cols
                }
            elif rollup_func.lower() == "mean":
                agg_methods_2: AggDictFloat = {
                    i: lambda x: np.log2(x.notna().sum()) + x.mean() for i in int_cols
                }
            elif rollup_func.lower() == "sum":
                agg_methods_2: AggDictFloat = {
                    i: lambda x: np.log2(np.nansum(2 ** (x.replace(0, np.nan))))
                    for i in int_cols
                }
            else:
                raise ValueError(
                    "The rollup function is not recognized. Please choose from the following: "
                    "median, mean, sum"
                )
    else:
        if rollup_func.lower() == "median":
            agg_methods_2: AggDictFloat = {i: lambda x: x.median() for i in int_cols}
        elif rollup_func.lower() == "mean":
            agg_methods_2: AggDictFloat = {i: lambda x: x.mean() for i in int_cols}
        elif rollup_func.lower() == "sum":
            agg_methods_2: AggDictFloat = {
                i: lambda x: np.log2(np.nansum(2 ** (x.replace(0, np.nan))))
                for i in int_cols
            }
        else:
            raise ValueError(
                "The rollup function is not recognized. Please choose from the following: "
                "median, mean, sum"
            )
    protein_lys_grouped = protein_lys.groupby(id_col, as_index=False).agg(
        {**agg_methods_0, **agg_methods_1, **agg_methods_2}
    )

    protein_lys_grouped[uniprot_col] = uniprot_id
    protein_lys_grouped[site_col] = [
        site.split(id_separator)[1]
        for site in cast("pd.Series[str]", protein_lys_grouped[id_col])
    ]
    protein_lys_grouped[pos_col] = [
        int(re.sub(r"\D", "", site))
        for site in cast("pd.Series[str]", protein_lys_grouped[site_col])
    ]
    protein_lys_grouped["Lytic site type"] = [
        "trypsin" if ("K" in i) or ("R" in i) else alternative_protease
        for i in cast("pd.Series[str]", protein_lys_grouped[site_col])
    ]
    protein_lys_grouped["All pept num"] = pept_num
    protein_lys_grouped["Semi-tryptic pept num"] = semi_num
    protein_lys_grouped["Non-tryptic pept num"] = non_num
    protein_lys_grouped[f"{alternative_protease} pept num"] = prok_num
    protein_lys_grouped[f"{alternative_protease} site num"] = (
        protein_lys_grouped["Lytic site type"].to_list().count(alternative_protease)
    )
    protein_lys_grouped["Tryp site num"] = (
        protein_lys_grouped["Lytic site type"].to_list().count("trypsin")
    )

    protein_lys_grouped.sort_values(by=[pos_col], inplace=True)
    protein_lys_grouped.index = protein_lys_grouped[id_col].to_list()  # type: ignore
    return protein_lys_grouped


def select_lytic_sites(
    site_df: pd.DataFrame,
    site_type: str = "prok",
    site_type_col: str = "Lytic site type",
) -> pd.DataFrame:
    """_summary_

    Args:
        site_df (_type_): _description_
        site_type (str, optional): _description_. Defaults to "prok".
        site_type_col (str, optional): _description_. Defaults to "Lytic site type".
    """
    site_df_out = site_df[site_df[site_type_col] == site_type].copy()
    return site_df_out
