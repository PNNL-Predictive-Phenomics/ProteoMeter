from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import numpy as np
import pandas as pd

from proteometer import fasta
from proteometer.params import Params
from proteometer.peptide import strip_peptide
from proteometer.utils import expsum

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
    """
    Filters out contaminants and reverse hits from a peptide DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing peptide data.
        search_tool (Literal["maxquant", "msfragger", "fragpipe"]): The search tool used for data generation.
        protein_id_col_pept (str): Column name containing protein IDs in the peptide DataFrame.
        uniprot_col (str): Column name to store UniProt IDs.

    Returns:
        pd.DataFrame: Filtered DataFrame with contaminants and reverse hits removed.
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
    """
    Filters out contaminants and reverse hits from a protein DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing protein data.
        search_tool (Literal["maxquant", "msfragger", "fragpipe"]): The search tool used for data generation.
        protein_id_col_prot (str): Column name containing protein IDs in the protein DataFrame.
        uniprot_col (str): Column name to store UniProt IDs.

    Returns:
        pd.DataFrame: Filtered DataFrame with contaminants and reverse hits removed.
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
    """
    Filters proteins based on the minimum number of peptides.

    Args:
        df2filter (pd.DataFrame): Input DataFrame containing proteomics data.
        peptide_counts_col (str): Column name containing peptide counts.
        search_tool (Literal["maxquant", "msfragger", "fragpipe"]): The search tool used for data generation.
        min_pept_count (int, optional): Minimum number of peptides required. Defaults to 2.

    Returns:
        pd.DataFrame: Filtered DataFrame with proteins having at least `min_pept_count` peptides.
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
    """
    Cleans peptide sequences by removing modifications and returns a DataFrame with cleaned peptides.

    Args:
        pept_df (pd.DataFrame): Input DataFrame containing peptide data.
        peptide_col (str): Column name containing peptide sequences.
        clean_pept_col (str, optional): Column name to store cleaned peptide sequences. Defaults to "clean_pept".

    Returns:
        pd.DataFrame: DataFrame with an additional column for cleaned peptide sequences.
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
    """
    Analyzes the tryptic pattern of peptides and classifies them as tryptic, semi-tryptic, or non-tryptic.

    Args:
        pept_df (pd.DataFrame): Input DataFrame containing peptide data.
        prot_seq (str): Protein sequence to analyze against.
        peptide_col (str): Column name containing peptide sequences.
        clean_pept_col (str, optional): Column name for cleaned peptide sequences. Defaults to "clean_pept".

    Returns:
        pd.DataFrame: DataFrame with additional columns for peptide start, end, and type.
    """
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
    """
    Selects peptides based on their digestion pattern.

    Args:
        pept_df (pd.DataFrame): Input DataFrame containing peptide data.
        prot_seq (str): Protein sequence to analyze against.
        tryptic_pattern (str, optional): Digestion pattern to filter peptides. Defaults to "all".
            must be one of: all, any-tryptic, tryptic, semi-tryptic, non-tryptic.
        peptide_col (str, optional): Column name containing peptide sequences. Defaults to "Sequence".
        clean_pept_col (str, optional): Column name for cleaned peptide sequences. Defaults to "clean_pept".

    Returns:
        pd.DataFrame: Filtered DataFrame with peptides matching the specified digestion pattern.
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
    peptide_col: str,
    description: str = "",
    anova_type: str = "[Group]",
    keep_non_tryptic: bool = True,
    id_separator: str = "@",
    sig_type: str = "pval",
    sig_thr: float = 0.05,
) -> pd.DataFrame:
    """
    Analyzes tryptic patterns and calculates statistics for peptides.

    Args:
        protein (pd.DataFrame): Input DataFrame containing proteomics data.
        sequence (str): Protein sequence to analyze against.
        pairwise_ttest_groups (Iterable[TTestGroup]): Groups for pairwise t-tests.
        peptide_col (str): Column name containing peptide sequences.
        description (str, optional): Protein description to add to data frame. Defaults to "".
        anova_type (str, optional): Type of ANOVA analysis. Defaults to "[Group]".
        keep_non_tryptic (bool, optional): Whether to keep non-tryptic peptides. Defaults to True.
        id_separator (str, optional): Separator for peptide IDs. Defaults to "@".
        sig_type (str, optional): Significance type (e.g., "pval"). Defaults to "pval".
        sig_thr (float, optional): Significance threshold. Defaults to 0.05.

    Returns:
        pd.DataFrame: DataFrame with analyzed tryptic patterns and statistics.
    """
    # protein.reset_index(drop=True, inplace=True)
    seq_len = len(sequence)
    protein["Protein description"] = description
    protein["Protein length"] = seq_len
    protein["Tryp Pept num"] = (
        protein[(protein["pept_type"] == "Tryptic")].copy().shape[0]
    )
    protein["Semi Pept num"] = (
        protein[(protein["pept_type"] == "Semi-tryptic")].copy().shape[0]
    )

    pairwise_ttest_names = [
        pairwise_ttest_group.label() for pairwise_ttest_group in pairwise_ttest_groups
    ]
    if len(list(pairwise_ttest_groups)) > 2:
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


def rollup_to_lytic_site(
    double_pept: pd.DataFrame,
    prot_seqs: list[fasta.SeqRecord],
    int_cols: Iterable[str],
    par: Params,
) -> pd.DataFrame:
    """
    Converts the double-peptide data frame to a site-level data frame.

    Args:
        double_pept (pd.DataFrame): The double-peptide data frame.
        prot_seqs (list[fasta.SeqRecord]): The list of protein sequences.
        int_cols (Iterable[str]): The names of columns to with intensity values.
        anova_cols (list[str]): The columns for ANOVA.
        pairwise_ttest_groups (Iterable[stats.TTestGroup]): The pairwise T-test groups.
        metadata (pd.DataFrame): The metadata data frame.
        par (Params): The parameters for limitied proteolysis analysis.

    Returns:
        pd.DataFrame: A data frame with the site-level data.
    """
    double_site: list[pd.DataFrame] = []
    for uniprot_id in double_pept[par.uniprot_col].unique():
        pept_df = cast(
            "pd.DataFrame",
            double_pept[double_pept[par.uniprot_col] == uniprot_id].copy(),
        )
        uniprot_seqs = [prot_seq for prot_seq in prot_seqs if uniprot_id in prot_seq.id]
        if not uniprot_seqs:
            Warning(
                f"Protein {uniprot_id} not found in the fasta file. Skipping the protein."
            )
            continue
        elif len(uniprot_seqs) > 1:
            Warning(
                f"Multiple proteins with the same ID {uniprot_id} found in the fasta file. Using the first one."
            )
        bio_seq = uniprot_seqs[0]
        if bio_seq.seq is None:
            raise ValueError(f"Protein sequence for {uniprot_id} is empty.")
        prot_seq = str(bio_seq.seq)
        prot_desc = bio_seq.description
        pept_df_r = rollup_single_protein_to_lytic_site(
            pept_df,
            int_cols,
            par.uniprot_col,
            prot_seq,
            residue_col=par.residue_col,
            description=prot_desc,
            tryptic_pattern="all",
            peptide_col=par.peptide_col,
            rollup_func="sum",
        )
        double_site.append(pept_df_r)
    return pd.concat(double_site)


# This function is to analyze the digestion site pattern of the peptides in LiP pept dataframe
def rollup_single_protein_to_lytic_site(
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
    rollup_func: Literal["median", "mean", "sum"] = "sum",
) -> pd.DataFrame:
    """
    Rolls up peptide-level limited proteolysis data to lytic sites.

    Args:
        df (pd.DataFrame): Input DataFrame containing peptide data.
        int_cols (Iterable[str]): Columns with intensity values to aggregate.
        uniprot_col (str): Column name for UniProt IDs.
        sequence (str): Protein sequence to analyze against.
        residue_col (str, optional): Column name for lytic residues. Defaults to "Residue".
        description (str, optional): Protein description to add to data frame. Defaults to "".
        tryptic_pattern (str, optional): Digestion pattern to filter peptides. Defaults to "all".
        peptide_col (str, optional): Column name containing peptide sequences. Defaults to "Sequence".
        clean_pept_col (str, optional): Column name for cleaned peptide sequences. Defaults to "clean_pept".
        id_separator (str, optional): Separator for IDs. Defaults to "@".
        id_col (str, optional): Column name for IDs. Defaults to "id".
        pept_type_col (str, optional): Column name for peptide types. Defaults to "pept_type".
        site_col (str, optional): Column name for lytic sites. Defaults to "Site".
        pos_col (str, optional): Column name for positions. Defaults to "Pos".
        multiply_rollup_counts (bool, optional): Whether to multiply rollup counts. Defaults to True.
        ignore_NA (bool, optional): Whether to ignore NA values. Defaults to True.
        alternative_protease (str, optional): Name of the alternative protease. Defaults to "ProK".
        rollup_func (Literal["median", "mean", "sum"], optional): Aggregation function. Defaults to "median".

    Returns:
        pd.DataFrame: DataFrame with rolled-up lytic site data and aggregated statistics.
    """
    if df.empty:
        Warning("The input dataframe is empty. Please check the input dataframe.")
        return df
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
        if start_lytic_pos == 0:
            start_lytic_site = "K"
            start_lytic_res = start_lytic_site + str(start_lytic_pos)
        else:
            start_lytic_site = sequence[start_lytic_pos - 1]
            start_lytic_res = start_lytic_site + str(start_lytic_pos)
        end_lytic_site = sequence[end_lytic_pos - 1]
        end_lytic_res = end_lytic_site + str(end_lytic_pos)
        lyticsites.append([start_lytic_res, end_lytic_res])

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
        Warning(
            f"The resulted dataframe of digestion site in {uniprot_id} is empty. "
            "Please check the input dataframe."
        )
        return protein2explode

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
                    i: lambda x: (
                        (np.log2(len(x)) + x.median())
                        if (not x[x.notna()].empty)
                        else np.nan
                    )
                    for i in int_cols
                }
            elif rollup_func.lower() == "mean":
                agg_methods_2: AggDictFloat = {
                    i: lambda x: (
                        (np.log2(len(x)) + x.mean())
                        if (not x[x.notna()].empty)
                        else np.nan
                    )
                    for i in int_cols
                }
            elif rollup_func.lower() == "sum":
                agg_methods_2: AggDictFloat = {i: expsum for i in int_cols}
            else:
                raise ValueError(
                    "The rollup function is not recognized. Please choose from the following: "
                    "median, mean, sum"
                )
        else:
            if rollup_func.lower() == "median":
                agg_methods_2: AggDictFloat = {
                    i: lambda x: (
                        (np.log2(x.notna().sum()) + x.median())
                        if (not x[x.notna()].empty)
                        else np.nan
                    )
                    for i in int_cols
                }
            elif rollup_func.lower() == "mean":
                agg_methods_2: AggDictFloat = {
                    i: lambda x: (
                        (np.log2(x.notna().sum()) + x.mean())
                        if (not x[x.notna()].empty)
                        else np.nan
                    )
                    for i in int_cols
                }
            elif rollup_func.lower() == "sum":
                agg_methods_2: AggDictFloat = {i: expsum for i in int_cols}
            else:
                raise ValueError(
                    "The rollup function is not recognized. Please choose from the following: "
                    "median, mean, sum"
                )
    else:
        if rollup_func.lower() == "median":
            agg_methods_2: AggDictFloat = {
                i: lambda x: (x.median() if (not x[x.notna()].empty) else np.nan)
                for i in int_cols
            }
        elif rollup_func.lower() == "mean":
            agg_methods_2: AggDictFloat = {
                i: lambda x: (x.mean() if (not x[x.notna()].empty) else np.nan)
                for i in int_cols
            }
        elif rollup_func.lower() == "sum":
            agg_methods_2: AggDictFloat = {i: expsum for i in int_cols}
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
    """
    Selects lytic sites based on the specified site type.

    Args:
        site_df (pd.DataFrame): Input DataFrame containing lytic site data.
        site_type (str, optional): Type of lytic site to select. Defaults to "prok".
        site_type_col (str, optional): Column name for lytic site types. Defaults to "Lytic site type".

    Returns:
        pd.DataFrame: Filtered DataFrame with selected lytic sites.
    """
    site_df_out = site_df[site_df[site_type_col] == site_type].copy()
    return site_df_out


def delta_prok_site(
    peptide_df: pd.DataFrame,
    site_df: pd.DataFrame,
    int_cols: list[str],
    site_type_col: str = "Type",
    site_protein_col: str = "Protein",
    pept_protein_col: str = "Protein",
    protein_length_col: str = "Protein length",
    site_pept_col: str = "Peptide",
    pept_pept_col: str = "Peptide",
    position_col: str = "Pos",
    pept_start_col: str = "pept_start",
    pept_end_col: str = "pept_end",
    rollup_method: Literal["median", "mean", "sum"] = "median",
) -> pd.DataFrame:
    """
    Computes exposure values for each lytic (ProK) site.

    This is computed as the average log intensity of peptides for which the site
    is a lytic site minus the average log intensity peptides that contain the
    site in their sequence. The average function is determined by the rollup_method parameter.

    Args:
        peptide_df (pd.DataFrame): DataFrame containing peptide data.
        site_df (pd.DataFrame): DataFrame containing lytic site data.
        int_cols (list[str]): List of columns to aggregate.
        site_type_col (str, optional): Column name for lytic site types. Defaults to "Type".
        site_protein_col (str, optional): Column name for protein IDs in the lytic site DataFrame. Defaults to "Protein".
        pept_protein_col (str, optional): Column name for protein IDs in the peptide DataFrame. Defaults to "Protein".
        protein_length_col (str, optional): Column name for protein lengths. Defaults to "Protein length".
        site_pept_col (str, optional): Column name for peptides in the lytic site DataFrame. Defaults to "Peptide".
        pept_pept_col (str, optional): Column name for peptides in the peptide DataFrame. Defaults to "Peptide".
        position_col (str, optional): Column name for positions in the lytic site DataFrame. Defaults to "Pos".
        pept_start_col (str, optional): Column name for start positions in the peptide DataFrame. Defaults to "pept_start".
        pept_end_col (str, optional): Column name for end positions in the peptide DataFrame. Defaults to "pept_end".
        rollup_method (Literal["median", "mean", "sum"], optional): Aggregation method to use. Defaults to "median". The "sum" is done in linear space.

    Returns:
        pd.DataFrame: DataFrame with delta values for each lytic site.
    """
    prok = site_df[site_df[site_type_col] == "ProK"]
    pept_dfs = dict(list(peptide_df.groupby(pept_protein_col)))
    df_rows = []
    for _, row in prok.iterrows():  # type: ignore
        pept_df = pept_dfs[
            row[site_protein_col]
        ]  # peptides for protein this site (row) is in
        if row[protein_length_col] == row[position_col]:
            continue
        pepts_to_match = cast("list[str]", row[site_pept_col].split("; "))
        pepts_cleaved_at_site = pept_df[pept_df[pept_pept_col].isin(pepts_to_match)]
        if pepts_cleaved_at_site.shape[0] == 0:
            raise ValueError(
                "Unable to match a peptide to a lytic site. Please verify that the site dataframe is derived from the peptide dataframe."
            )

        pepts_with_site_inside = cast(
            "pd.DataFrame",
            pept_df[
                (pept_df[pept_start_col] < row[position_col])
                & (pept_df[pept_end_col] > row[position_col])
            ],
        )

        if pepts_with_site_inside.shape[0] == 0:
            continue

        if rollup_method == "median":
            log_exposure_ratios = cast(
                "pd.Series[float]",
                pepts_cleaved_at_site[int_cols].median(axis=0)
                - pepts_with_site_inside[int_cols].median(axis=0),
            )
        elif rollup_method == "mean":
            log_exposure_ratios = cast(
                "pd.Series[float]",
                pepts_cleaved_at_site[int_cols].mean(axis=0)
                - pepts_with_site_inside[int_cols].mean(axis=0),
            )
        elif rollup_method == "sum":
            log_exposure_ratios = cast(
                "pd.Series[float]",
                pepts_cleaved_at_site[int_cols].aggregate(expsum, axis=0)
                - pepts_with_site_inside[int_cols].aggregate(expsum, axis=0),
            )
        else:
            raise ValueError(
                "The rollup method is not recognized. Please choose from the following: median, mean, sum"
            )

        site_info = pd.Series(
            {
                "Protein": row[site_protein_col],
                "Pos": row[position_col],
            }
        )

        df_rows.append(pd.concat([site_info, log_exposure_ratios], axis=0))

    df_out = pd.DataFrame(df_rows)
    return df_out
