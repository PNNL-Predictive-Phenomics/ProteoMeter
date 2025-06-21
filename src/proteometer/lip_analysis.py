from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations_with_replacement
from typing import cast

import pandas as pd

import proteometer.abundance as abundance
import proteometer.fasta as fasta
import proteometer.lip as lip
import proteometer.normalization as normalization
import proteometer.parse_metadata as parse_metadata
import proteometer.stats as stats
from proteometer.params import Params
from proteometer.utils import filter_missingness, generate_index


def lip_analysis(
    par: Params, drop_samples: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs statistical analysis on the provided limited proteolysis data.

    Args:
        par (Params): Parameters for the limited proteolysis analysis, including
            file paths and settings.
        drop_samples (list[str], optional): List of samples to drop from the
            analysis. Defaults to None.

    Returns:
        tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]: The resulting limited
            proteolysis data frames after analysis.
            These are the double-digested peptide data frame, the rollup of the
            peptide data to the single site, and the processed global protein
            data frame (in that order).
    """
    if drop_samples is None:
        drop_samples = []

    prot_seqs = fasta.get_sequences_from_fasta(par.fasta_file)

    metadata = pd.read_csv(par.metadata_file, sep="\t")
    metadata = metadata[~metadata[par.metadata_sample_col].isin(drop_samples)]

    global_prot = pd.read_csv(par.global_prot_file, sep="\t").drop(columns=drop_samples)
    global_pept = pd.read_csv(par.global_pept_file, sep="\t").drop(columns=drop_samples)
    double_pept = pd.read_csv(par.double_pept_file, sep="\t").drop(columns=drop_samples)

    int_cols = parse_metadata.int_columns(metadata, par)
    anova_cols = parse_metadata.anova_columns(metadata, par)
    group_cols, groups = parse_metadata.group_columns(metadata, par)
    pairwise_ttest_groups = parse_metadata.t_test_groups(metadata, par)

    if not par.search_tool:
        raise ValueError(
            "Please specify the search tool used to generate the LiP input files in the configuration file."
            "Currently supported tools: MaxQuant, MSFragger, FragPipe."
        )

    double_pept = lip.filter_contaminants_reverse_pept(
        double_pept, par.search_tool, par.protein_col, par.uniprot_col
    )
    global_pept = lip.filter_contaminants_reverse_pept(
        global_pept, par.search_tool, par.protein_col, par.uniprot_col
    )
    global_prot = lip.filter_contaminants_reverse_prot(
        global_prot, par.search_tool, par.protein_col, par.uniprot_col
    )

    double_pept = generate_index(
        double_pept, par.uniprot_col, par.peptide_col, par.id_separator
    )
    global_pept = generate_index(
        global_pept, par.uniprot_col, par.peptide_col, par.id_separator
    )
    global_prot = generate_index(global_prot, par.uniprot_col)

    if not par.log2_scale:  # if not already in log2, transform it
        double_pept = stats.log2_transformation(double_pept, int_cols)
        global_pept = stats.log2_transformation(global_pept, int_cols)
        global_prot = stats.log2_transformation(global_prot, int_cols)

    double_pept = filter_missingness(
        double_pept, groups, group_cols, par.min_replicates_qc
    )
    global_pept = filter_missingness(
        global_pept, groups, group_cols, par.min_replicates_qc
    )
    global_prot = filter_missingness(
        global_prot, groups, group_cols, par.min_replicates_qc
    )

    # must correct protein abundance, before we can use it to correct peptide
    # data; depending on normalization scheme, we may need to test significance
    # of deviations also, so statistics must be calculated for `global_prot`
    # before `global_pept` and `double_pept`
    global_prot = abundance.global_prot_normalization_and_stats(
        global_prot=global_prot,
        int_cols=int_cols,
        anova_cols=anova_cols,
        pairwise_ttest_groups=pairwise_ttest_groups,
        metadata=metadata,
        par=par,
    )

    double_pept = normalization.peptide_normalization(
        global_pept=global_pept,
        mod_pept=double_pept,
        int_cols=int_cols,
        par=par,
    )

    double_site = lip.rollup_to_lytic_site(
        double_pept,
        prot_seqs,
        int_cols,
        par,
    )

    if par.batch_correction:
        normalization.batch_correction(
            double_pept,
            metadata,
            par.batch_correct_samples,
            batch_col=par.metadata_batch_col,
            sample_col=par.metadata_sample_col,
        )
        normalization.batch_correction(
            double_site,
            metadata,
            par.batch_correct_samples,
            batch_col=par.metadata_batch_col,
            sample_col=par.metadata_sample_col,
        )

    if par.abundance_correction:
        double_pept = abundance.prot_abund_correction(
            double_pept,
            global_prot,
            par,
            columns_to_correct=int_cols,
            pairwise_ttest_groups=pairwise_ttest_groups,
        )
        double_site = abundance.prot_abund_correction(
            double_site,
            global_prot,
            par,
            columns_to_correct=int_cols,
            pairwise_ttest_groups=pairwise_ttest_groups,
        )

    double_pept = _double_pept_statistics(
        double_pept,
        prot_seqs,
        anova_cols,
        pairwise_ttest_groups,
        metadata,
        par,
    )

    double_site = _double_site_statistics(
        double_site,
        anova_cols,
        pairwise_ttest_groups,
        metadata,
        par,
    )

    global_prot = _annotate_global_prot(global_prot, par)
    double_site = _annotate_double_site(double_site, par)

    return double_pept, double_site, global_prot


def _double_pept_statistics(
    double_pept: pd.DataFrame,
    prot_seqs: list[fasta.SeqRecord],
    anova_cols: list[str],
    pairwise_ttest_groups: Iterable[stats.TTestGroup],
    metadata: pd.DataFrame,
    par: Params,
) -> pd.DataFrame:
    pept_list: list[pd.DataFrame] = []
    if anova_cols:
        double_pept = stats.anova(
            double_pept,
            anova_cols,
            metadata,
            par.anova_factors,
            par.metadata_sample_col,
        )
    double_pept = stats.pairwise_ttest(double_pept, pairwise_ttest_groups)
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
        factor_names = [
            f"[{f1} * {f2}]" if f1 != f2 else f"[{f1}]"
            for f1, f2 in combinations_with_replacement(par.anova_factors, r=2)
        ]
        pept_df = lip.get_tryptic_types(pept_df, prot_seq, par.peptide_col)
        for factor_name in factor_names:
            pept_df = lip.analyze_tryptic_pattern(
                pept_df,
                prot_seq,
                pairwise_ttest_groups,
                par.peptide_col,
                description=prot_desc,
                anova_type=factor_name,
                id_separator=par.id_separator,
                sig_type=par.sig_type,
                sig_thr=par.sig_thr,
            )
        if pept_df.shape[0] < 1:
            Warning(
                f"Protein {uniprot_id} has no peptides that could be mapped to the sequence. Skipping the protein."
            )
            continue

        pept_list.append(pept_df)
    return pd.concat(pept_list)


def _double_site_statistics(
    double_site: pd.DataFrame,
    anova_cols: list[str],
    pairwise_ttest_groups: Iterable[stats.TTestGroup],
    metadata: pd.DataFrame,
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
    if anova_cols:
        double_site = stats.anova(
            double_site,
            anova_cols,
            metadata,
            par.anova_factors,
            par.metadata_sample_col,
        )

    double_site = stats.pairwise_ttest(double_site, pairwise_ttest_groups)

    return double_site


def _annotate_global_prot(global_prot: pd.DataFrame, par: Params) -> pd.DataFrame:
    """
    Annotates the global protein data frame with additional columns for analysis.

    Args:
        global_prot (pd.DataFrame): The global proteomics data frame to be annotated.
        par (Params): Parameters containing column names and separators for annotation.

    Returns:
        pd.DataFrame: The annotated global protein data frame with additional columns such as type, experiment, residue, site, and protein.
    """

    global_prot[par.type_col] = "Global"
    global_prot[par.experiment_col] = "LiP"
    global_prot[par.residue_col] = "GLB"
    global_prot[par.site_col] = (
        global_prot[par.uniprot_col]
        + par.id_separator
        + global_prot[par.residue_col].astype(str)
    )
    global_prot[par.protein_col] = global_prot[par.protein_col].map(
        lambda x: x.split("|")[-1]  # type: ignore
    )

    return global_prot


def _annotate_double_site(double_site: pd.DataFrame, par: Params) -> pd.DataFrame:
    double_site[par.type_col] = [
        "Tryp"
        if (
            i.split(par.id_separator)[1][0] == "K"
            or i.split(par.id_separator)[1][0] == "R"
        )
        else "ProK"
        for i in cast(Iterable[str], double_site.index)
    ]
    double_site[par.experiment_col] = "LiP"
    double_site[par.residue_col] = double_site[par.site_col]
    double_site[par.site_col] = (
        double_site[par.uniprot_col] + par.id_separator + double_site[par.site_col]
    )
    double_site[par.protein_col] = double_site[par.protein_col].map(
        lambda x: x.split("|")[-1]  # type: ignore
    )
    return double_site
