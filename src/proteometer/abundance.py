from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import cast

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import proteometer.normalization as normalization
import proteometer.stats as stats
from proteometer.params import Params


def get_prot_abund_scalars(
    prot: pd.DataFrame,
    pairwise_ttest_name: str,
    sig_type: str = "pval",
    sig_thr: float = 0.05,
) -> dict[str, float]:
    """Return a dictionary of protein abundance scalars for the given pairwise t-test.

    Args:
        prot (pd.DataFrame): DataFrame containing protein-level data.
        pairwise_ttest_name (str): Name of the pairwise t-test.
        sig_type (str, optional): Type of significance metric to use for filtering.
            Defaults to "pval".
        sig_thr (float, optional): Threshold for significance filtering. Defaults to 0.05.

    Returns:
        dict[str, float]: Dictionary of protein abundance scalars.
    """
    prot = stats.calculate_pairwise_scalars(
        prot, pairwise_ttest_name, sig_type, sig_thr
    )
    scalar_dict = dict(
        zip(
            cast(Iterable[str], prot.index),
            cast(Iterable[float], prot[f"{pairwise_ttest_name}_scalar"]),
        )
    )
    return scalar_dict


def prot_abund_correction(
    pept: pd.DataFrame,
    prot: pd.DataFrame,
    par: Params,
    columns_to_correct: Iterable[str] | None = None,
    pairwise_ttest_groups: Iterable[stats.TTestGroup] | None = None,
    non_tt_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Perform protein abundance correction based on the provided parameters.

    This function applies either paired or unpaired sample abundance correction
    depending on the `abundance_correction_paired_samples` attribute of the `par` parameter.

    Args:
        pept (pd.DataFrame): A DataFrame containing peptide-level data.
        prot (pd.DataFrame): A DataFrame containing protein-level data.
        par (Params): A parameter object containing configuration for abundance correction.
        columns_to_correct (Iterable[str] | None, optional):
            Columns to correct for paired sample abundance correction.
            Required if `par.abundance_correction_paired_samples` is True.
        pairwise_ttest_groups (Iterable[stats.TTestGroup] | None, optional):
            Groups for pairwise t-tests in unpaired sample abundance correction.
            Required if `par.abundance_correction_paired_samples` is False.
        non_tt_cols (Iterable[str] | None, optional):
            Columns that should not be included in the t-test correction.

    Returns:
        pd.DataFrame: A DataFrame with corrected protein abundances.

    Raises:
        ValueError: If `columns_to_correct` is not provided for paired sample correction.
        ValueError: If `pairwise_ttest_groups` is not provided for unpaired sample correction.
    """
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
    pept: pd.DataFrame,
    prot: pd.DataFrame,
    pairwise_ttest_groups: Iterable[stats.TTestGroup],
    uniprot_col: str,
    sig_type: str = "pval",
    sig_thr: float = 0.05,
) -> pd.DataFrame:
    """
    Adjusts peptide abundance values based on protein abundance scalars for
    significant pairwise t-test groups.

    This function iterates over a collection of pairwise t-test groups, computes
    or retrieves protein abundance scalars, and applies these scalars to adjust
    the peptide abundance values for the specified treatment samples.

    Args:
        pept (pd.DataFrame): DataFrame containing peptide-level data. Must include
            a column corresponding to `uniprot_col` for mapping protein identifiers.
        prot (pd.DataFrame): DataFrame containing protein-level data. Must include
            columns for protein abundance scalars or data required to compute them.
        pairwise_ttest_groups (Iterable[stats.TTestGroup]): An iterable of
            TTestGroup objects, each representing a pairwise t-test group with
            associated metadata (e.g., labels and treatment samples).
        uniprot_col (str): Column name in `pept` that contains UniProt identifiers
            for mapping to protein abundance data.
        sig_type (str, optional): Type of significance metric to use for filtering
            (e.g., "pval" for p-value or "adj-p" for adjusted p-value). Defaults to "pval".
        sig_thr (float, optional): Threshold for significance filtering. Only proteins
            meeting this threshold will have their abundance scalars applied. Defaults to 0.05.

    Returns:
        pd.DataFrame: Updated `pept` DataFrame with adjusted abundance values for
            treatment samples and additional columns for protein abundance scalars.
    """
    for pairwise_ttest_group in pairwise_ttest_groups:
        if pairwise_ttest_group.label() not in prot.columns:
            scalar_dict = get_prot_abund_scalars(
                prot, pairwise_ttest_group.label(), sig_type, sig_thr
            )
        else:
            scalar_dict = dict(
                zip(
                    cast("pd.Index[str]", prot.index),
                    cast(
                        "pd.Series[float]",
                        prot[f"{pairwise_ttest_group.label()}"],
                    ),
                )
            )
        pept[f"{pairwise_ttest_group.label()}_scalar"] = [
            scalar_dict.get(uniprot_id, 0)
            for uniprot_id in cast("pd.Series[str]", pept[uniprot_col])
        ]
        pept[pairwise_ttest_group.treat_samples] = pept[  # type: ignore
            pairwise_ttest_group.treat_samples
        ].subtract(
            cast("pd.Series[float]", pept[f"{pairwise_ttest_group.label()}_scalar"]),
            axis=0,
        )
    return pept


# correct the PTM or LiP data using all protein abundance recommended approach
def prot_abund_correction_matched(
    pept: pd.DataFrame,
    prot: pd.DataFrame,
    columns_to_correct: Iterable[str],
    uniprot_col: str,
    non_tt_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Correct the peptide abundance data using the protein abundance values.

    This function takes the peptide data and corrects the intensity values
    for each peptide using the protein abundance values from the protein
    data. The correction is only applied to the treatment samples.

    Args:
        pept (pd.DataFrame): A DataFrame containing peptide-level data.
        prot (pd.DataFrame): A DataFrame containing protein-level data.
        columns_to_correct (Iterable[str]): Columns to correct for protein
            abundance changes. Must be shared by `pept` and `prot`.
        uniprot_col (str): Column name for the Uniprot ID in both `pept` and `prot`.
        non_tt_cols (Iterable[str] | None, optional): Columns that should not
            be included in the abundance correction. Must be shared by `pept` and `prot`.

    Returns:
        pd.DataFrame: Updated `pept` DataFrame with adjusted abundance values
            for treatment samples and additional columns for protein abundance
            scalars.
    """
    pept_new: list[pd.DataFrame] = []
    if non_tt_cols is None:
        non_tt_cols = columns_to_correct
    for uniprot_id in pept[uniprot_col].unique():
        pept_sub = cast(pd.DataFrame, pept[pept[uniprot_col] == uniprot_id].copy())
        if uniprot_id in prot[uniprot_col].unique():
            prot_abund_row = cast(
                "pd.Series[float]", prot.loc[uniprot_id, columns_to_correct]
            )
            prot_abund = prot_abund_row.astype(float).fillna(0)
            prot_abund_median = cast(float, prot_abund_row[non_tt_cols].median())  # type: ignore
            if not np.isnan(prot_abund_median):
                prot_abund_scale = (~prot_abund_row.isna()).astype(
                    float
                ) * prot_abund_median
                pept_sub[columns_to_correct] = (  # type: ignore
                    pept_sub[columns_to_correct]
                    .sub(prot_abund, axis=1)
                    .add(prot_abund_scale, axis=1)
                )
        pept_new.append(pept_sub)

    return pd.concat(pept_new)


def global_prot_normalization_and_stats(
    global_prot: pd.DataFrame,
    int_cols: list[str],
    anova_cols: list[str],
    pairwise_ttest_groups: Iterable[stats.TTestGroup],
    metadata: pd.DataFrame,
    par: Params,
) -> pd.DataFrame:
    """
    Perform global protein normalization and statistical analysis.

    This function applies normalization and statistical tests to global proteomics
    data. It handles both median normalization and batch correction, depending on
    the parameters provided in the `par` object. It also performs ANOVA and pairwise t-tests.

    Args:
        global_prot (pd.DataFrame): DataFrame containing global protein-level data.
        int_cols (list[str]): List of column names representing intensity data to normalize.
        anova_cols (list[str]): List of column names for ANOVA analysis.
        pairwise_ttest_groups (Iterable[stats.TTestGroup]): Iterable of TTestGroup objects
            for performing pairwise t-tests (each defines a control-treatment pair).
        metadata (pd.DataFrame): DataFrame containing metadata for batch correction and
            ANOVA analysis.
        par (Params): Parameter object containing configuration for normalization and
            statistical analysis.

    Returns:
        pd.DataFrame: The normalized and statistically analyzed global protein data.
    """
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
    if anova_cols:
        global_prot = stats.anova(
            global_prot,
            anova_cols,
            metadata,
            par.anova_factors,
            par.metadata_sample_col,
        )
    global_prot = stats.pairwise_ttest(global_prot, pairwise_ttest_groups)

    return global_prot


def fasta_to_sequence_map(fasta_path: str) -> dict[str, str]:
    """
    Reads a FASTA file and creates a mapping of protein IDs to sequences.

    This function uses Biopython to parse a FASTA file. The identifier for each
    sequence is taken from the header line (the part before the first whitespace).

    Example FASTA entry:
    >sp|P12345|GENE_NAME Protein description...
    MKL...

    The resulting key-value pair would be:
    'sp|P12345|GENE_NAME': 'MKL...'

    Args:
        fasta_path (str): The path to the input FASTA file.

    Returns:
        dict[str, str]: A dictionary where keys are protein identifiers
                        and values are the corresponding amino acid sequences.
    """
    sequences: dict[str, str] = {}
    with open(fasta_path, "r") as fasta_data:
        for record in cast(Iterable[SeqRecord], SeqIO.parse(fasta_data, "fasta")):
            seq_id = cast(str, record.id)
            seq_str = str(cast(Seq, record.seq))
            sequences[seq_id] = seq_str
    return sequences


# Implementation of iBAQ calculation
def count_theoretical_peptides(
    sequence: str,
    min_len: int = 6,
    max_len: int = 30,
    missed_cleavages: int = 2,
    protease: str = "trypsin",
) -> int:
    """
    Count the number of theoretically observable tryptic peptides for a protein sequence.

    Args:
        sequence (str): The amino acid sequence of the protein.
        min_len (int, optional): Minimum length of peptides to count. Defaults to 6.
        max_len (int, optional): Maximum length of peptides to count. Defaults to 30.
        missed_cleavages (int, optional): The number of missed cleavages to allow. Defaults to 2.
        protease (str, optional): The protease used for digestion. Defaults to "trypsin".

    Returns:
        int: The number of theoretically observable peptides.
    """
    if not sequence:
        return 0

    # Standard tryptic cleavage rule: after K or R, but not if followed by P
    if protease.lower() == "trypsin":
        cleavage_sites = {
            i + 1
            for i, aa in enumerate(sequence[:-1])
            if aa in "KR" and sequence[i + 1] != "P"
        }
    else:
        raise ValueError("Unsupported protease: {}".format(protease))
    cleavage_sites.add(0)
    cleavage_sites.add(len(sequence))

    sites = sorted(list(cleavage_sites))
    count = 0

    for i in range(len(sites) - 1):
        for j in range(i + 1, min(i + 2 + missed_cleavages, len(sites))):
            start = sites[i]
            end = sites[j]
            pep_len = end - start
            if min_len <= pep_len <= max_len:
                count += 1
    return count


def calculate_ibaq(
    prot_df: pd.DataFrame,
    intensity_cols: list[str],
    sequences: Mapping[str, str],
    prot_id_col: str = "UniProt",
    min_pep_len: int = 6,
    max_pep_len: int = 30,
    missed_cleavages: int = 2,
    id_matching: str = "exact",
    log2scale_input: bool = False,
) -> pd.DataFrame:
    """
    Calculate iBAQ (intensity-based absolute quantification) values.

    iBAQ is calculated by dividing the protein intensities by the number of
    theoretically observable peptides.

    Args:
        prot_df (pd.DataFrame): DataFrame with protein data, including intensity
            columns. The DataFrame index must contain protein identifiers that
            correspond to the keys in the `sequences` mapping.
        intensity_cols (list[str]): List of column names with protein intensities.
        sequences (Mapping[str, str]): A mapping (e.g., a dictionary) from protein
            identifiers to their amino acid sequences. This can be generated from a
            FASTA file using libraries like Biopython.
        prot_id_col (str, optional): The name of the column containing protein IDs.
            Defaults to "UniProt".
        min_pep_len (int, optional): Minimum length of peptides to consider. Defaults to 6.
        max_pep_len (int, optional): Maximum length of peptides to consider. Defaults to 30.
        missed_cleavages (int, optional): The number of missed cleavages to allow. Defaults to 2.
        id_matching (str, optional): The method for matching protein IDs. Defaults to "exact".
            Other option would be "contain(s)" or "startswith".
        log2scale_input (bool, optional): Whether the input intensity values are already log2-transformed. Defaults to False.

    Returns:
        pd.DataFrame: A new DataFrame with iBAQ values for each sample. The original
            intensity columns are replaced by iBAQ columns with an "_iBAQ" suffix.
    """
    ibaq_df = prot_df.copy()

    # Map protein IDs from the index to their sequences
    # Create a mapping from protein IDs in the dataframe to sequences
    # by checking if the protein ID is a substring of any key in the sequences dictionary.
    # This handles cases where FASTA headers are more complex than just the UniProt ID.
    if id_matching.lower() == "exact":
        protein_sequences: pd.Series[str | None] = ibaq_df[prot_id_col].apply(  # type: ignore
            lambda prot_id: sequences.get(prot_id)  # type: ignore
        )
    elif id_matching.lower() == "contain" or id_matching.lower() == "contains":
        protein_sequences: pd.Series[str | None] = ibaq_df[prot_id_col].apply(  # type: ignore
            lambda prot_id: next(  # pyright: ignore[reportUnknownLambdaType]
                (seq for header, seq in sequences.items() if prot_id in header), None
            )
        )
    elif id_matching.lower() == "startswith":
        protein_sequences: pd.Series[str | None] = ibaq_df[prot_id_col].apply(  # type: ignore
            lambda prot_id: next(  # pyright: ignore[reportUnknownLambdaType]
                (
                    seq
                    for header, seq in sequences.items()
                    if header.startswith(prot_id)  # type: ignore
                ),
                None,
            )
        )
    else:
        raise ValueError(f"Unsupported id_matching value: {id_matching}")

    # Calculate the number of theoretical peptides for each protein
    theoretical_peptides: pd.Series[float] = protein_sequences.apply(
        lambda seq: count_theoretical_peptides(  # pyright: ignore[reportUnknownLambdaType]
            str(seq),  # type: ignore
            min_pep_len,
            max_pep_len,
            missed_cleavages,
        )
        if pd.notna(seq)  # pyright: ignore[reportUnknownArgumentType]
        else 0
    )

    # Avoid division by zero for proteins with no theoretical peptides
    theoretical_peptides = theoretical_peptides.replace(0, np.nan)

    # Calculate iBAQ for each intensity column
    for col in intensity_cols:
        if log2scale_input:
            ibaq_df[col] = ibaq_df[col].apply(lambda x: 2**x if pd.notna(x) else x)  # type: ignore
        ibaq_df[col] = ibaq_df[col] / theoretical_peptides
        if log2scale_input:
            ibaq_df[col] = ibaq_df[col].apply(
                lambda x: np.log2(x) if pd.notna(x) else x  # type: ignore
            )

    return ibaq_df


def calculate_ibaq_from_fasta(
    prot_df: pd.DataFrame,
    fasta_file: str,
    intensity_cols: list[str],
    prot_id_col: str = "UniProt",
    min_pep_len: int = 6,
    max_pep_len: int = 30,
    missed_cleavages: int = 2,
    id_matching: str = "exact",
    log2scale_input: bool = False,
) -> pd.DataFrame:
    """
    Calculate iBAQ (intensity-based absolute quantification) values from a FASTA file.

    Args:
        prot_df (pd.DataFrame): DataFrame containing protein information.
        fasta_file (str): Path to the FASTA file containing protein sequences.
        intensity_cols (list[str]): List of column names with protein intensities.
        prot_id_col (str, optional): The name of the column containing protein IDs. Defaults to "UniProt".
        min_pep_len (int, optional): Minimum length of peptides to consider. Defaults to 6.
        max_pep_len (int, optional): Maximum length of peptides to consider. Defaults to 30.
        missed_cleavages (int, optional): The number of missed cleavages to allow. Defaults to 2.
        id_matching (str, optional): The method for matching protein IDs. Defaults to "exact". Other options include "contains" and "startswith".
        log2scale_input (bool, optional): Whether the input intensity values are already log2-transformed. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame with iBAQ values for each sample.
    """
    sequences = fasta_to_sequence_map(fasta_file)
    return calculate_ibaq(
        prot_df,
        intensity_cols,
        sequences,
        prot_id_col=prot_id_col,
        min_pep_len=min_pep_len,
        max_pep_len=max_pep_len,
        missed_cleavages=missed_cleavages,
        id_matching=id_matching,
        log2scale_input=log2scale_input,
    )
