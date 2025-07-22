from __future__ import annotations

import pandas as pd

import proteometer.abundance as abundance
import proteometer.normalization as normalization
import proteometer.parse_metadata as parse_metadata
import proteometer.ptm as ptm
import proteometer.rollup as rollup
import proteometer.stats as stats
from proteometer.params import Params
from proteometer.utils import filter_missingness, generate_index


def ptm_analysis(
    par: Params, drop_samples: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs the PTM proteomics processing and statistical analysis pipeline.

    This function reads in data from proteomics files specified in the `par`
    object, and performs normalization and batch correction, site-level rollup,
    and statistical analysis. The resulting DataFrame contains the processed PTM
    data, including the statistical results.

    Args:
        par (Params): A Params object that contains all the parameters for the analysis.
        drop_samples (list[str], optional): A list of samples to be dropped from the analysis. Default is None.

    Returns:
        tuple[pd.DataFrame,pd.DataFrame]: Two pandas DataFrames that contains the result of the PTM analysis.
            The first is the processed PTM data, and the second is the global proteomics data.
    """
    if drop_samples is None:
        drop_samples = []

    metadata = pd.read_csv(par.metadata_file, sep="\t")
    metadata = metadata[~metadata[par.metadata_sample_col].isin(drop_samples)]

    global_prot = pd.read_csv(par.global_prot_file, sep="\t").drop(columns=drop_samples)
    global_pept = pd.read_csv(par.global_pept_file, sep="\t").drop(columns=drop_samples)
    ptm_pept = [pd.read_csv(f, sep="\t") for f in par.ptm_pept_files]

    int_cols = parse_metadata.int_columns(metadata, par)
    anova_cols = parse_metadata.anova_columns(metadata, par)
    group_cols, groups = parse_metadata.group_columns(metadata, par)
    pairwise_ttest_groups = parse_metadata.t_test_groups(metadata, par)

    ptm_pept = [
        generate_index(pept, par.uniprot_col, par.peptide_col, par.id_separator)
        for pept in ptm_pept
    ]

    global_prot = generate_index(global_prot, par.uniprot_col)

    if not par.log2_scale:  # if not already in log2, transform it
        ptm_pept = [stats.log2_transformation(pept, int_cols) for pept in ptm_pept]
        global_pept = stats.log2_transformation(global_pept, int_cols)
        global_prot = stats.log2_transformation(global_prot, int_cols)

    ptm_pept = [
        filter_missingness(pept, groups, group_cols, par.min_replicates_qc)
        for pept in ptm_pept
    ]
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

    ptm_pept = [
        normalization.peptide_normalization(
            global_pept=global_pept,
            mod_pept=pept,
            int_cols=int_cols,
            par=par,
        )
        for pept in ptm_pept
    ]

    ptm_rolled = [
        rollup.rollup_to_site(
            pept,
            int_cols,
            par.uniprot_col,
            par.peptide_col,
            par.residue_col,
            ";",
            par.id_col,
            par.id_separator,
            par.site_col,
            rollup_func="sum",
        )
        for pept in ptm_pept
    ]

    if par.batch_correction:
        ptm_rolled = [
            normalization.batch_correction(
                mod_pept,
                metadata,
                par.batch_correct_samples,
                batch_col=par.metadata_batch_col,
                sample_col=par.metadata_sample_col,
            )
            for mod_pept in ptm_rolled
        ]

    if par.abundance_correction:
        ptm_rolled = [
            abundance.prot_abund_correction(
                pept,
                global_prot,
                par,
                columns_to_correct=int_cols,
            )
            for pept in ptm_rolled
        ]

    ptm_rolled = _rollup_stats(
        ptm_rolled=ptm_rolled,
        anova_cols=anova_cols,
        pairwise_ttest_groups=pairwise_ttest_groups,
        metadata=metadata,
        par=par,
    )

    global_prot = ptm.combine_multi_ptms({"global": global_prot}, par)
    ptm_dict: dict[str, pd.DataFrame] = {}
    ptm_dict.update({name: rolled for name, rolled in zip(par.ptm_names, ptm_rolled)})
    all_ptms = ptm.combine_multi_ptms(ptm_dict, par)

    all_ptms = filter_missingness(
        all_ptms, groups, group_cols, par.min_replicates_qc
    )
    global_prot = filter_missingness(
        global_prot, groups, group_cols, par.min_replicates_qc
    )

    return all_ptms, global_prot


def ptm_analysis_return_all(par: Params) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Runs the PTM proteomics processing and statistical analysis pipeline.

    This function reads in data from proteomics files specified in the `par`
    object, and performs normalization and batch correction, site-level rollup,
    and statistical analysis. The resulting DataFrame contains the processed PTM
    data, including the statistical results.

    Args:
        par: A Params object that contains all the parameters for the analysis.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Three pandas DataFrames containing the results of the PTM analysis.
            The first is the processed PTM data, which includes statistical results.
            The second is the global proteomics data, normalized and batch-corrected.
            The third is the uncorrected PTM data, which contains raw site-level data before normalization or correction.
    """
    metadata = pd.read_csv(par.metadata_file, sep="\t")
    global_prot = pd.read_csv(par.global_prot_file, sep="\t")
    global_pept = pd.read_csv(par.global_pept_file, sep="\t")
    ptm_pept = [pd.read_csv(f, sep="\t") for f in par.ptm_pept_files]

    int_cols = parse_metadata.int_columns(metadata, par)
    anova_cols = parse_metadata.anova_columns(metadata, par)
    group_cols, groups = parse_metadata.group_columns(metadata, par)
    pairwise_ttest_groups = parse_metadata.t_test_groups(metadata, par)

    ptm_pept = [
        generate_index(pept, par.uniprot_col, par.peptide_col, par.id_separator)
        for pept in ptm_pept
    ]

    global_prot = generate_index(global_prot, par.uniprot_col)

    if not par.log2_scale:  # if not already in log2, transform it
        ptm_pept = [stats.log2_transformation(pept, int_cols) for pept in ptm_pept]
        global_pept = stats.log2_transformation(global_pept, int_cols)
        global_prot = stats.log2_transformation(global_prot, int_cols)

    ptm_pept = [
        filter_missingness(pept, groups, group_cols, par.min_replicates_qc)
        for pept in ptm_pept
    ]
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

    ptm_pept = [
        normalization.peptide_normalization(
            global_pept=global_pept,
            mod_pept=pept,
            int_cols=int_cols,
            par=par,
        )
        for pept in ptm_pept
    ]

    ptm_rolled = [
        rollup.rollup_to_site(
            pept,
            int_cols,
            par.uniprot_col,
            par.peptide_col,
            par.residue_col,
            ";",
            par.id_col,
            par.id_separator,
            par.site_col,
            rollup_func="sum",
        )
        for pept in ptm_pept
    ]

    if par.batch_correction:
        ptm_rolled = [
            normalization.batch_correction(
                mod_pept,
                metadata,
                par.batch_correct_samples,
                batch_col=par.metadata_batch_col,
                sample_col=par.metadata_sample_col,
            )
            for mod_pept in ptm_rolled
        ]

    ptm_rolled_uncorrected = ptm_rolled.copy()

    ptm_rolled_uncorrected = _rollup_stats(
        ptm_rolled=ptm_rolled_uncorrected,
        anova_cols=anova_cols,
        pairwise_ttest_groups=pairwise_ttest_groups,
        metadata=metadata,
        par=par,
    )

    ptm_rolled = [
        abundance.prot_abund_correction(
            pept,
            global_prot,
            par,
            columns_to_correct=int_cols,
        )
        for pept in ptm_rolled
    ]

    ptm_rolled = _rollup_stats(
        ptm_rolled=ptm_rolled,
        anova_cols=anova_cols,
        pairwise_ttest_groups=pairwise_ttest_groups,
        metadata=metadata,
        par=par,
    )

    global_prot = ptm.combine_multi_ptms({"global": global_prot}, par)
    ptm_dict: dict[str, pd.DataFrame] = {}
    ptm_dict.update({name: rolled for name, rolled in zip(par.ptm_names, ptm_rolled)})
    all_ptms = ptm.combine_multi_ptms(ptm_dict, par)
    ptm_dict_uncorrected: dict[str, pd.DataFrame] = {}
    ptm_dict_uncorrected.update({name: rolled for name, rolled in zip(par.ptm_names, ptm_rolled_uncorrected)})
    all_ptms_uncorrected = ptm.combine_multi_ptms(ptm_dict_uncorrected, par)

    all_ptms = filter_missingness(
        all_ptms, groups, group_cols, par.min_replicates_qc
    )
    global_prot = filter_missingness(
        global_prot, groups, group_cols, par.min_replicates_qc
    )
    all_ptms_uncorrected = filter_missingness(
        all_ptms_uncorrected, groups, group_cols, par.min_replicates_qc
    )

    return all_ptms, global_prot, all_ptms_uncorrected


def _rollup_stats(
    ptm_rolled: list[pd.DataFrame],
    anova_cols: list[str],
    pairwise_ttest_groups: list[stats.TTestGroup],
    metadata: pd.DataFrame,
    par: Params,
):
    """Perform statistical analysis on rolled up PTM data.

    This function applies ANOVA and pairwise t-tests on the provided
    list of DataFrames containing rolled up PTM data, if the relevant
    columns are specified.

    Args:
        ptm_rolled (list[pd.DataFrame]): List of DataFrames with rolled up PTM data.
        anova_cols (list[str]): List of column names for performing ANOVA.
        pairwise_ttest_groups (list[stats.TTestGroup]): List of TTestGroup objects
            specifying control-treatment pairs for pairwise t-tests.
        metadata (pd.DataFrame): DataFrame containing metadata for ANOVA analysis.
        par (Params): Parameter object containing configuration for statistical analysis.

    Returns:
        list[pd.DataFrame]: List of DataFrames with statistical analysis applied.
    """

    if anova_cols:
        ptm_rolled = [
            stats.anova(
                rolled, anova_cols, metadata, par.anova_factors, par.metadata_sample_col
            )
            for rolled in ptm_rolled
        ]
    ptm_rolled = [
        stats.pairwise_ttest(rolled, pairwise_ttest_groups) for rolled in ptm_rolled
    ]

    return ptm_rolled
