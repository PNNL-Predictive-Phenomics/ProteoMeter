# type: ignore
import pandas as pd
from params import Params

import proteometer.abundance as abundance
import proteometer.normalization as normalization
import proteometer.parse_metadata as parse_metadata
import proteometer.ptm as ptm
import proteometer.rollup as rollup
import proteometer.stats as stats
from proteometer.utils import check_missingness, generate_index


def ptm_analysis(par: Params | None = None):
    if par is None:
        par = Params(ptm_version=True)

    metadata = pd.read_csv(par.metadata_file, sep="\t")
    global_prot = pd.read_csv(par.global_prot_file, sep="\t")
    global_pept = pd.read_csv(par.global_pept_file, sep="\t")
    redox_pept = pd.read_csv(par.redox_pept_file, sep="\t")
    phospho_pept = pd.read_csv(par.phospho_pept_file, sep="\t")
    acetyl_pept = pd.read_csv(par.acetyl_pept_file, sep="\t")

    int_cols = parse_metadata.int_columns(metadata, par)
    anova_cols = parse_metadata.anova_columns(metadata, par)
    group_cols = parse_metadata.group_columns(metadata, par)
    pairwise_ttest_groups = parse_metadata.t_test_groups(metadata, par)
    user_pairwise_ttest_groups = parse_metadata.user_t_test_groups(metadata, par)

    redox_pept = generate_index(
        redox_pept, par.uniprot_col, par.peptide_col, par.id_separator
    )
    phospho_pept = generate_index(
        phospho_pept, par.uniprot_col, par.peptide_col, par.id_separator
    )
    acetyl_pept = generate_index(
        acetyl_pept, par.uniprot_col, par.peptide_col, par.id_separator
    )
    global_prot = generate_index(global_prot, par.uniprot_col)

    if not par.log2_scale:  # if not already in log2, transform it
        redox_pept = stats.log2_transformation(redox_pept, int_cols)
        phospho_pept = stats.log2_transformation(phospho_pept, int_cols)
        acetyl_pept = stats.log2_transformation(acetyl_pept, int_cols)
        global_prot = stats.log2_transformation(global_prot, int_cols)

    # must correct protein abundance, before we can use it to correct peptide
    # data; depending on normalization scheme, we may need to test significance
    # of deviations also, so statistics must be calculated for `global_prot`
    # before `global_pept` and `double_pept`
    global_prot = abundance.global_prot_normalization_and_stats(
        global_prot=global_prot,
        int_cols=int_cols,
        anova_cols=anova_cols,
        pairwise_ttest_groups=pairwise_ttest_groups,
        user_pairwise_ttest_groups=user_pairwise_ttest_groups,
        metadata=metadata,
        par=par,
    )

    redox, phospho, acetyl = _peptide_normalization_and_correction(
        redox_pept=redox_pept,
        phospho_pept=phospho_pept,
        acetyl_pept=acetyl_pept,
        global_pept=global_pept,
        global_prot=global_prot,
        int_cols=int_cols,
        metadata=metadata,
        par=par,
    )

    redox, phospho, acetyl = _rollup_stats(
        redox=redox,
        phospho=phospho,
        acetyl=acetyl,
        anova_cols=anova_cols,
        pairwise_ttest_groups=pairwise_ttest_groups,
        user_pairwise_ttest_groups=user_pairwise_ttest_groups,
        metadata=metadata,
        par=par,
    )

    all_ptms = ptm.combine_multi_ptms(
        {"global": global_prot, "redox": redox, "phospho": phospho, "acetyl": acetyl},
        par.residue_col,
        par.uniprot_col,
        par.site_col,
        par.site_number_col,
        par.id_separator,
        par.id_col,
    )

    all_ptms = check_missingness(all_ptms, par.groups, group_cols)

    return all_ptms


def _peptide_normalization_and_correction(
    redox_pept: pd.DataFrame,
    phospho_pept: pd.DataFrame,
    acetyl_pept: pd.DataFrame,
    global_pept: pd.DataFrame,
    global_prot: pd.DataFrame,
    int_cols: list[str],
    metadata: pd.DataFrame,
    par: Params,
):
    if par.experiment_type == "TMT":
        redox_pept = normalization.tmt_normalization(redox_pept, global_pept, int_cols)
        phospho_pept = normalization.tmt_normalization(
            phospho_pept, global_pept, int_cols
        )
        acetyl_pept = normalization.tmt_normalization(
            acetyl_pept, global_pept, int_cols
        )
        if par.batch_correction:
            global_pept = normalization.median_normalization(
                global_pept,
                int_cols,
                metadata,
                par.batch_correct_samples,
                batch_col=par.metadata_batch_col,
                sample_col=par.metadata_sample_col,
            )
    else:
        redox_pept = normalization.median_normalization(redox_pept, int_cols)
        phospho_pept = normalization.median_normalization(phospho_pept, int_cols)
        acetyl_pept = normalization.median_normalization(acetyl_pept, int_cols)
        global_pept = normalization.median_normalization(global_pept, int_cols)

    redox = rollup.rollup_to_site(
        redox_pept,
        int_cols,
        par.uniprot_col,
        par.peptide_col,
        par.residue_col,
        ";",
        par.id_col,
        par.id_separator,
        par.site_col,
        rollup_func="Sum",
    )
    phospho = rollup.rollup_to_site(
        phospho_pept,
        int_cols,
        par.uniprot_col,
        par.peptide_col,
        par.residue_col,
        ";",
        par.id_col,
        par.id_separator,
        par.site_col,
        rollup_func="Sum",
    )
    acetyl = rollup.rollup_to_site(
        acetyl_pept,
        int_cols,
        par.uniprot_col,
        par.peptide_col,
        par.residue_col,
        ";",
        par.id_col,
        par.id_separator,
        par.site_col,
        rollup_func="Sum",
    )

    if par.batch_correction:
        redox = normalization.batch_correction(
            redox,
            metadata,
            par.batch_correct_samples,
            batch_col=par.metadata_batch_col,
            sample_col=par.metadata_sample_col,
        )
        phospho = normalization.batch_correction(
            phospho,
            metadata,
            par.batch_correct_samples,
            batch_col=par.metadata_batch_col,
            sample_col=par.metadata_sample_col,
        )
        acetyl = normalization.batch_correction(
            acetyl,
            metadata,
            par.batch_correct_samples,
            batch_col=par.metadata_batch_col,
            sample_col=par.metadata_sample_col,
        )
        global_pept = normalization.batch_correction(
            global_pept,
            metadata,
            par.batch_correct_samples,
            batch_col=par.metadata_batch_col,
            sample_col=par.metadata_sample_col,
        )

    redox = abundance.prot_abund_correction(
        redox, global_prot, int_cols, par.uniprot_col
    )
    phospho = abundance.prot_abund_correction(
        phospho, global_prot, int_cols, par.uniprot_col
    )
    acetyl = abundance.prot_abund_correction(
        acetyl, global_prot, int_cols, par.uniprot_col
    )
    global_pept = abundance.prot_abund_correction(
        global_pept, global_prot, int_cols, par.uniprot_col
    )

    return global_pept, redox, phospho, acetyl


def _rollup_stats(
    redox,
    phospho,
    acetyl,
    anova_cols,
    pairwise_ttest_groups,
    user_pairwise_ttest_groups,
    metadata,
    par,
):
    if len(par.groups) > 2:
        redox = par.anova(redox, anova_cols, metadata)
        phospho = par.anova(phospho, anova_cols, metadata)
        acetyl = par.anova(acetyl, anova_cols, metadata)
        redox = par.anova(redox, anova_cols, metadata, par.anova_factors)
        phospho = par.anova(phospho, anova_cols, metadata, par.anova_factors)
        acetyl = par.anova(acetyl, anova_cols, metadata, par.anova_factors)
    redox = stats.pairwise_ttest(redox, pairwise_ttest_groups)
    phospho = stats.pairwise_ttest(phospho, pairwise_ttest_groups)
    acetyl = stats.pairwise_ttest(acetyl, pairwise_ttest_groups)

    redox = stats.pairwise_ttest(redox, user_pairwise_ttest_groups)
    phospho = stats.pairwise_ttest(phospho, user_pairwise_ttest_groups)
    acetyl = stats.pairwise_ttest(acetyl, user_pairwise_ttest_groups)

    return redox, phospho, acetyl
