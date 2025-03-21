# type: ignore
import pandas as pd

import proteometer.abundance as abundance
import proteometer.normalization as normalization
import proteometer.parse_metadata as parse_metadata
import proteometer.ptm as ptm
import proteometer.rollup as rollup
import proteometer.stats as stats
from proteometer.params import Params
from proteometer.utils import check_missingness, generate_index


def ptm_analysis(par: Params | None = None):
    if par is None:
        par = Params(ptm_version=True)

    metadata = pd.read_csv(par.metadata_file, sep="\t")
    global_prot = pd.read_csv(par.global_prot_file, sep="\t")
    global_pept = pd.read_csv(par.global_pept_file, sep="\t")
    ptm_pept = [pd.read_csv(f, sep="\t") for f in par.ptm_pept_files]

    int_cols = parse_metadata.int_columns(metadata, par)
    anova_cols = parse_metadata.anova_columns(metadata, par)
    group_cols, groups = parse_metadata.group_columns(metadata, par)
    pairwise_ttest_groups = parse_metadata.t_test_groups(metadata, par)
    user_pairwise_ttest_groups = parse_metadata.user_t_test_groups(metadata, par)

    ptm_pept = [
        generate_index(pept, par.uniprot_col, par.peptide_col, par.id_separator)
        for pept in ptm_pept
    ]

    global_prot = generate_index(global_prot, par.uniprot_col)

    if not par.log2_scale:  # if not already in log2, transform it
        ptm_pept = [stats.log2_transformation(pept, int_cols) for pept in ptm_pept]
        global_pept = stats.log2_transformation(global_pept, int_cols)
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

    ptm_pept = normalization.peptide_normalization_and_correction(
        ptm_pept=ptm_pept,
        global_pept=global_pept,
        int_cols=int_cols,
        metadata=metadata,
        par=par,
    )

    if par.abundance_correction:
        ptm_pept = [
            abundance.prot_abund_correction(pept, global_prot, int_cols, par)
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
            rollup_func="Sum",
        )
        for pept in ptm_pept
    ]

    ptm_rolled = _rollup_stats(
        ptm_rolled=ptm_rolled,
        anova_cols=anova_cols,
        groups=groups,
        pairwise_ttest_groups=pairwise_ttest_groups,
        user_pairwise_ttest_groups=user_pairwise_ttest_groups,
        metadata=metadata,
        par=par,
    )

    ptm_dict = {"global": global_prot}
    ptm_dict.update({name: rolled for name, rolled in zip(par.ptm_names, ptm_rolled)})
    all_ptms = ptm.combine_multi_ptms(
        ptm_dict,
        par.residue_col,
        par.uniprot_col,
        par.site_col,
        par.site_number_col,
        par.id_separator,
        par.id_col,
    )

    all_ptms = check_missingness(all_ptms, groups, group_cols)

    return all_ptms


def _rollup_stats(
    ptm_rolled,
    anova_cols,
    groups,
    pairwise_ttest_groups,
    user_pairwise_ttest_groups,
    metadata,
    par: Params,
):
    if len(groups) > 2:
        ptm_rolled = [
            stats.anova(
                rolled, anova_cols, metadata, par.anova_factors, par.metadata_sample_col
            )
            for rolled in ptm_rolled
        ]
    ptm_rolled = [
        stats.pairwise_ttest(rolled, pairwise_ttest_groups) for rolled in ptm_rolled
    ]
    ptm_rolled = [
        stats.pairwise_ttest(rolled, user_pairwise_ttest_groups)
        for rolled in ptm_rolled
    ]

    return ptm_rolled
