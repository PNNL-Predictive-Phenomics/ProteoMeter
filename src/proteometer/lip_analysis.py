# type: ignore
import numpy as np
import pandas as pd
from params import Params

import proteometer.abundance as abundance
import proteometer.fasta as fasta
import proteometer.lip as lip
import proteometer.normalization as normalization
import proteometer.stats as stats
from proteometer.utils import check_missingness, generate_index


def lip_analysis(par: Params | None = None):
    if par is None:
        par = Params()

    prot_seqs = fasta.get_sequences_from_fasta(par.fasta_file)
    metadata = pd.read_csv(par.metadata_file, sep="\t")
    global_prot = pd.read_csv(par.global_prot_file, sep="\t")
    global_pept = pd.read_csv(par.global_pept_file, sep="\t")
    double_pept = pd.read_csv(par.double_pept_file, sep="\t")

    int_cols = _int_columns(metadata, par)
    anova_cols = _anova_columns(metadata, par)
    group_cols = _group_columns(metadata, par)
    pairwise_ttest_groups = _t_test_groups(metadata, par)
    user_pairwise_ttest_groups = _user_t_test_groups(metadata, par)

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

    # must correct protein abundance, before we can use it to correct peptide
    # data; depending on normalization scheme, we may need to test significance
    # of deviations also, so statistics must be calculated for `global_prot`
    # before `global_pept` and `double_pept`
    global_prot = _global_prot_normalization_and_stats(
        global_prot=global_prot,
        int_cols=int_cols,
        anova_cols=anova_cols,
        pairwise_ttest_groups=pairwise_ttest_groups,
        user_pairwise_ttest_groups=user_pairwise_ttest_groups,
        metadata=metadata,
        par=par,
    )

    global_pept, double_pept = _peptide_normalization_and_correction(
        global_pept=global_pept,
        double_pept=double_pept,
        global_prot=global_prot,
        int_cols=int_cols,
        metadata=metadata,
        par=par,
    )

    double_site = _double_site(
        double_pept,
        prot_seqs,
        int_cols,
        anova_cols,
        pairwise_ttest_groups,
        user_pairwise_ttest_groups,
        metadata,
        par,
    )

    global_prot = _annotate_global_prot(global_prot, par)
    double_site = _annotate_double_site(double_site, par)

    all_lips = (
        pd.concat([global_prot, double_site], axis=0, join="outer", ignore_index=True)
        .sort_values(by=["id", "Type", "Experiment", "Site"])
        .reset_index(drop=True)
    )

    all_lips = check_missingness(all_lips, par.groups, group_cols)

    return all_lips


def _group_columns(metadata: pd.DataFrame, par: Params):
    control_groups = list(
        metadata[
            metadata[par.metadata_condition_col] == par.metadata_control_condition
        ][par.metadata_group_col].unique()
    )
    control_group_cols = [
        metadata[metadata[par.metadata_group_col] == group][
            par.metadata_sample_col
        ].to_list()
        for group in control_groups
    ]
    treat_groups = list(
        metadata[
            metadata[par.metadata_condition_col] == par.metadata_treatment_condition
        ][par.metadata_group_col].unique()
    )
    treat_group_cols = [
        metadata[metadata[par.metadata_group_col] == group][
            par.metadata_sample_col
        ].to_list()
        for group in treat_groups
    ]
    return control_group_cols + treat_group_cols


def _int_columns(metadata: pd.DataFrame, par: Params):
    return metadata[par.metadata_sample_col].to_list()


def _anova_columns(metadata: pd.DataFrame, par: Params):
    tt_groups = list(
        metadata[metadata[par.metadata_condition_col] == par.pooled_chanel_condition][
            par.metadata_group_col
        ].unique()
    )
    tt_group_cols = [
        metadata[metadata[par.metadata_group_col] == group][
            par.metadata_sample_col
        ].to_list()
        for group in tt_groups
    ]
    anova_cols = [
        sample
        for sample in metadata[par.metadata_sample_col].values
        if sample not in np.flatten(tt_group_cols)
    ]
    return anova_cols


# TODO: get better types for t-test groups
def _t_test_groups(metadata: pd.DataFrame, par: Params):
    pairwise_pars = metadata[par.pairwise_factor].unique()
    pairwise_ttest_groups = []
    for pairwise_par in pairwise_pars:
        for control_group in list(
            set(
                metadata[
                    (
                        metadata[par.metadata_condition_col]
                        == par.metadata_control_condition
                    )
                    & (metadata[par.pairwise_factor] == pairwise_par)
                ][par.metadata_group_col]
            )
        ):
            for treat_group in list(
                set(
                    metadata[
                        (
                            metadata[par.metadata_condition_col]
                            == par.metadata_treatment_condition
                        )
                        & (metadata[par.pairwise_factor] == pairwise_par)
                    ][par.metadata_group_col]
                )
            ):
                pairwise_ttest_groups.append(
                    [
                        f"{treat_group}/{control_group}",
                        control_group,
                        treat_group,
                        metadata[metadata[par.metadata_group_col] == control_group][
                            par.metadata_sample_col
                        ].to_list(),
                        metadata[metadata[par.metadata_group_col] == treat_group][
                            par.metadata_sample_col
                        ].to_list(),
                    ]
                )

    return pairwise_ttest_groups


def _user_t_test_groups(metadata: pd.DataFrame, par: Params):
    user_pairwise_ttest_groups = []
    for user_test_pair in par.user_ttest_pairs:
        user_ctrl_group = user_test_pair[0]
        user_treat_group = user_test_pair[1]
        user_pairwise_ttest_groups.append(
            [
                f"{user_treat_group}/{user_ctrl_group}",
                user_ctrl_group,
                user_treat_group,
                metadata[metadata[par.metadata_group_col] == user_ctrl_group][
                    par.metadata_sample_col
                ].to_list(),
                metadata[metadata[par.metadata_group_col] == user_treat_group][
                    par.metadata_sample_col
                ].to_list(),
            ]
        )
    return user_pairwise_ttest_groups


def _global_prot_normalization_and_stats(
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


def _peptide_normalization_and_correction(
    global_pept: pd.DataFrame,
    double_pept: pd.DataFrame,
    global_prot: pd.DataFrame,
    int_cols: list[str],
    metadata: pd.DataFrame,
    par: Params,
):
    if par.experiment_type == "TMT":
        double_pept = normalization.tmt_normalization(
            double_pept, global_pept, int_cols
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
        double_pept = normalization.median_normalization(double_pept, int_cols)
        global_pept = normalization.median_normalization(global_pept, int_cols)

    if par.batch_correction:
        double_pept = normalization.batch_correction(
            double_pept,
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

    double_pept = abundance.prot_abund_correction(
        double_pept, global_prot, int_cols, par.uniprot_col
    )
    global_pept = abundance.prot_abund_correction(
        global_pept, global_prot, int_cols, par.uniprot_col
    )

    return global_pept, double_pept


def _double_site(
    double_pept,
    prot_seqs,
    int_cols,
    anova_cols,
    pairwise_ttest_groups,
    user_pairwise_ttest_groups,
    metadata,
    par,
):
    double_site = []
    for uniprot_id in double_pept[par.uniprot_col].unique():
        pept_df = double_pept[double_pept[par.uniprot_col] == uniprot_id].copy()
        uniprot_seq = [prot_seq for prot_seq in prot_seqs if uniprot_id in prot_seq.id]
        if len(uniprot_seq) < 1:
            Warning(
                f"Protein {uniprot_id} not found in the fasta file. Skipping the protein."
            )
            continue
        elif len(uniprot_seq) > 1:
            Warning(
                f"Multiple proteins with the same ID {uniprot_id} found in the fasta file. Using the first one."
            )
        bio_seq = uniprot_seq[0]
        prot_seq = bio_seq.seq
        prot_desc = bio_seq.description
        pept_df_r = lip.rollup_to_lytic_site(
            pept_df,
            int_cols,
            par.uniprot_col,
            prot_seq,
            residue_col="Residue",
            description=prot_desc,
            tryptic_pattern="all",
            peptide_col=par.peptide_col,
            rollup_func="median",
        )
        if pept_df_r is None or pept_df_r.shape[0] < 1:
            Warning(
                f"Protein {uniprot_id} has no peptides that could be mapped to the sequence. Skipping the protein."
            )
            continue
        if len(par.groups) > 2:
            pept_df_a = stats.anova(pept_df_r, anova_cols, metadata)
            pept_df_a = stats.anova(pept_df_a, anova_cols, metadata, par.anova_factors)
        pept_df_p = stats.pairwise_ttest(pept_df_a, pairwise_ttest_groups)
        pept_df_p = stats.pairwise_ttest(pept_df_p, user_pairwise_ttest_groups)
        double_site.append(pept_df_p)
    double_site = pd.concat(double_site).copy()


def _annotate_global_prot(global_prot, par):
    global_prot[par.type_col] = "Global"
    global_prot[par.experiment_col] = "LiP"
    global_prot[par.residue_col] = "GLB"
    global_prot[par.site_col] = (
        global_prot[par.uniprot_col]
        + par.id_separator
        + global_prot[par.residue_col].astype(str)
    )
    global_prot[par.protein_col] = global_prot[par.protein_col].map(
        lambda x: x.split("|")[-1]
    )

    return global_prot


def _annotate_double_site(double_site, par):
    double_site[par.type_col] = [
        "Tryp"
        if (
            i.split(par.id_separator)[1][0] == "K"
            or i.split(par.id_separator)[1][0] == "R"
        )
        else "ProK"
        for i in double_site.index
    ]
    double_site[par.experiment_col] = "LiP"
    double_site[par.residue_col] = double_site[par.site_col]
    double_site[par.site_col] = (
        double_site[par.uniprot_col] + par.id_separator + double_site[par.site_col]
    )
    double_site[par.protein_col] = double_site[par.protein_col].map(
        lambda x: x.split("|")[-1]
    )
    return double_site
