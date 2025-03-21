# type: ignore
import pandas as pd

import proteometer.abundance as abundance
import proteometer.fasta as fasta
import proteometer.lip as lip
import proteometer.normalization as normalization
import proteometer.parse_metadata as parse_metadata
import proteometer.stats as stats
from proteometer.params import Params
from proteometer.utils import check_missingness, generate_index


def lip_analysis(par: Params):
    prot_seqs = fasta.get_sequences_from_fasta(par.fasta_file)
    metadata = pd.read_csv(par.metadata_file, sep="\t")
    global_prot = pd.read_csv(par.global_prot_file, sep="\t")
    global_pept = pd.read_csv(par.global_pept_file, sep="\t")
    double_pept = pd.read_csv(par.double_pept_file, sep="\t")

    int_cols = parse_metadata.int_columns(metadata, par)
    anova_cols = parse_metadata.anova_columns(metadata, par)
    group_cols, groups = parse_metadata.group_columns(metadata, par)
    pairwise_ttest_groups = parse_metadata.t_test_groups(metadata, par)
    user_pairwise_ttest_groups = parse_metadata.user_t_test_groups(metadata, par)

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
    global_prot = abundance.global_prot_normalization_and_stats(
        global_prot=global_prot,
        int_cols=int_cols,
        anova_cols=anova_cols,
        pairwise_ttest_groups=pairwise_ttest_groups,
        user_pairwise_ttest_groups=user_pairwise_ttest_groups,
        metadata=metadata,
        par=par,
    )

    double_pept = normalization.peptide_normalization_and_correction(
        global_pept=global_pept,
        mod_pept=double_pept,
        int_cols=int_cols,
        metadata=metadata,
        par=par,
    )

    if par.abundance_correction:
        double_pept = abundance.prot_abund_correction(
            double_pept, global_prot, int_cols, par
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

    all_lips = check_missingness(all_lips, groups, group_cols)

    return all_lips


def _double_site(
    double_pept,
    prot_seqs,
    int_cols,
    anova_cols,
    pairwise_ttest_groups,
    user_pairwise_ttest_groups,
    metadata,
    par: Params,
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
            residue_col=par.residue_col,
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
        if anova_cols:
            pept_df_a = stats.anova(pept_df_r, anova_cols, metadata)
            pept_df_a = stats.anova(pept_df_a, anova_cols, metadata, par.anova_factors)
        pept_df_p = stats.pairwise_ttest(pept_df_a, pairwise_ttest_groups)
        pept_df_p = stats.pairwise_ttest(pept_df_p, user_pairwise_ttest_groups)
        double_site.append(pept_df_p)
    double_site = pd.concat(double_site).copy()


def _annotate_global_prot(global_prot, par: Params):
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


def _annotate_double_site(double_site, par: Params):
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
