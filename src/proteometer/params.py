# type: ignore
import os

import tomllib


class Params:
    def __init__(self, toml_file_path: str) -> None:
        with open(toml_file_path, "rb") as toml_file:
            cfg = tomllib.load(toml_file)
        self.data_dir = f"{os.path.abspath(cfg['paths']['data_dir'])}"
        self.result_dir = f"{os.path.abspath(cfg['paths']['results_dir'])}"
        os.makedirs(self.result_dir, exist_ok=True)

        # All the required files
        self.fasta_file = f"{self.data_dir}/{cfg['paths']['fasta_file']}"
        self.metadata_file = f"{self.data_dir}/{cfg['paths']['metadata_file']}"
        self.global_prot_file = f"{self.data_dir}/{cfg['paths']['global_prot_file']}"
        self.global_pept_file = f"{self.data_dir}/{cfg['paths']['global_pept_file']}"

        self.double_pept_file = (
            f"{self.data_dir}/{cfg['paths']['lip']['double_pept_file']}"
        )

        self.id_separator = cfg["symbols"]["id_separator"]

        self.ptm_names = cfg["symbols"]["ptm"]["ptm_names"]
        self.ptm_pept_files = [
            f"{self.data_dir}/{ptm_pept_file}"
            for ptm_pept_file in cfg["paths"]["ptm"]["ptm_pept_files"]
        ]
        self.ptm_symbols = cfg["symbols"]["ptm"]["ptm_symbols"]

        # Experiment information
        self.experiment_name = cfg["experiment"]["experiment_name"]
        self.search_tool = cfg["experiment"]["search_tool"]
        self.experiment_type = cfg["experiment"]["experiment_type"]  # TMT or Label-free

        # Statistics setup
        self.pairwise_factor = cfg["statistics"]["pairwise_factor"]
        self.anova_factors = cfg["statistics"]["anova_factors"]  # Optional
        self.user_ttest_pairs = cfg["statistics"]["user_ttest_pairs"]

        # Abundance correction, generally recommended to help decompose effects
        # of changing protein abundance from changes in the fraction of protein
        # in a modified state and to reduce noise. However, sometimes only the
        # total concentration of one protein form (e.g., its active form) is of
        # interest, and so we may wish to skip this step when we don't care
        # about the source of the change.
        self.abundance_correction = cfg["corrections"]["abundance_correction"]

        # When global proteomics data and PTM/LiP data are drawn from the same
        # samples (i.e., they are paired), we can use this pairing to correct
        # for abundance changes. Otherwise, we must rely on a statistical test
        # of the population averages (with threshhold given by
        # `abudnance_unpaired_sig_thr`)
        self.abundance_correction_paired_samples = cfg["corrections"][
            "abundance_correction_paired_samples"
        ]
        self.abudnance_unpaired_sig_thr = cfg["corrections"][
            "abundance_unpaired_sig_thr"
        ]

        # normaly the batch correction only for TMT data
        # If it is TMT experiment then batch correction might be needed. User
        # need to provide a list of column names of samples are used for batch
        # correction. If "" provided, then batch_correct_samples should be all
        # samples except the pooled channel samples
        self.batch_correct_samples = cfg["corrections"]["batch_correct_samples"]

        # TMT data are usually processed into log2 scale, but not always
        self.log2_scale = cfg["corrections"]["log2_scale"]
        # If there are multiple batches
        self.batch_correction = cfg["corrections"]["batch_correction"]

        # Unique to TMT data
        self.pooled_chanel_condition = cfg["corrections"]["pooled_chanel_condition"]

        self.sig_thr = cfg["corrections"]["sig_thr"]
        self.sig_type = cfg["corrections"]["sig_type"]  # "pval" or "adj-p"
        self.missing_thr = cfg["corrections"]["missing_thr"]
        self.min_pept_count = cfg["corrections"]["min_pept_count"]  # unique to lip

        self.metadata_batch_col = cfg["metadata"]["metadata_batch_col"]
        self.metadata_sample_col = cfg["metadata"]["metadata_sample_col"]
        self.metadata_group_col = cfg["metadata"]["metadata_group_col"]
        self.metadata_condition_col = cfg["metadata"]["metadata_condition_col"]
        self.metadata_control_condition = cfg["metadata"]["metadata_control_condition"]
        self.metadata_treatment_condition = cfg["metadata"][
            "metadata_treatment_condition"
        ]

        # Output table columns
        self.id_col = cfg["data_columns"]["id_col"]
        self.uniprot_col = cfg["data_columns"]["uniprot_col"]
        self.protein_col = cfg["data_columns"]["protein_col"]
        self.peptide_col = cfg["data_columns"]["peptide_col"]
        self.site_col = cfg["data_columns"]["site_col"]
        self.residue_col = cfg["data_columns"]["residue_col"]
        self.type_col = cfg["data_columns"]["type_col"]
        self.experiment_col = cfg["data_columns"]["experiment_col"]
        self.site_number_col = cfg["data_columns"]["site_number_col"]
