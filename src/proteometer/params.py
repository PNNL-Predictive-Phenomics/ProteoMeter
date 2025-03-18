# type: ignore
import os


# TODO: make configurable by user with a config file
class Params:
    def __init__(self):
        self.working_dir = "."
        self.data_dir = f"{os.path.abspath(self.working_dir)}/data/LiP"  # same
        self.result_dir = f"{os.path.abspath(self.working_dir)}/results/LiP"  # same
        os.Path(self.result_dir).mkdir(parents=True, exist_ok=True)

        # All the required files
        self.fasta_file = f"{self.data_dir}/reference_proteome.fasta"
        self.metadata_file = f"{self.data_dir}/metadata.txt"  #
        self.global_prot_file = f"{self.data_dir}/trypsin_prot.tsv"
        self.global_pept_file = f"{self.data_dir}/trypsin_pept.tsv"
        self.double_pept_file = f"{self.data_dir}/double_pept.tsv"

        # Experiment information
        self.experiment_name = "Test"  # same

        # this might not be necessary anymore, maybe fore preprocessing, but not
        # here
        self.search_tool = "FragPipe"
        self.experiment_type = "Label-free"  # TMT or Label-free

        # Statistics setup
        self.pairwise_factor = "Time"
        self.anova_factors = ["Treatment", self.pairwise_factor]  # Optional

        # normali the batch correction only for TMT data

        # If it is TMT experiment then batch correction might be needed. User
        # need to provide a list of column names of samples are used for batch
        # correction. If "" provided, then batch_correct_samples should be all
        # samples except the pooled channel samples
        self.batch_correct_samples = ""

        # TMT data are usually processed into log2 scale, but not always
        self.log2_scale = False
        # If there is multiple batches
        self.batch_correction = (
            True  # If False, then no batch correction will be performed
        )

        # Unique to TMT data
        self.pooled_chanel_condition = "Total"

        # User defined pair-wise comparison groups, possibly need a function to
        # generate from a list of group pairs
        self.user_ttest_pairs = [
            ["Infected_8h", "Infected_16h"],
            ["Infected_8h", "Infected_16h"],
        ]

        ### Unique for PTM data
        self.phospho_symbol = "#"
        self.redox_symbol = "@"
        self.acetyl_symbol = "@"
        self.phospho_ascore_col = "AScore"

        self.id_separator = "@"  # same
        self.sig_thr = 0.05  # same
        self.sig_type = "pval"  # same either "pval" or "adj-p"
        self.missing_thr = 0  # same
        self.min_pept_count = 2  # unique to lip

        self.metadata_batch_col = "Batch"
        self.metadata_sample_col = "Sample"
        self.metadata_group_col = "Group"
        self.metadata_condition_col = "Condition"
        self.metadata_control_condition = "Control"
        self.metadata_treatment_condition = "Treatment"

        # Output table columns
        self.id_col = "id"
        self.uniprot_col = "UniProt"
        self.protein_col = "Protein"
        self.peptide_col = "Peptide"
        self.site_col = "Site"
        self.residue_col = "Residue"
        self.type_col = "Type"
        self.experiment_col = "Experiment"
        self.site_number_col = "site_number"
