from __future__ import annotations

import os
from typing import Literal

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

        self.id_separator = str(cfg["symbols"]["id_separator"])

        self.ptm_names = [str(x) for x in cfg["symbols"]["ptm"]["ptm_names"]]
        self.ptm_pept_files = [
            f"{self.data_dir}/{ptm_pept_file}"
            for ptm_pept_file in cfg["paths"]["ptm"]["ptm_pept_files"]
        ]

        if len(self.ptm_names) != len(self.ptm_pept_files):
            raise ValueError(
                "The number of ptm names must be equal to the number of ptm pept files"
            )

        self.ptm_symbols = [str(x) for x in cfg["symbols"]["ptm"]["ptm_symbols"]]

        if len(cfg["symbols"]["ptm"]["ptm_symbols"]) != len(self.ptm_names):
            raise ValueError(
                "The number of ptm names must be equal to the number of ptm symbols"
            )

        if len(cfg["symbols"]["ptm"]["ptm_abbreviations"]) != len(self.ptm_names):
            raise ValueError(
                "The number of ptm names must be equal to the number of ptm abbreviations"
            )
        self.ptm_abbreviations = {
            name: str(x)
            for name, x in zip(
                self.ptm_names, cfg["symbols"]["ptm"]["ptm_abbreviations"]
            )
        }

        # Experiment information
        self.experiment_name = str(cfg["experiment"]["experiment_name"])

        search_tool = str(cfg["experiment"]["lip"]["search_tool"]).lower()
        if search_tool not in {
            "maxquant",
            "msfragger",
            "fragpipe",
            "",
        }:
            raise ValueError(
                f"LiP search tool {search_tool} must be 'maxquant', 'msfragger', or 'fragpipe'."
                "Others are not currently supported. Use '' if not importing LiP data."
            )
        self.search_tool: Literal["maxquant", "msfragger", "fragpipe", ""] = search_tool  # type: ignore

        if cfg["experiment"]["experiment_type"] not in {"TMT", "Label-free"}:
            raise ValueError("Experiment type must be 'TMT' or 'Label-free'")
        self.experiment_type: Literal["TMT", "Label-free"] = cfg["experiment"][
            "experiment_type"
        ]  # TMT or Label-free

        # Statistics setup
        self.anova_factors = [str(x) for x in cfg["statistics"]["anova_factors"]]
        self.ttest_pairs = [
            [str(x) for x in y] for y in cfg["statistics"]["ttest_pairs"]
        ]

        # Abundance correction, generally recommended to help decompose effects
        # of changing protein abundance from changes in the fraction of protein
        # in a modified state and to reduce noise. However, sometimes only the
        # total concentration of one protein form (e.g., its active form) is of
        # interest, and so we may wish to skip this step when we don't care
        # about the source of the change.
        self.abundance_correction = bool(cfg["corrections"]["abundance_correction"])

        # When global proteomics data and PTM/LiP data are drawn from the same
        # samples (i.e., they are paired), we can use this pairing to correct
        # for abundance changes. Otherwise, we must rely on a statistical test
        # of the population averages (with threshhold given by
        # `abudnance_unpaired_sig_thr`)
        self.abundance_correction_paired_samples = bool(
            cfg["corrections"]["abundance_correction_paired_samples"]
        )
        self.abudnance_unpaired_sig_thr = float(
            cfg["corrections"]["abundance_unpaired_sig_thr"]
        )

        # normaly the batch correction only for TMT data
        # If it is TMT experiment then batch correction might be needed. User
        # need to provide a list of column names of samples are used for batch
        # correction.
        self.batch_correct_samples = [
            str(x) for x in cfg["corrections"]["batch_correct_samples"]
        ]

        # TMT data are usually processed into log2 scale, but not always
        self.log2_scale = bool(cfg["corrections"]["log2_scale"])
        # If there are multiple batches
        self.batch_correction = bool(cfg["corrections"]["batch_correction"])

        # Unique to TMT data
        self.pooled_chanel_condition = str(
            cfg["corrections"]["pooled_chanel_condition"]
        )

        self.sig_thr = float(cfg["corrections"]["sig_thr"])

        if cfg["corrections"]["sig_type"] not in {"pval", "adj-p"}:
            raise ValueError("sig_type must be 'pval' or 'adj-p'")
        self.sig_type: Literal["pval", "adj-p"] = cfg["corrections"]["sig_type"]

        self.min_replicates_qc = int(cfg["corrections"]["min_replicates_qc"])
        self.min_pept_count = int(cfg["corrections"]["min_pept_count"])

        self.metadata_batch_col = str(cfg["metadata"]["metadata_batch_col"])
        self.metadata_sample_col = str(cfg["metadata"]["metadata_sample_col"])
        self.metadata_group_col = str(cfg["metadata"]["metadata_group_col"])
        self.metadata_condition_col = str(cfg["metadata"]["metadata_condition_col"])
        self.metadata_control_condition = str(
            cfg["metadata"]["metadata_control_condition"]
        )
        self.metadata_treatment_condition = str(
            cfg["metadata"]["metadata_treatment_condition"]
        )

        # Output table columns
        self.id_col = str(cfg["data_columns"]["id_col"])
        self.uniprot_col = str(cfg["data_columns"]["uniprot_col"])
        self.protein_col = str(cfg["data_columns"]["protein_col"])
        self.peptide_col = str(cfg["data_columns"]["peptide_col"])
        self.site_col = str(cfg["data_columns"]["site_col"])
        self.residue_col = str(cfg["data_columns"]["residue_col"])
        self.type_col = str(cfg["data_columns"]["type_col"])
        self.experiment_col = str(cfg["data_columns"]["experiment_col"])
        self.site_number_col = str(cfg["data_columns"]["site_number_col"])
