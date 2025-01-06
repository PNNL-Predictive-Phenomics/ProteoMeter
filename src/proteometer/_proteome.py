# type: ignore
"""
Created on Thu Jan 18 09:02:15 2024

@author: cies677

>>> 1+1
2
>>> 1+2
3
"""

import csv
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.stats.multitest as multitest
from Bio import SeqIO as SeqIO

warnings.filterwarnings("ignore")


home = str(Path.home())


class Proteome:
    def __init__(
        self,
        filenames: dict[str, str],
        params: dict[str, Any],
        columnnames: dict[str, list[str]],
    ):
        self.fasta_name = filenames["fasta_file"]
        self.trypsin_pept_file = filenames["trypsin_pept_file"]
        self.trypsin_prot_file = filenames["trypsin_prot_file"]
        self.double_pept_file = filenames["double_pept_file"]
        self.experiment_type = params["experiment_type"]
        self.sample = params["sample"]
        self.data_type = params["data_type"]

        if params["id_separator"] is None:
            self.id_separator: str = "@"
        else:
            self.id_separator = params["id_separator"]

        if params["sig_thr"] is None:
            self.sig_thr: float = 0.05
        else:
            self.sig_thr = params["sig_thr"]

        if params["sig_thr_type"] is None:
            self.sig_thr_type = "adj-p"
        else:
            self.adj_thr = params["sig_thr_type"]

        if params["prot_missing_thr"] is None:
            self.prot_missing_thr = 0.5
        else:
            self.prot_missing_thr = params["prot_missing_thr"]

        if params["pept_missing_thr"] is None:
            self.pept_missing_thr = 0.5
        else:
            self.pept_missing_thr = params["pept_missing_thr"]

        if params["uniprot_col"] is None:
            self.uniprot_col = "Uniprot"
        else:
            self.uniprot_col = params["uniprot_col"]

        if params["data_type"].lower() == "lip":
            if params["lip_sig_num_thr"] is None:
                self.lip_sig_num_thr = 1
            else:
                self.lip_sig_num_thr = params["lip_sig_num_thr"]

            if params["lip_min_pept_count"] is None:
                self.lip_min_pept_count = 1
            else:
                self.lip_min_pept_count = params["lip_min_pept_count"]

            if params["lip_search_tool"] is None:
                raise NotImplementedError("Error: lip_search_tool must be declared")
            else:
                self.lip_search_tool = params["lip_search_tool"]

        self.double_ctrl_cols = columnnames["double_ctrl_cols"]
        self.double_treat_cols = columnnames["double_treat_cols"]
        self.double_int_cols = columnnames["double_int_cols"]
        self.trypsin_ctrl_cols = columnnames["trypsin_ctrl_cols"]
        self.trypsin_treat_cols = columnnames["trypsin_treat_cols"]
        self.trypsin_int_cols = columnnames["trypsin_int_cols"]
        self.prot_info_cols = columnnames["prot_info_cols"]
        self.info_cols = columnnames["info_cols"]
        self.prot_trypsin_cols = self.prot_info_cols + self.trypsin_int_cols
        self.trypsin_cols = self.info_cols + self.trypsin_int_cols
        self.double_cols = self.info_cols + self.double_int_cols

        self.prot_seq_obj = SeqIO.parse(self.fasta_name, "fasta")
        self.prot_seqs = [seq_item for seq_item in self.prot_seq_obj]

        # Implement a yml file to store this info and other read-in parameters
        # from users
        self.ProteinID_col_prot = None
        self.ProteinID_col_pept = None
        self.PeptCounts_col = None
        if self.lip_search_tool.lower() == "maxquant":
            self.ProteinID_col_prot = "Majority protein IDs"
            self.ProteinID_col_pept = "Leading razor protein"
            self.PeptCounts_col = "Peptide counts (all)"
        elif (
            (self.ProteinID_col_pept is not None)
            and (self.ProteinID_col_prot is not None)
            and (self.PeptCounts_col is not None)
        ):
            pass
        else:
            raise NotImplementedError(
                "The error was triggered because either the search"
                " tool is not specified or not columns specification"
                " are not provided. Please specify the search tool or"
                " provide the columns for protein IDs in both protein"
                " table and peptide table as well as peptide counts."
            )

        sniffer = csv.Sniffer()
        for file in [
            self.trypsin_pept_file,
            self.trypsin_prot_file,
            self.double_pept_file,
        ]:
            with open(file) as f:
                lines = f.readlines()
            check = sniffer.sniff(lines[0])
            if check.delimiter != "\t":
                raise NotImplementedError(
                    "Error: " + file + " must be tab delimited file"
                )
            else:
                self.trypsin_pept = pd.read_table(self.trypsin_pept_file, sep="\t")
                self.trypsin_prot = pd.read_csv(self.trypsin_prot_file, sep="\t")
                self.double_pept = pd.read_csv(self.double_pept_file, sep="\t")
                self.trypsin_pept_cols = self.trypsin_pept.columns.tolist()
                self.double_pept_cols = self.double_pept.columns.tolist()
                self.log = ["Proteome object created, data loaded, columns defined"]

        """ Replace zeros in relevant columns with np.nan """
        self.trypsin_pept[self.trypsin_int_cols] = self.trypsin_pept[
            self.trypsin_int_cols
        ].replace(0, np.nan)
        self.trypsin_prot[self.trypsin_int_cols] = self.trypsin_prot[
            self.trypsin_int_cols
        ].replace(0, np.nan)
        self.double_pept[self.double_int_cols] = self.double_pept[
            self.double_int_cols
        ].replace(0, np.nan)

        # return self

    # @classmethod
    def filter_pept(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[
            (df["Reverse"].isna())
            & (df["Potential contaminant"].isna())
            & (~df[self.ProteinID_col_pept].str.contains("(?i)Contaminant"))
            & (~df[self.ProteinID_col_pept].str.contains("(?i)REV__"))
            & (~df[self.ProteinID_col_pept].str.contains("(?i)CON__"))
        ].copy()
        df[self.uniprot_col] = df[self.ProteinID_col_pept].copy()
        return df

    def filter_prot(self, df: pd.DataFrame) -> pd.DataFrame:
        """filter reversed or potential contaminants"""

        if self.ProteinID_col_prot is None:
            raise ValueError("Error: ProteinID_col_prot is not specified")
        df = df[
            (df["Only identified by site"].isna())
            & (df["Reverse"].isna())
            & (df["Potential contaminant"].isna())
            & (~df[self.ProteinID_col_prot].str.contains("(?i)Contaminant"))
            & (~df[self.ProteinID_col_prot].str.contains("(?i)REV__"))
            & (~df[self.ProteinID_col_prot].str.contains("CON__"))
        ].copy()
        if type(df[self.ProteinID_col_prot]) is pd.Series[str]:
            raise TypeError(
                "Error: The ProteinID_col_prot column does not contain (only) strings."
            )
        df[self.uniprot_col] = [
            ids.split(";")[0] for ids in df[self.ProteinID_col_prot]
        ]
        """ filter protein groups with less than two peptides """
        df["Pept count"] = [
            int(count.split(";")[0]) for count in df[self.PeptCounts_col]
        ]
        df = df[df["Pept count"] >= self.lip_min_pept_count].copy()
        return df

    def prot_missingness(
        self,
        df: pd.DataFrame,
        ctrl_cols: list[str],
        treat_cols: list[str],
        thresh: float,
    ) -> pd.DataFrame:
        df["ctrl missingness"] = df[ctrl_cols].isna().sum(axis=1)
        df["treat missingness"] = df[treat_cols].isna().sum(axis=1)
        df["missingness"] = df["ctrl missingness"] + df["treat missingness"]
        df = df[
            ~(
                (df["ctrl missingness"] > thresh * len(ctrl_cols))
                | (df["treat missingness"] > thresh * len(treat_cols))
            )
        ].copy()
        return df

    def pept_missingness(
        self,
        df: pd.DataFrame,
        ctrl_cols: list[str],
        treat_cols: list[str],
        thresh: float,
    ) -> pd.DataFrame:
        df["ctrl missingness"] = df[ctrl_cols].isna().sum(axis=1)
        df["treat missingness"] = df[treat_cols].isna().sum(axis=1)
        df["missingness"] = df["ctrl missingness"] + df["treat missingness"]
        df = df[df["missingness"] < len(self.double_int_cols)].copy()
        df = df[
            ~(
                (df["ctrl missingness"] > thresh * len(ctrl_cols))
                & (df["ctrl missingness"] < len(ctrl_cols))
            )
            & ~(
                (df["treat missingness"] > thresh * len(treat_cols))
                & (df["treat missingness"] < len(treat_cols))
            )
        ].copy()
        return df

    def log2_transform(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        df[cols] = np.log2(df[cols])
        return df

    def median_normalization(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        median_correction = (
            df[cols].median(axis=0, skipna=True)
            - df[cols].median(axis=0, skipna=True).mean()
        )
        df[cols] = df[cols].sub(median_correction, axis=1)
        return df

    def generate_id(self, df: pd.DataFrame, df_type: str) -> pd.DataFrame:
        if df_type == "pept":
            df["id"] = df[self.uniprot_col] + self.id_separator + df["Sequence"]
        elif df_type == "prot":
            df["id"] = df[self.uniprot_col]
        df.index = df["id"]
        return df

    def filter_data(self) -> None:
        if self.lip_search_tool.lower() == "maxquant":
            if "Trypsin peptides filtered" not in self.log:
                self.trypsin_pept = self.filter_pept(self.trypsin_pept)
                self.log.append("Trypsin peptides filtered")
            if "Trypsin peptides missingness filtered" not in self.log:
                self.trypsin_pept = self.pept_missingness(
                    self.trypsin_pept,
                    self.trypsin_ctrl_cols,
                    self.trypsin_treat_cols,
                    self.pept_missing_thr,
                )
                self.log.append("Trypsin peptides missingness filtered")
            if "Trypsin proteins filtered" not in self.log:
                self.trypsin_prot = self.filter_prot(self.trypsin_prot)
                self.log.append("Trypsin proteins filtered")
            if "Trypsin proteins missinginess filtered" not in self.log:
                self.trypsin_prot = self.prot_missingness(
                    self.trypsin_prot,
                    self.trypsin_ctrl_cols,
                    self.trypsin_treat_cols,
                    self.prot_missing_thr,
                )
                self.log.append("Trypsin proteins missingness filtered")
            if "Double digest peptides filtered" not in self.log:
                self.double_pept = self.filter_pept(self.double_pept)
                self.log.append("Double digest peptides filtered")
            if "Double peptides missingness filtered" not in self.log:
                self.double_pept = self.pept_missingness(
                    self.double_pept,
                    self.double_ctrl_cols,
                    self.double_treat_cols,
                    self.pept_missing_thr,
                )
                self.log.append("Double peptides missingness filtered")

        else:
            warnings.warn(
                "Please specify the search tool or user need to filter"
                " the data by themselves. From here forward, data are"
                " treated as filtered already!",
                UserWarning,
            )
        return None

    def log2_transform_data(self) -> None:
        if "Trypsin peptides log2 transformed" not in self.log:
            self.trypsin_pept = self.log2_transform(
                self.trypsin_pept, self.trypsin_int_cols
            )
            self.log.append("Trypsin peptides log2 transformed")
        if "Trypsin proteins log2 transformed" not in self.log:
            self.trypsin_prot = self.log2_transform(
                self.trypsin_prot, self.trypsin_int_cols
            )
            self.log.append("Trypsin proteins log2 transformed")
        if "Double digest peptides log2 transformed" not in self.log:
            self.double_pept = self.log2_transform(
                self.double_pept, self.double_int_cols
            )
            self.log.append("Double digest peptides log2 transformed")

    def normalize_data(self) -> None:
        if "Trypsin peptides median normalized" not in self.log:
            self.trypsin_pept = self.median_normalization(
                self.trypsin_pept, self.trypsin_int_cols
            )
            self.log.append("Trypsin peptides median normalized")
        if "Trypsin proteins median normalized" not in self.log:
            self.trypsin_prot = self.median_normalization(
                self.trypsin_prot, self.trypsin_int_cols
            )
            self.log.append("Trypsin proteins median normalized")
        if "Double digest peptides median normalized" not in self.log:
            self.double_pept = self.median_normalization(
                self.double_pept, self.double_int_cols
            )
            self.log.append("Double digest peptides median normalized")

    def id_data(self) -> None:
        if "Trypsin peptides id generated" not in self.log:
            self.trypsin_pept = self.generate_id(self.trypsin_pept, "pept")
            self.log.append("Trypsin peptides id generated")
        if "Trypsin proteins id generated" not in self.log:
            self.trypsin_prot = self.generate_id(self.trypsin_prot, "prot")
            self.log.append("Trypsin proteins id generated")
        if "Double digest peptides id generated" not in self.log:
            self.double_pept = self.generate_id(self.double_pept, "pept")
            self.log.append("Double digest peptides id generated")

    ## how should we handle fillna here?
    def studentTtest(
        self,
        df: pd.DataFrame,
        ctrl_cols: list[str],
        treat_cols: list[str],
        fillval: float,
    ) -> pd.DataFrame:
        df["Treat/Control"] = (
            df[treat_cols].mean(axis=1) - df[ctrl_cols].mean(axis=1)
        ).fillna(0)
        df["Treat/Control_pval"] = sp.stats.ttest_ind(
            df[treat_cols], df[ctrl_cols], axis=1, nan_policy="omit"
        ).pvalue.fillna(0, inplace=True)  # not default fillna val
        df["Treat/Control_adj-p"] = sp.stats.false_discovery_control(
            df["Treat/Control_pval"].fillna(fillval)
        )
        df["Treat/Control_adj-p_BY"] = sp.stats.false_discovery_control(
            df["Treat/Control_pval"].fillna(fillval), method="by"
        )
        df["Treat/Control_adj-p_bonferroni"] = multitest.multipletests(
            df["Treat/Control_pval"].fillna(fillval), method="bonferroni"
        )[1]
        return df

    """ TECHNICALLY NOT BEING USED YET """

    def select_uniprot_ids(
        self, sig_df: pd.DataFrame, reg_df: pd.DataFrame, sig_num_thr: int = 0
    ) -> tuple[list[str], list[str], list[str]]:
        num_dict = sig_df[self.uniprot_col].value_counts().to_dict()
        sig_df["Sig Pept num"] = [num_dict.get(k, 0) for k in sig_df[self.uniprot_col]]
        reg_df["Sig Pept num"] = [num_dict.get(k, 0) for k in reg_df[self.uniprot_col]]
        reg_uniprot_ids = list(set(reg_df[self.uniprot_col]))
        sig_uniprot_ids = list(set(sig_df[self.uniprot_col]))
        top_sig_uniprot_ids = [k for k, v in num_dict.items() if v > sig_num_thr]
        return (reg_uniprot_ids, sig_uniprot_ids, top_sig_uniprot_ids)

    """ NOT BEING USED YET """

    def get_sig_double_pept(self, thr: float, thr_type: str) -> pd.DataFrame:
        if thr is None:
            thr = self.sig_thr
        if thr_type is None:
            thr_type = self.sig_thr_type

        if thr_type == "adj-p":
            x = self.double_pept[self.double_pept["Treat/Control_adj-p"] < thr].copy()
        elif thr_type == "p-val":
            x = self.double_pept[self.double_pept["Treat/Control_pval"] < thr].copy()
        else:
            raise NotImplementedError("Error: thr_type must be adj-p or p-val")
        return x

    ## this gets redundant with other functions
    def analyze_trypsin_prot(self) -> None:
        if "Trypsin proteins filtered" not in self.log:
            self.trypsin_prot = self.filter_prot(self.trypsin_prot)
            self.log.append("Trypsin proteins filtered")

        if "Trypsin proteins log2 transformed" not in self.log:
            self.trypsin_prot = self.log2_transform(
                self.trypsin_prot, self.trypsin_int_cols
            )
            self.log.append("Trypsin proteins log2 transformed")

        if "Trypsin proteins missinginess filtered" not in self.log:
            self.trypsin_prot = self.prot_missingness(
                self.trypsin_prot,
                self.trypsin_ctrl_cols,
                self.trypsin_treat_cols,
                self.prot_missing_thr,
            )
            self.log.append("Trypsin proteins missingness filtered")

        if "Trypsin proteins median normalized" not in self.log:
            self.trypsin_prot = self.median_normalization(
                self.trypsin_prot, self.trypsin_int_cols
            )
            self.log.append("Trypsin proteins median normalized")

        if "Trypsin proteins p-values generated" not in self.log:
            self.trypsin_prot = self.studentTtest(
                self.trypsin_prot, self.trypsin_ctrl_cols, self.trypsin_treat_cols, 1
            )
            self.log.append("Trypsin proteins p-values generated")

        if "Scalar dictionary generated" not in self.log:
            self.trypsin_prot["Scalar"] = [
                self.trypsin_prot["Treat/Control"][i] if p < self.sig_thr else 0
                for i, p in enumerate(self.trypsin_prot["Treat/Control_pval"])
            ]
            self.log.append("Scalar dictionary generated")

    ## this gets redundant with other functions
    def analyze_double_pept(self) -> None:
        if "Scalar dictionary generated" not in self.log:
            self.analyze_trypsin_prot()

        scalar_dict = dict(zip(self.trypsin_prot.index, self.trypsin_prot["Scalar"]))

        ## note that a copy is made here in the original, trying without copy for now
        self.double_pept["Scalar"] = [
            scalar_dict.get(uniprot_id, 0) for uniprot_id in self.double_pept["Uniprot"]
        ]

        if "Double digest peptides filtered" not in self.log:
            self.double_pept = self.filter_pept(self.double_pept)
            self.log.append("Double digest peptides filtered")

        if "Double peptides missingness filtered" not in self.log:
            self.double_pept = self.pept_missingness(
                self.double_pept,
                self.double_ctrl_cols,
                self.double_treat_cols,
                self.pept_missing_thr,
            )
            self.log.append("Double peptides missingness filtered")

        if "Double peptides p-values generated" not in self.log:
            self.double_pept = self.studentTtest(
                self.double_pept, self.double_ctrl_cols, self.double_treat_cols, 0
            )
            self.log.append("Double peptides p-values generated")


#    ''' SOMEHOW THIS IS ALL WRONG NOW '''
#     def run_analysis(self):
# #        scalar_dict = self.generate_scalar_dict()

#         self.double_pept = self.pept_analysis(
#             self.double_pept,
#             self.double_ctrl_cols,
#             self.double_treat_cols
#         )

#         self.an_trypsin_pept = self.pept_analysis(
#             self.trypsin_pept,
#             self.trypsin_ctrl_cols,
#             self.trypsin_treat_cols
#         )

#         self.double_uniprot_ids = self.uniprot_ids(
#             self.double_pept_sig,
#             self.double_pept
#         )

#         self.DD_nmrc_uniprot_ids = self.uniprot_ids(
#             self.double_pept_sig_nmrc,
#             self.an_double_pept_nmrc
#         )

#         self.trypsin_uniprot_ids = self.uniprot_ids(
#             self.trypsin_pept_sig,
#             self.an_trypsin_pept
#         )

#         self.T_nmrc_uniprot_ids = self.uniprot_ids(
#             self.trypsin_pept_sig_nmrc,
#             self.trypsin_double_pept_nmrc
#         )


if __name__ == "__main__":
    """DEFINE INPUT TYPES"""
    experiment_type = "P1T1"
    sample = "Crona"
    data_type = "LiP"
    lip_search_tool = "MaxQuant"

    """ CUSTOM PARAM LIST """
    ### read in yml file here to store all this data?
    id_separator = None
    sig_thr = None
    sig_thr_type = None
    prot_missing_thr = None
    pept_missing_thr = None
    lip_sig_num_thr = None
    lip_min_pept_count = None
    uniprot_col = None

    """ LOCATE DATA """
    workdir = f"./data/{experiment_type}/{sample}/{data_type}"
    result_dir = f"./results/{experiment_type}/{sample}"
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    fasta_file = "./data/P1T1/Crona/uniprot-proteome_UP000005640+reviewed_yes.fasta"

    trypsin_pept_file = f"{workdir}/Trypsin only/peptides.txt"
    trypsin_prot_file = f"{workdir}/Trypsin only/proteinGroups.txt"
    double_pept_file = f"{workdir}/Double digest/peptides.txt"
    # double_prot_file = f'{workdir}/Double digest/proteinGroups.txt'

    """ DESCRIBE METADATA """
    ### I don't know what these options should look like
    prot_info_cols = [
        "id",
        "Uniprot",
        "Gene names",
        "Protein names",
        "Treat/Control",
        "Treat/Control_pval",
        "Scalar",
    ]

    info_cols = [
        "id",
        "Sequence",
        "Uniprot",
        "Gene names",
        "Protein names",
        "Amino acid before",
        "Last amino acid",
        "Length",
        "Start position",
        "End position",
        "Missed cleavages",
        "Scalar",
        "Ctrl missingness",
        "Treat missingness",
        "Total missingness",
        "Control",
        "Treat",
        "Treat/Control",
        "Treat/Control_pval",
        "Treat/Control_adj-p",
        "Sig Pept num",
    ]

    # we don't have columns with 'LiP' in header yet, need temporary
    # work around - 'dev' mode
    # check with song about real sol'n
    mode = "dev"
    if (
        data_type.lower() == "lip"
        and lip_search_tool.lower() == "maxquant"
        and mode != "dev"
    ):
        trypsin_ctrl_cols = [
            "LFQ intensity Mock_1 LiP",
            "LFQ intensity Mock_2 LiP",
            "LFQ intensity Mock_3 LiP",
            "LFQ intensity Mock_4 LiP",
            "LFQ intensity Mock_5 LiP",
        ]
        trypsin_treat_cols = [
            "LFQ intensity Infected_1 LiP",
            "LFQ intensity Infected_2 LiP",
            "LFQ intensity Infected_3 LiP",
            "LFQ intensity Infected_4 LiP",
            "LFQ intensity Infected_5 LiP",
        ]
        double_ctrl_cols = [
            "LFQ intensity Mock_1 LiP",
            "LFQ intensity Mock_2 LiP",
            "LFQ intensity Mock_3 LiP",
            "LFQ intensity Mock_4 LiP",
            "LFQ intensity Mock_5 LiP",
        ]
        double_treat_cols = [
            "LFQ intensity Infected_1 LiP",
            "LFQ intensity Infected_2 LiP",
            "LFQ intensity Infected_3 LiP",
            "LFQ intensity Infected_4 LiP",
            "LFQ intensity Infected_5 LiP",
        ]

    else:
        trypsin_ctrl_cols = [
            "LFQ intensity Mock_1",
            "LFQ intensity Mock_2",
            "LFQ intensity Mock_3",
            "LFQ intensity Mock_4",
            "LFQ intensity Mock_5",
        ]
        trypsin_treat_cols = [
            "LFQ intensity Infected_1",
            "LFQ intensity Infected_2",
            "LFQ intensity Infected_3",
            "LFQ intensity Infected_4",
            "LFQ intensity Infected_5",
        ]
        double_ctrl_cols = [
            "LFQ intensity Mock_1",
            "LFQ intensity Mock_2",
            "LFQ intensity Mock_3",
            "LFQ intensity Mock_4",
            "LFQ intensity Mock_5",
        ]
        double_treat_cols = [
            "LFQ intensity Infected_1",
            "LFQ intensity Infected_2",
            "LFQ intensity Infected_3",
            "LFQ intensity Infected_4",
            "LFQ intensity Infected_5",
        ]

    """ FORMAT INPUT DATA """
    columnnames = {
        "double_ctrl_cols": double_ctrl_cols,
        "double_treat_cols": double_treat_cols,
        "double_int_cols": double_ctrl_cols + double_treat_cols,
        "trypsin_ctrl_cols": trypsin_ctrl_cols,
        "trypsin_treat_cols": trypsin_treat_cols,
        "trypsin_int_cols": trypsin_ctrl_cols + trypsin_treat_cols,
        "prot_info_cols": prot_info_cols,
        "info_cols": info_cols,
    }

    filenames = {
        "fasta_file": fasta_file,
        "trypsin_pept_file": trypsin_pept_file,
        "trypsin_prot_file": trypsin_prot_file,
        "double_pept_file": double_pept_file,
    }

    params = {
        "id_separator": id_separator,
        "sig_thr": sig_thr,
        "sig_thr_type": sig_thr_type,
        "lip_sig_num_thr": lip_sig_num_thr,
        "prot_missing_thr": prot_missing_thr,
        "pept_missing_thr": pept_missing_thr,
        "lip_min_pept_count": lip_min_pept_count,
        "experiment_type": experiment_type,
        "sample": sample,
        "data_type": data_type,
        "lip_search_tool": lip_search_tool,
        "uniprot_col": uniprot_col,
    }

    test = Proteome(filenames, params, columnnames)
#    test.filter_data()
#    test.log2_transform_data()
#    test.normalize_data()
#    test.id_data()
#    test.trypsin_prot.to_pickle('trypsin_prot.pkl')
