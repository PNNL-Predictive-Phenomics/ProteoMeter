# type: ignore
import re

import pandas as pd

from proteometer.peptide import strip_peptide


def get_res_names(residues):
    res_names = [
        [res for res in re.findall(r"[A-Z]\d+[a-z\-]+", residue)]
        if residue[0] != "P"
        else [residue]
        for residue in residues
    ]
    return res_names


def get_res_pos(residues):
    res_pos = [
        [int(res) for res in re.findall(r"\d+", residue)] if residue[0] != "P" else [0]
        for residue in residues
    ]
    return res_pos


def get_protein_res(proteome, uniprot_id, prot_seqs):
    protein = proteome[proteome["uniprot_id"] == uniprot_id]
    protein.reset_index(drop=True, inplace=True)
    prot_seq_search = [seq for seq in prot_seqs if seq.id == uniprot_id]
    prot_seq = prot_seq_search[0]
    sequence = str(prot_seq.seq)
    clean_pepts = [strip_peptide(pept) for pept in protein["peptide"].to_list()]
    protein["clean_pept"] = clean_pepts
    pept_start = [sequence.find(clean_pept) for clean_pept in clean_pepts]
    pept_end = [
        sequence.find(clean_pept) + len(clean_pept) for clean_pept in clean_pepts
    ]
    protein["pept_start"] = pept_start
    protein["pept_end"] = pept_end
    protein["residue"] = [
        [res + str(sequence.find(clean_pept) + i) for i, res in enumerate(clean_pept)]
        for clean_pept in clean_pepts
    ]
    protein_res = protein.explode("residue")
    protein_res.reset_index(drop=True, inplace=True)
    return protein_res


def count_site_number(df, uniprot_col, site_number_col="site_number"):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    site_number = df.groupby(uniprot_col).size()
    site_number.name = site_number_col
    df = pd.merge(df, site_number, left_on=uniprot_col, right_index=True)
    return df


def count_site_number_with_global_proteomics(
    df, uniprot_col, id_col, site_number_col="site_number"
):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    site_number = df.groupby(uniprot_col).size() - 1
    site_number.name = site_number_col
    for uniprot in site_number.index:
        df.loc[df[id_col] == uniprot, site_number_col] = site_number[uniprot]
    return df
