# %%# -*- coding: utf-8 -*-
import warnings

from proteometer.lip_analysis import lip_analysis
from proteometer.params import Params
import pandas as pd
import pytest
from proteometer.lip import get_tryptic_types
from proteometer.lip import get_clean_peptides
import numpy as np
from proteometer.lip import rollup_to_lytic_site

# %%
def test_lip_simple():
    warnings.filterwarnings(
        "error"
    )  # example constructed to be numerically fine; let's test that
    par = Params("tests/data/test_config_lip.toml")
    dfs = lip_analysis(par)
    warnings.resetwarnings()
    for df in dfs:
        assert df is not None
        print(df)
        print(df.columns)

# %%
def test_get_tryptic_types_tryptic():
    # Protein: MAAKTR
    # Peptide: MAK (positions 1-3), semi-tryptic (K at end, but not preceded by K/R)
    prot_seq = "MAAKTR"
    pept_df = pd.DataFrame({"Sequence": ["MAAK"]})
    result = get_tryptic_types(pept_df.copy(), prot_seq, "Sequence")
    assert result.loc[0, "pept_start"] == 1
    assert result.loc[0, "pept_end"] == 4
    # assert result.loc[0, "pept_type"] in ["Tryptic", "Semi-tryptic"]  # N-term is special
    assert result.loc[0, "pept_type"] == "Tryptic"

def test_get_tryptic_types_semi_tryptic():
    # Protein: MAAKTR
    # Peptide: AAK (positions 2-4), tryptic (K at end)
    prot_seq = "MAAKTR"
    pept_df = pd.DataFrame({"Sequence": ["AAK"]})
    result = get_tryptic_types(pept_df.copy(), prot_seq, "Sequence")
    assert result.loc[0, "pept_start"] == 2
    assert result.loc[0, "pept_end"] == 4
    assert result.loc[0, "pept_type"] == "Semi-tryptic"

def test_get_tryptic_types_non_tryptic():
    # Protein: MAAKTR
    # Peptide: AAKT (positions 2-5), not ending or starting at K/R
    prot_seq = "MAAKTR"
    pept_df = pd.DataFrame({"Sequence": ["AAKT"]})
    result = get_tryptic_types(pept_df.copy(), prot_seq, "Sequence")
    assert result.loc[0, "pept_start"] == 2
    assert result.loc[0, "pept_end"] == 5
    assert result.loc[0, "pept_type"] == "Non-tryptic"

def test_get_tryptic_types_not_matched():
    # Peptide not in protein
    prot_seq = "MAAKTR"
    pept_df = pd.DataFrame({"Sequence": ["XYZ"]})
    result = get_tryptic_types(pept_df.copy(), prot_seq, "Sequence")
    assert result.loc[0, "pept_start"] == 0
    assert result.loc[0, "pept_end"] == 2  # 0 + len("XYZ")
    assert result.loc[0, "pept_type"] == "Not-matched"

def test_get_tryptic_types_empty_df():
    prot_seq = "MAAKTR"
    pept_df = pd.DataFrame({"Sequence": []})
    with pytest.raises(ValueError):
        get_tryptic_types(pept_df, prot_seq, "Sequence")

def test_get_tryptic_types_with_clean_pept_col():
    # If clean_pept_col is already present, it should use it
    prot_seq = "MAAKTR"
    pept_df = pd.DataFrame({"Sequence": ["AAK"], "clean_pept": ["AAK"]})
    result = get_tryptic_types(pept_df.copy(), prot_seq, "Sequence", clean_pept_col="clean_pept")
    test_get_tryptic_types_tryptic()
    test_get_tryptic_types_semi_tryptic()
    test_get_tryptic_types_non_tryptic()
    test_get_tryptic_types_not_matched()
    test_get_tryptic_types_empty_df()


# %%
def test_get_clean_peptides_basic():
    # Assume strip_peptide returns the sequence unchanged if no modifications
    df = pd.DataFrame({"Sequence": ["MAAK", "AAK", "AAKT"]})
    result = get_clean_peptides(df.copy(), "Sequence")
    assert "clean_pept" in result.columns
    assert list(result["clean_pept"]) == ["MAAK", "AAK", "AAKT"]

def test_get_clean_peptides_custom_col():
    df = pd.DataFrame({"Seq": ["PEPTIDEA", "PEPTIDEB"]})
    result = get_clean_peptides(df.copy(), "Seq", clean_pept_col="stripped")
    assert "stripped" in result.columns
    assert list(result["stripped"]) == ["PEPTIDEA", "PEPTIDEB"]

def test_get_clean_peptides_empty_df():
    df = pd.DataFrame({"Sequence": []})
    result = get_clean_peptides(df.copy(), "Sequence")
    assert result.empty
    assert "clean_pept" in result.columns

def test_get_clean_peptides_with_modifications(monkeypatch):
    # Simulate strip_peptide removing modifications
    df = pd.DataFrame({"Sequence": ["M[Oxidation]AAK", "A[Phospho]AK"]})
    monkeypatch.setattr("proteometer.lip.strip_peptide", lambda s, nip_off: s.replace("[Oxidation]", "").replace("[Phospho]", ""))
    result = get_clean_peptides(df.copy(), "Sequence")
    assert list(result["clean_pept"]) == ["MAAK", "AAK"]


# %%
def test_rollup_to_lytic_site_basic(monkeypatch):
    # Minimal test for rollup_to_lytic_site with one peptide

    # Patch strip_peptide to identity
    monkeypatch.setattr("proteometer.lip.strip_peptide", lambda s, nip_off: s)

    df = pd.DataFrame({
        "Sequence": ["MAAK"],
        "Intensity": [10.0],
        "UniProt": ["P12345"]
    })
    result = rollup_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id"
    )
    # Should have two lytic sites for one peptide: start and end
    assert result.shape[0] == 2
    assert set(result["Site"]) == {"K0", "K4"}
    assert all(result["UniProt"] == "P12345")
    assert "Intensity" in result.columns
    assert np.isfinite(result["Intensity"]).all()
    assert "All pept num" in result.columns
    assert result["All pept num"].iloc[0] == 1

def test_rollup_to_lytic_site_multiple_peptides(monkeypatch):
    # Test with two peptides, overlapping lytic sites

    monkeypatch.setattr("proteometer.lip.strip_peptide", lambda s, nip_off: s)

    df = pd.DataFrame({
        "Sequence": ["MAAK", "AAKT"],
        "Intensity": [10.0, 20.0],
        "UniProt": ["P12345", "P12345"]
    })
    result = rollup_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id"
    )
    # Should have 4 lytic sites (2 per peptide, possibly overlapping)
    assert result.shape[0] == 4
    assert set(result["UniProt"]) == {"P12345"}
    assert set(result["Site"]).issuperset({"K0", "K4", "M1", "T5"})
    assert "Intensity" in result.columns
    assert np.isfinite(result["Intensity"]).all()
    assert "All pept num" in result.columns
    assert result["All pept num"].iloc[0] == 2

def test_rollup_to_lytic_site_empty(monkeypatch):
    # Test with empty dataframe

    monkeypatch.setattr("proteometer.lip.strip_peptide", lambda s, nip_off: s)
    df = pd.DataFrame({"Sequence": [], "Intensity": [], "UniProt": []})
    result = rollup_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id"
    )
    assert result.empty

def test_rollup_to_lytic_site_rollup_func_mean(monkeypatch):
    # Test rollup_func="mean"

    monkeypatch.setattr("proteometer.lip.strip_peptide", lambda s, nip_off: s)
    df = pd.DataFrame({
        "Sequence": ["MAAK", "AAK"],
        "Intensity": [10.0, 30.0],
        "UniProt": ["P12345", "P12345"]
    })
    result = rollup_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id",
        rollup_func="mean"
    )
    assert "Intensity" in result.columns
    assert np.isfinite(result["Intensity"]).all()
    assert np.isclose(result[result["Site"] == "K4"]["Intensity"], 21.0)

def test_rollup_to_lytic_site_rollup_func_median(monkeypatch):
    # Test rollup_func="median"

    monkeypatch.setattr("proteometer.lip.strip_peptide", lambda s, nip_off: s)
    df = pd.DataFrame({
        "Sequence": ["MAAK", "AAK"],
        "Intensity": [10.0, 30.0],
        "UniProt": ["P12345", "P12345"]
    })
    result = rollup_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id",
        rollup_func="median"
    )
    assert "Intensity" in result.columns
    assert np.isfinite(result["Intensity"]).all()
    assert np.isclose(result[result["Site"] == "K4"]["Intensity"], 21.0)

def test_rollup_to_lytic_site_rollup_func_sum(monkeypatch):
    # Test rollup_func="sum"

    monkeypatch.setattr("proteometer.lip.strip_peptide", lambda s, nip_off: s)
    df = pd.DataFrame({
        "Sequence": ["MAAK", "AAK"],
        "Intensity": [10.0, 30.0],
        "UniProt": ["P12345", "P12345"]
    })
    result = rollup_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id",
        rollup_func="sum"
    )
    assert "Intensity" in result.columns
    assert np.isfinite(result["Intensity"]).all()
    expected = np.log2(2 ** 10 + 2 ** 30)
    assert np.isclose(result[result["Site"] == "K4"]["Intensity"], expected)

def test_rollup_to_lytic_site_rollup_func_ignore_na(monkeypatch):
    # Test rollup_func="sum"
    warnings.filterwarnings(
        "error"
    )  # example constructed to be numerically fine; let's test that
    monkeypatch.setattr("proteometer.lip.strip_peptide", lambda s, nip_off: s)
    df = pd.DataFrame({
        "Sequence": ["MAAK", "AAK"],
        "Intensity": [np.nan, 30.0],
        "UniProt": ["P12345", "P12345"]
    })
    result = rollup_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id",
        rollup_func="sum"
    )

    expected = np.log2(2 ** 30)
    assert np.isclose(result[result["Site"] == "K4"]["Intensity"], expected)
    result = rollup_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id",
        rollup_func="median"
    )
    expected = 31.0
    assert np.isclose(result[result["Site"] == "K4"]["Intensity"], expected)

    result = rollup_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id",
        rollup_func="mean"
    )
    expected = 31.0
    assert np.isclose(result[result["Site"] == "K4"]["Intensity"], expected)

def test_rollup_to_lytic_site_rollup_func_remove_na(monkeypatch):
    # Test rollup_func="sum"
    warnings.filterwarnings(
        "error"
    )  # example constructed to be numerically fine; let's test that
    monkeypatch.setattr("proteometer.lip.strip_peptide", lambda s, nip_off: s)
    df = pd.DataFrame({
        "Sequence": ["MAAK", "AAK"],
        "Intensity": [np.nan, 30.0],
        "UniProt": ["P12345", "P12345"]
    })
    result = rollup_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id",
        rollup_func="sum",
        ignore_NA=False
    )
    expected = 30.0
    assert np.isclose(result[result["Site"] == "K4"]["Intensity"], expected)
    
    result = rollup_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id",
        rollup_func="median",
        ignore_NA=False
    )
    expected = 30.0
    print(result[result["Site"] == "K4"]["Intensity"])
    assert np.isclose(result[result["Site"] == "K4"]["Intensity"], expected)

    result = rollup_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id",
        rollup_func="mean",
        ignore_NA=False
    )
    expected = 30.0
    assert np.isclose(result[result["Site"] == "K4"]["Intensity"], expected)


def test_rollup_to_lytic_site_alternative_protease(monkeypatch):
    # Test with alternative_protease argument

    monkeypatch.setattr("proteometer.lip.strip_peptide", lambda s, nip_off: s)
    df = pd.DataFrame({
        "Sequence": ["MAAK"],
        "Intensity": [10.0],
        "UniProt": ["P12345"]
    })
    result = rollup_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id",
        alternative_protease="Chymo"
    )
    assert "Chymo pept num" in result.columns
    assert "Chymo site num" in result.columns
    assert "Tryp site num" in result.columns


# %%
