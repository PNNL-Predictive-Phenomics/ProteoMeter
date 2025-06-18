import warnings

import numpy as np
import pandas as pd
import pytest

from proteometer.lip import (
    get_clean_peptides,
    get_tryptic_types,
    rollup_single_protein_to_lytic_site,
)
from proteometer.lip_analysis import lip_analysis
from proteometer.params import Params


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


def test_get_clean_peptides_basic():
    # Assume strip_peptide returns the sequence unchanged if no modifications
    df = pd.DataFrame({"Sequence": ["MAAK", "AAK", "AAKT"]})
    result = get_clean_peptides(df, "Sequence")
    assert "clean_pept" in result.columns
    assert list(result["clean_pept"]) == ["MAAK", "AAK", "AAKT"]  # type: ignore


def test_get_clean_peptides_custom_col():
    df = pd.DataFrame({"Seq": ["PEPTIDEA", "PEPTIDEB"]})
    result = get_clean_peptides(df, "Seq", clean_pept_col="stripped")
    assert "stripped" in result.columns
    assert list(result["stripped"]) == ["PEPTIDEA", "PEPTIDEB"]  # type: ignore


def test_get_clean_peptides_empty_df():
    df = pd.DataFrame({"Sequence": []})
    result = get_clean_peptides(df, "Sequence")
    assert result.empty
    assert "clean_pept" in result.columns


def test_get_clean_peptides_with_modifications():
    # Simulate strip_peptide removing modifications
    df = pd.DataFrame({"Sequence": ["M@AAK", "A#AK"]})
    result = get_clean_peptides(df, "Sequence")
    assert list(result["clean_pept"]) == ["MAAK", "AAK"]  # type: ignore


def test_rollup_single_protein_to_lytic_site_basic():
    # Minimal test for rollup_single_protein_to_lytic_site with one peptide
    df = pd.DataFrame(
        {"Sequence": ["MAAK"], "Intensity": [10.0], "UniProt": ["P12345"]}
    )
    result = rollup_single_protein_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id",
    )
    # Should have two lytic sites for one peptide: start and end
    assert result.shape[0] == 2
    assert set(result["Site"]) == {"K0", "K4"}  # type: ignore
    assert all(result["UniProt"] == "P12345")
    assert "Intensity" in result.columns
    assert np.isfinite(result["Intensity"]).all()  # type: ignore
    assert "All pept num" in result.columns
    assert result["All pept num"].iloc[0] == 1


def test_rollup_single_protein_to_lytic_site_multiple_peptides():
    # Test with two peptides, overlapping lytic sites
    df = pd.DataFrame(
        {
            "Sequence": ["MAAK", "AAKT"],
            "Intensity": [10.0, 20.0],
            "UniProt": ["P12345", "P12345"],
        }
    )
    result = rollup_single_protein_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id",
    )
    # Should have 4 lytic sites (2 per peptide, possibly overlapping)
    assert result.shape[0] == 4
    assert set(result["UniProt"]) == {"P12345"}  # type: ignore
    assert set(result["Site"]).issuperset({"K0", "K4", "M1", "T5"})  # type: ignore
    assert "Intensity" in result.columns
    assert np.isfinite(result["Intensity"]).all()  # type: ignore
    assert "All pept num" in result.columns
    assert result["All pept num"].iloc[0] == 2


def test_rollup_single_protein_to_lytic_site_empty():
    # Test with empty dataframe
    df = pd.DataFrame({"Sequence": [], "Intensity": [], "UniProt": []})
    result = rollup_single_protein_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id",
    )
    assert result.empty


def test_rollup_single_protein_to_lytic_site_rollup_func_mean():
    df = pd.DataFrame(
        {
            "Sequence": ["MAAK", "AAK"],
            "Intensity": [10.0, 30.0],
            "UniProt": ["P12345", "P12345"],
        }
    )
    result = rollup_single_protein_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id",
        rollup_func="mean",
    )
    assert "Intensity" in result.columns
    assert np.isfinite(result["Intensity"]).all()  # type: ignore
    assert np.isclose(result[result["Site"] == "K4"]["Intensity"], 21.0)  # type: ignore


def test_rollup_single_protein_to_lytic_site_rollup_func_median():
    # Test rollup_func="median"
    df = pd.DataFrame(
        {
            "Sequence": ["MAAK", "AAK"],
            "Intensity": [10.0, 30.0],
            "UniProt": ["P12345", "P12345"],
        }
    )
    result = rollup_single_protein_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id",
        rollup_func="median",
    )
    assert "Intensity" in result.columns
    assert np.isfinite(result["Intensity"]).all()  # type: ignore
    assert np.isclose(result[result["Site"] == "K4"]["Intensity"], 21.0)  # type: ignore


def test_rollup_single_protein_to_lytic_site_rollup_func_sum():
    df = pd.DataFrame(
        {
            "Sequence": ["MAAK", "AAK"],
            "Intensity": [10.0, 30.0],
            "UniProt": ["P12345", "P12345"],
        }
    )
    result = rollup_single_protein_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id",
        rollup_func="sum",
    )
    expected = np.log2(2**10 + 2**30)
    assert "Intensity" in result.columns
    assert np.isfinite(result["Intensity"]).all()  # type: ignore
    assert np.isclose(result[result["Site"] == "K4"]["Intensity"], expected)  # type: ignore


def test_rollup_single_protein_to_lytic_site_rollup_func_ignore_na():
    # Test rollup_func="sum"
    warnings.filterwarnings(
        "error"
    )  # example constructed to be numerically fine; let's test that
    df = pd.DataFrame(
        {
            "Sequence": ["MAAK", "AAK"],
            "Intensity": [np.nan, 30.0],
            "UniProt": ["P12345", "P12345"],
        }
    )
    result = rollup_single_protein_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id",
        rollup_func="sum",
    )

    expected = np.log2(2**30)
    assert np.isclose(result[result["Site"] == "K4"]["Intensity"], expected)  # type: ignore

    result = rollup_single_protein_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id",
        rollup_func="median",
    )
    expected = 31.0
    assert np.isclose(result[result["Site"] == "K4"]["Intensity"], expected)  # type: ignore

    result = rollup_single_protein_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id",
        rollup_func="mean",
    )
    expected = 31.0
    assert np.isclose(result[result["Site"] == "K4"]["Intensity"], expected)  # type: ignore


def test_rollup_single_protein_to_lytic_site_rollup_func_remove_na():
    # Test rollup_func="sum"
    warnings.filterwarnings(
        "error"
    )  # example constructed to be numerically fine; let's test that
    df = pd.DataFrame(
        {
            "Sequence": ["MAAK", "AAK"],
            "Intensity": [np.nan, 30.0],
            "UniProt": ["P12345", "P12345"],
        }
    )
    result = rollup_single_protein_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id",
        rollup_func="sum",
        ignore_NA=False,
    )
    expected = 30.0
    assert np.isclose(result[result["Site"] == "K4"]["Intensity"], expected)  # type: ignore

    result = rollup_single_protein_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id",
        rollup_func="median",
        ignore_NA=False,
    )
    expected = 30.0
    assert np.isclose(result[result["Site"] == "K4"]["Intensity"], expected)  # type: ignore

    result = rollup_single_protein_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id",
        rollup_func="mean",
        ignore_NA=False,
    )
    expected = 30.0
    assert np.isclose(result[result["Site"] == "K4"]["Intensity"], expected)  # type: ignore


def test_rollup_single_protein_to_lytic_site_alternative_protease():
    # Test with alternative_protease argument
    df = pd.DataFrame(
        {"Sequence": ["MAAK"], "Intensity": [10.0], "UniProt": ["P12345"]}
    )
    result = rollup_single_protein_to_lytic_site(
        df,
        int_cols=["Intensity"],
        uniprot_col="UniProt",
        sequence="MAAKTR",
        peptide_col="Sequence",
        clean_pept_col="clean_pept",
        id_col="id",
        alternative_protease="Chymo",
    )
    assert "Chymo pept num" in result.columns
    assert "Chymo site num" in result.columns
    assert "Tryp site num" in result.columns
