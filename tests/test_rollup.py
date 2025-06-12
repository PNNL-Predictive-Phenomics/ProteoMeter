from typing import TypeAlias, Union

import numpy as np
import pandas as pd
import pytest

from proteometer.rollup import rollup_to_site

TestData: TypeAlias = pd.DataFrame
TestDataFixture = Union[TestData]


@pytest.fixture
def simple_df():
    return pd.DataFrame(
        {
            "UniProt": ["P1", "P1", "P2"],
            "Peptide": ["PEPTIDEA", "PEPTIDEB", "PEPTIDEC"],
            "Residue": ["10;20", "10", "30"],
            "Intensity1": [1.0, 2.0, 3.0],
            "Intensity2": [4.0, np.nan, 6.0],
            "Other": ["A", "B", "C"],
        }
    )


def test_rollup_sum(simple_df: TestDataFixture):
    df_out = rollup_to_site(
        df_ori=simple_df,
        int_cols=["Intensity1", "Intensity2"],
        uniprot_col="UniProt",
        peptide_col="Peptide",
        residue_col="Residue",
        residue_sep=";",
        id_col="id",
        id_separator="@",
        site_col="Site",
        multiply_rollup_counts=True,
        ignore_NA=True,
        rollup_func="sum",
    )
    # Check index and columns
    assert "P1@10" in df_out.index
    assert "P1@20" in df_out.index
    assert "P2@30" in df_out.index
    # Check aggregation
    # P1@10: from row 0 (1.0, 4.0) and row 1 (2.0, nan)
    # Intensity1: expsum([1.0, 2.0]) = 1.0 + 2.0 = 3.0
    # Intensity2: expsum([4.0, nan]) = 4.0
    assert np.isclose(df_out.loc["P1@10", "Intensity1"], np.log2(6.0))  # type: ignore
    assert np.isclose(df_out.loc["P1@10", "Intensity2"], 4.0)  # type: ignore
    # P1@20: only row 0
    assert np.isclose(df_out.loc["P1@20", "Intensity1"], 1.0)  # type: ignore
    assert np.isclose(df_out.loc["P1@20", "Intensity2"], 4.0)  # type: ignore
    # P2@30: only row 2
    assert np.isclose(df_out.loc["P2@30", "Intensity1"], 3.0)  # type: ignore
    assert np.isclose(df_out.loc["P2@30", "Intensity2"], 6.0)  # type: ignore


def test_rollup_mean(simple_df: TestDataFixture):
    df_out = rollup_to_site(
        df_ori=simple_df,
        int_cols=["Intensity1"],
        uniprot_col="UniProt",
        peptide_col="Peptide",
        residue_col="Residue",
        rollup_func="mean",
        multiply_rollup_counts=True,
        ignore_NA=True,
    )
    # For P1@10: mean([1.0, 2.0]) = 1.5, log2(2) + 1.5 = 1 + 1.5 = 2.5
    assert np.isclose(df_out.loc["P1@10", "Intensity1"], 2.5)  # type: ignore
    # For P1@20: only one value, log2(1) + 1.0 = 0 + 1.0 = 1.0
    assert np.isclose(df_out.loc["P1@20", "Intensity1"], 1.0)  # type: ignore
    # For P2@30: only one value, log2(1) + 3.0 = 3.0
    assert np.isclose(df_out.loc["P2@30", "Intensity1"], 3.0)  # type: ignore


def test_rollup_median_ignore_na_false(simple_df: TestDataFixture):
    df = simple_df.copy()
    df.loc[1, "Intensity1"] = np.nan
    df_out = rollup_to_site(
        df_ori=df,
        int_cols=["Intensity1"],
        uniprot_col="UniProt",
        peptide_col="Peptide",
        residue_col="Residue",
        rollup_func="median",
        multiply_rollup_counts=True,
        ignore_NA=False,
    )
    # For P1@10: values are 1.0, nan -> median is 1.0, log2(1) + 1.0 = 1.0
    assert np.isclose(df_out.loc["P1@10", "Intensity1"], 1.0)  # type: ignore
    # For P1@20: only 1.0
    assert np.isclose(df_out.loc["P1@20", "Intensity1"], 1.0)  # type: ignore


def test_rollup_to_site_invalid_func(simple_df: TestDataFixture):
    with pytest.raises(ValueError):
        rollup_to_site(
            df_ori=simple_df,
            int_cols=["Intensity1"],
            uniprot_col="UniProt",
            peptide_col="Peptide",
            residue_col="Residue",
            rollup_func="invalid",  # type: ignore
        )


def test_rollup_to_site_no_multiply(simple_df: TestDataFixture):
    df_out = rollup_to_site(
        df_ori=simple_df,
        int_cols=["Intensity1"],
        uniprot_col="UniProt",
        peptide_col="Peptide",
        residue_col="Residue",
        multiply_rollup_counts=False,
        rollup_func="mean",
    )
    # For P1@10: mean([1.0, 2.0]) = 1.5
    assert np.isclose(df_out.loc["P1@10", "Intensity1"], 1.5)  # type: ignore
    # For P1@20: 1.0
    assert np.isclose(df_out.loc["P1@20", "Intensity1"], 1.0)  # type: ignore


def test_rollup_to_site_nan_inf_handling(simple_df: TestDataFixture):
    df = simple_df.copy()
    df.loc[0, "Intensity1"] = np.inf
    df.loc[1, "Intensity1"] = -np.inf
    df_out = rollup_to_site(
        df_ori=df,
        int_cols=["Intensity1"],
        uniprot_col="UniProt",
        peptide_col="Peptide",
        residue_col="Residue",
        rollup_func="sum",
    )
    # Inf and -Inf should be replaced with nan
    assert (
        np.isnan(df_out.loc["P1@10", "Intensity1"])  # type: ignore
        or np.isnan(
            df_out.loc["P1@20", "Intensity1"]  # type: ignore
        )
    )
