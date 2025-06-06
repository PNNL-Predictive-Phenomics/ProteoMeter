from __future__ import annotations

import numpy as np
import pandas as pd

from proteometer.abundance import prot_abund_correction_matched


def test_prot_abund_correction_matched_basic():
    pept = pd.DataFrame(
        {
            "uniprot": ["P1", "P2", "P3", "P1"],
            "peptide": ["ABCD", "EFGH", "IJKLM", "NPQRST"],
            "C_R_1": [1, 2, 3, 4],
            "C_R_2": [2, 3, 4, 5],
            "T_R_1": [10, 20, 30, 40],
            "T_R_2": [11, 21, 31, 41],
        }
    )
    prot = pd.DataFrame(
        {
            "uniprot": ["P1", "P2", "P3"],
            "C_R_1": [1, 2, 3],
            "C_R_2": [2, 3, 4],
            "T_R_1": [10, 20, 30],
            "T_R_2": [11, 21, 31],
        }
    )
    prot = prot.set_index("uniprot", drop=False)
    columns_to_correct = ["C_R_1", "C_R_2", "T_R_1", "T_R_2"]
    # The median for each protein row is used for scaling
    result = prot_abund_correction_matched(pept, prot, columns_to_correct, "uniprot")
    # For each peptide, the corrected value should be:
    # (pept[col] - prot[col]) + median(prot row)
    print(result)
    assert result.iloc[0][columns_to_correct].to_list() == [
        1 - 1 + 6,
        2 - 2 + 6,
        10 - 10 + 6,
        11 - 11 + 6,
    ]
    assert result.iloc[1][columns_to_correct].to_list() == [
        4 - 1 + 6,
        5 - 2 + 6,
        40 - 10 + 6,
        41 - 11 + 6,
    ]
    assert result.iloc[2][columns_to_correct].to_list() == [
        2 - 2 + 23 / 2,
        3 - 3 + 23 / 2,
        20 - 20 + 23 / 2,
        21 - 21 + 23 / 2,
    ]
    assert result.iloc[3][columns_to_correct].to_list() == [
        3 - 3 + 17,
        4 - 4 + 17,
        30 - 30 + 17,
        31 - 31 + 17,
    ]


def test_prot_abund_correction_matched_missing_protein():
    # Peptide with a Uniprot ID not in protein table should remain unchanged
    pept = pd.DataFrame(
        {
            "uniprot": ["P4"],
            "C_R_1": [5],
            "C_R_2": [6],
            "T_R_1": [7],
            "T_R_2": [8],
        }
    )
    prot = pd.DataFrame(
        {
            "uniprot": ["P1", "P2"],
            "C_R_1": [1, 2],
            "C_R_2": [2, 3],
            "T_R_1": [10, 20],
            "T_R_2": [11, 21],
        }
    ).set_index("uniprot", drop=False)
    columns_to_correct = ["C_R_1", "C_R_2", "T_R_1", "T_R_2"]
    result = prot_abund_correction_matched(pept, prot, columns_to_correct, "uniprot")
    # Should be unchanged
    pd.testing.assert_frame_equal(result.reset_index(drop=True), pept)


def test_prot_abund_correction_matched_with_non_tt_cols():
    # Test with non_tt_cols specified (subset of columns)
    pept = pd.DataFrame(
        {
            "uniprot": ["P1"],
            "C_R_1": [1],
            "C_R_2": [2],
            "T_R_1": [3],
            "T_R_2": [4],
        }
    )
    prot = pd.DataFrame(
        {
            "uniprot": ["P1"],
            "C_R_1": [10],
            "C_R_2": [20],
            "T_R_1": [30],
            "T_R_2": [40],
        }
    ).set_index("uniprot", drop=False)
    columns_to_correct = ["C_R_1", "C_R_2", "T_R_1", "T_R_2"]
    non_tt_cols = ["C_R_1", "C_R_2"]
    result = prot_abund_correction_matched(
        pept, prot, columns_to_correct, "uniprot", non_tt_cols=non_tt_cols
    )

    median_val = 15  # over non_tt_cols only

    # petp - prot + median
    expected = [
        1 - 10 + median_val,
        2 - 20 + median_val,
        3 - 30 + median_val,
        4 - 40 + median_val,
    ]
    assert result.iloc[0][columns_to_correct].to_list() == expected


def test_prot_abund_correction_matched_nan_handling():
    # Test with NaN in protein abundance
    pept = pd.DataFrame(
        {
            "uniprot": ["P1"],
            "C_R_1": [1],
            "C_R_2": [2],
            "T_R_1": [3],
            "T_R_2": [4],
        }
    )
    prot = pd.DataFrame(
        {
            "uniprot": ["P1"],
            "C_R_1": [np.nan],
            "C_R_2": [20],
            "T_R_1": [np.nan],
            "T_R_2": [40],
        }
    ).set_index("uniprot", drop=False)
    columns_to_correct = ["C_R_1", "C_R_2", "T_R_1", "T_R_2"]
    result = prot_abund_correction_matched(pept, prot, columns_to_correct, "uniprot")

    # NaNs in prot should be treated as 0 for subtraction, but median ignores NaN
    # petp - prot + median
    median_val = 30
    expected = [
        1 - 0 + median_val * 0,  # no correction because prot is NaN
        2 - 20 + median_val,
        3 - 0 + median_val * 0,  # no correction because prot is NaN
        4 - 40 + median_val,
    ]
    assert result.iloc[0][columns_to_correct].to_list() == expected
