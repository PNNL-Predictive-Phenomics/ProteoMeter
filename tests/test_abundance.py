# %%# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from proteometer.abundance import prot_abund_correction_matched


# %%
def test_prot_abund_correction_matched_basic():
    # Create peptide and protein DataFrames with matching Uniprot IDs and columns
    pept = pd.DataFrame({
        'uniprot': ['P1', 'P2', 'P3', 'P1'],
        'peptide': ['ABCD', 'EFGH', 'IJKLM', 'NPQRST'],
        'C_R_1': [1, 2, 3, 4],
        'C_R_2': [2, 3, 4, 5],
        'T_R_1': [10, 20, 30, 40],
        'T_R_2': [11, 21, 31, 41],
    })
    prot = pd.DataFrame({
        'uniprot': ['P1', 'P2', 'P3'],
        'C_R_1': [1, 2, 3],
        'C_R_2': [2, 3, 4],
        'T_R_1': [10, 20, 30],
        'T_R_2': [11, 21, 31],
    }).set_index('uniprot')
    columns_to_correct = ['C_R_1', 'C_R_2', 'T_R_1', 'T_R_2']
    # The median for each protein row is used for scaling
    result = prot_abund_correction_matched(
        pept, prot, columns_to_correct, 'uniprot'
    )
    # For each peptide, the corrected value should be:
    # (pept[col] - prot[col]) + median(prot row)
    for uniprot_id in ['P1', 'P2', 'P3']:
        prot_row = prot.loc[uniprot_id, columns_to_correct]
        median_val = prot_row.median()
        pept_rows = result[result['uniprot'] == uniprot_id]
        for idx, row in pept_rows.iterrows():
            for col in columns_to_correct:
                expected = (pept.loc[idx, col] - prot_row[col]) + median_val
                assert np.isclose(row[col], expected)

def test_prot_abund_correction_matched_missing_protein():
    # Peptide with a Uniprot ID not in protein table should remain unchanged
    pept = pd.DataFrame({
        'uniprot': ['P4'],
        'C_R_1': [5],
        'C_R_2': [6],
        'T_R_1': [7],
        'T_R_2': [8],
    })
    prot = pd.DataFrame({
        'uniprot': ['P1', 'P2'],
        'C_R_1': [1, 2],
        'C_R_2': [2, 3],
        'T_R_1': [10, 20],
        'T_R_2': [11, 21],
    }).set_index('uniprot')
    columns_to_correct = ['C_R_1', 'C_R_2', 'T_R_1', 'T_R_2']
    result = prot_abund_correction_matched(
        pept, prot, columns_to_correct, 'uniprot'
    )
    # Should be unchanged
    pd.testing.assert_frame_equal(result.reset_index(drop=True), pept)

def test_prot_abund_correction_matched_with_non_tt_cols():
    # Test with non_tt_cols specified (subset of columns)
    pept = pd.DataFrame({
        'uniprot': ['P1'],
        'C_R_1': [1],
        'C_R_2': [2],
        'T_R_1': [3],
        'T_R_2': [4],
    })
    prot = pd.DataFrame({
        'uniprot': ['P1'],
        'C_R_1': [10],
        'C_R_2': [20],
        'T_R_1': [30],
        'T_R_2': [40],
    }).set_index('uniprot')
    columns_to_correct = ['C_R_1', 'C_R_2', 'T_R_1', 'T_R_2']
    non_tt_cols = ['C_R_1', 'C_R_2']
    result = prot_abund_correction_matched(
        pept, prot, columns_to_correct, 'uniprot', non_tt_cols=non_tt_cols
    )
    # The median is over non_tt_cols only
    median_val = prot.loc['P1', non_tt_cols].median()
    for col in columns_to_correct:
        expected = (pept.loc[0, col] - prot.loc['P1', col]) + median_val
        assert np.isclose(result.iloc[0][col], expected)

def test_prot_abund_correction_matched_nan_handling():
    # Test with NaN in protein abundance
    pept = pd.DataFrame({
        'uniprot': ['P1'],
        'C_R_1': [1],
        'C_R_2': [2],
        'T_R_1': [3],
        'T_R_2': [4],
    })
    prot = pd.DataFrame({
        'uniprot': ['P1'],
        'C_R_1': [np.nan],
        'C_R_2': [20],
        'T_R_1': [np.nan],
        'T_R_2': [40],
    }).set_index('uniprot')
    columns_to_correct = ['C_R_1', 'C_R_2', 'T_R_1', 'T_R_2']
    result = prot_abund_correction_matched(
        pept, prot, columns_to_correct, 'uniprot'
    )
    # NaNs in prot should be treated as 0 for subtraction, but median ignores NaN
    median_val = prot.loc['P1', columns_to_correct].median()
    for col in columns_to_correct:
        prot_val = prot.loc['P1', col]
        prot_val = 0 if np.isnan(prot_val) else prot_val
        expected = (pept.loc[0, col] - prot_val) + median_val
        assert np.isclose(result.iloc[0][col], expected)

# %%


