from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from Bio import SeqIO

from proteometer.abundance import (
    calculate_ibaq,
    count_theoretical_peptides,
    fasta_to_sequence_map,
    prot_abund_correction_matched,
)


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


def test_fasta_to_sequence_map_basic(tmp_path: Path):
    """Test reading a basic FASTA file with multiple entries and complex headers."""
    fasta_content = """>sp|P12345|GENE_NAME_1 Protein description 1
ABCDEFG
HIJKLMN
>sp|P67890|GENE_NAME_2 Protein description 2
OPQRSTU
VWXYZ
"""
    fasta_file = tmp_path / "test.fasta"
    fasta_file.write_text(fasta_content)

    result = fasta_to_sequence_map(str(fasta_file))

    records = list(SeqIO.parse(str(fasta_file), "fasta"))
    expected_from_biopython = {rec.id: str(rec.seq) for rec in records}

    assert result == expected_from_biopython
    assert "sp|P12345|GENE_NAME_1" in result
    assert "sp|P67890|GENE_NAME_2" in result
    assert result["sp|P12345|GENE_NAME_1"] == "ABCDEFGHIJKLMN"


def test_fasta_to_sequence_map_empty_file(tmp_path: Path):
    """Test that an empty FASTA file results in an empty dictionary."""
    fasta_file = tmp_path / "empty.fasta"
    fasta_file.touch()

    result = fasta_to_sequence_map(str(fasta_file))
    assert result == {}


def test_fasta_to_sequence_map_file_not_found():
    """Test that a FileNotFoundError is raised for a non-existent file."""
    with pytest.raises(FileNotFoundError):
        fasta_to_sequence_map("non_existent_file.fasta")


def test_fasta_to_sequence_map_single_entry(tmp_path: Path):
    """Test a FASTA file with only one sequence."""
    fasta_content = """>P99999
SINGLESEQUENCE
"""
    fasta_file = tmp_path / "single.fasta"
    fasta_file.write_text(fasta_content)

    result = fasta_to_sequence_map(str(fasta_file))
    expected = {"P99999": "SINGLESEQUENCE"}
    assert result == expected


@pytest.mark.parametrize(
    "sequence, min_len, max_len, missed_cleavages, expected_count",
    [
        # Basic case with trypsin, 0 missed cleavages
        ("MKLRSG", 6, 30, 0, 0),  # MKLRSG (len 6)
        ("AKBKRCL", 1, 10, 0, 4),  # AK, BK, R, CL
        # Test with missed cleavages
        ("AKBKRCL", 1, 10, 1, 7),  # AK, BK, R, CL, AKBK, KR, RCL
        ("AKBKRCL", 1, 10, 2, 9),  # AK, BK, R, CL, AKBK, KR, RCL, AKBKR, BKRCL
        # Test length filtering
        ("AKBKRCL", 3, 10, 0, 0),  # RCL
        ("AKBKRCL", 1, 2, 0, 4),  # AK, BK, R, CL
        ("AKBKRCL", 8, 10, 2, 0),  # No peptides of this length
        # Test no cleavage sites
        ("ABCDEFGHI", 6, 30, 2, 1),  # The whole sequence
        # Test empty sequence
        ("", 6, 30, 2, 0),
        # Test KP/RP rule (no cleavage)
        ("MKPGRPL", 1, 30, 0, 1),  # MKPGRPL (no cleavage at P)
        # Test sequence ending with cleavage site
        ("ABCDEFK", 1, 30, 0, 1),  # ABCDEFK
        # More complex sequence
        (
            "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK",
            6,
            30,
            2,
            50,
        ),
    ],
)
def test_count_theoretical_peptides(
    sequence,  # type: ignore
    min_len,  # type: ignore
    max_len,  # type: ignore
    missed_cleavages,  # type: ignore
    expected_count,  # type: ignore
):
    """Test the counting of theoretical peptides under various conditions."""
    count = count_theoretical_peptides(
        sequence,  # type: ignore
        min_len=min_len,  # type: ignore
        max_len=max_len,  # type: ignore
        missed_cleavages=missed_cleavages,  # type: ignore
    )
    assert count == expected_count


def test_count_theoretical_peptides_unsupported_protease():
    """Test that an unsupported protease raises a ValueError."""
    with pytest.raises(ValueError, match="Unsupported protease: chymotrypsin"):
        count_theoretical_peptides("ABCDEFG", protease="chymotrypsin")


@pytest.fixture
def sample_data_for_ibaq() -> tuple[pd.DataFrame, dict[str, str]]:
    """Provides sample protein data and sequences for iBAQ tests."""
    prot_df = pd.DataFrame(
        {
            "UniProt": ["P1", "P2", "P3", "P4"],
            "Intensity1": [100.0, 200.0, 300.0, 400.0],
            "Intensity2": [150.0, 250.0, 350.0, 450.0],
        }
    )
    sequences = {
        "sp|P1|PROT1": "MKLRSGABCDEFG",  # Theoretical peptides (6-30): MKLRSGABCDEFG, ... (3)
        "sp|P2|PROT2": "AKBKRCLDEFGHIJKLMN",  # Theoretical peptides (6-30): CLDEFGHIJK, CLDEFGHIJKLMN, ... (5)
        "sp|P3|PROT3": "SHORT",  # No theoretical peptides in range
        # P4 is missing from sequences
    }
    return prot_df, sequences


def test_calculate_ibaq_exact_match(
    sample_data_for_ibaq: tuple[pd.DataFrame, dict[str, str]],
):
    """Test iBAQ calculation with exact protein ID matching."""
    prot_df, sequences = sample_data_for_ibaq
    # P1 has no exact match in sequences, so it should result in NaN
    prot_df_exact = prot_df.copy()
    prot_df_exact["UniProt"] = [
        "sp|P1|PROT1",
        "sp|P2|PROT2",
        "sp|P3|PROT3",
        "P4_nomatch",
    ]

    # Theoretical peptides: P1=1, P2=1, P3=0, P4=0
    # Expected iBAQ:
    # P1: Intensity / 1
    # P2: Intensity / 1
    # P3: Intensity / 0 -> NaN
    # P4: Intensity / 0 -> NaN

    result_df = calculate_ibaq(
        prot_df_exact,
        intensity_cols=["Intensity1", "Intensity2"],
        sequences=sequences,
        prot_id_col="UniProt",
        id_matching="exact",
    )

    assert result_df.loc[0, "Intensity1"] == 100.0 / 3
    assert result_df.loc[0, "Intensity2"] == 150.0 / 3
    assert result_df.loc[1, "Intensity1"] == 200.0 / 5
    assert result_df.loc[1, "Intensity2"] == 250.0 / 5
    assert pd.isna(result_df.loc[2, "Intensity1"])
    assert pd.isna(result_df.loc[3, "Intensity2"])


def test_calculate_ibaq_contain_match(
    sample_data_for_ibaq: tuple[pd.DataFrame, dict[str, str]],
):
    """Test iBAQ calculation with 'contain' protein ID matching."""
    prot_df, sequences = sample_data_for_ibaq

    # Theoretical peptides: P1=1, P2=1, P3=0, P4=0
    # Expected iBAQ:
    # P1: Intensity / 1
    # P2: Intensity / 1
    # P3: Intensity / 0 -> NaN
    # P4: Intensity / 0 -> NaN

    result_df = calculate_ibaq(
        prot_df,
        intensity_cols=["Intensity1", "Intensity2"],
        sequences=sequences,
        prot_id_col="UniProt",
        id_matching="contain",
    )

    assert result_df.loc[0, "Intensity1"] == 100.0 / 3
    assert result_df.loc[0, "Intensity2"] == 150.0 / 3
    assert result_df.loc[1, "Intensity1"] == 200.0 / 5
    assert result_df.loc[1, "Intensity2"] == 250.0 / 5
    assert pd.isna(result_df.loc[2, "Intensity1"])
    assert pd.isna(result_df.loc[3, "Intensity2"])


def test_calculate_ibaq_custom_params(
    sample_data_for_ibaq: tuple[pd.DataFrame, dict[str, str]],
):
    """Test iBAQ calculation with custom peptide length and missed cleavages."""
    prot_df, sequences = sample_data_for_ibaq

    # With min_len=2, max_len=10, missed_cleavages=1:
    # P1 seq: "MKLRSGABCDEFG" -> Peptides: MKL, RSGABCDEFG, MKLRSGABCDEFG. In range: MK, MKLR, LR, RSGABCDEFG (4)
    # P2 seq: "AKBKRCLDEFGHIJKLMN" -> Peptides: AK, BK, R, CLDEFGHI, AKBK, KR, ... In range: AK, BK, AKBK, BKR, LMN, CLDEFGHIJK (6)

    result_df = calculate_ibaq(
        prot_df,
        intensity_cols=["Intensity1"],
        sequences=sequences,
        prot_id_col="UniProt",
        id_matching="contain",
        min_pep_len=2,
        max_pep_len=10,
        missed_cleavages=1,
    )

    assert result_df.loc[0, "Intensity1"] == pytest.approx(100.0 / 4)
    assert result_df.loc[1, "Intensity1"] == pytest.approx(200.0 / 6)
    assert result_df.loc[2, "Intensity1"] == pytest.approx(300.0 / 2)


def test_calculate_ibaq_no_matching_protein(
    sample_data_for_ibaq: tuple[pd.DataFrame, dict[str, str]],
):
    """Test that proteins not in the sequence map result in NaN iBAQ."""
    prot_df, sequences = sample_data_for_ibaq  # type: ignore
    prot_df_no_match = pd.DataFrame({"UniProt": ["P99"], "Intensity1": [1000.0]})

    result_df = calculate_ibaq(
        prot_df_no_match,
        intensity_cols=["Intensity1"],
        sequences=sequences,
        prot_id_col="UniProt",
        id_matching="contain",
    )

    assert pd.isna(result_df.loc[0, "Intensity1"])


def test_calculate_ibaq_unsupported_id_matching():
    """Test that an unsupported id_matching value raises a ValueError."""
    prot_df = pd.DataFrame({"UniProt": ["P1"], "Intensity1": [100.0]})
    sequences = {"P1": "ABCDEFGK"}
    with pytest.raises(ValueError, match="Unsupported id_matching value: partial"):
        calculate_ibaq(
            prot_df,
            intensity_cols=["Intensity1"],
            sequences=sequences,
            id_matching="partial",
        )
