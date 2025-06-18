from __future__ import annotations

import pandas as pd

from proteometer.normalization import (
    batch_correction,
    median_normalize_columns,
    tmt_normalization,
)


def test_median_normalize_columns():
    df = pd.DataFrame(
        {"A1": [1, 2, 3], "A2": [4, 5, 6], "A3": [7, 8, 9]}, dtype="float64"
    )
    expected = pd.DataFrame(
        {"A1": [4, 5, 6], "A2": [4, 5, 6], "A3": [4, 5, 6]}, dtype="float64"
    )
    result = median_normalize_columns(df, ["A1", "A2", "A3"])
    pd.testing.assert_frame_equal(result, expected)


def test_median_normalize_columns_with_nan():
    df = pd.DataFrame(
        {
            "A1": [1, 2, 3, 0],
            "A2": [4, 5, 6, float("nan")],
            "A3": [7, 8, 9, 10],
        },
        dtype="float64",
    )
    expected = pd.DataFrame(
        {
            "A1": [4, 5, 6, 3],
            "A2": [4, 5, 6, float("nan")],
            "A3": [4, 5, 6, 7],
        },
        dtype="float64",
    )
    result = median_normalize_columns(df, ["A1", "A2", "A3"])

    print(expected)
    print(result)
    pd.testing.assert_frame_equal(result, expected)


def test_batch_correction():
    df = pd.DataFrame(
        {
            "A1": [1, 1, 4, 4],
            "A2": [1, 3, 2, float("nan")],
            "B1": [7, 8, 9, 10],
        },
        dtype="float64",
    )

    metadata = pd.DataFrame(
        {
            "Sample": ["A1", "A2", "B1"],
            "Batch": ["A", "A", "B"],
        }
    )

    result = batch_correction(df, metadata)
    expected = pd.DataFrame(
        {
            "A1": [4, 4, 7, 7],
            "A2": [4, 6, 5, float("nan")],
            "B1": [4, 5, 6, 7],
        },
        dtype="float64",
    )

    print(expected)
    print(result)

    pd.testing.assert_frame_equal(result, expected)


def test_tmt_normalization():
    df = pd.DataFrame(
        {
            "A1": [10, 20, 30, 0],  # 15
            "A2": [40, 50, 60, 10],  # 45
            "A3": [76, 80, 90, 10],  # 78
        },
        dtype="float64",
    )  # mean of medians is (15 + 45 + 78)/ 3 = 46

    df_global = pd.DataFrame(
        {
            "A1": [1, 2, 3, 0],  # 2
            "A2": [4, 5, 6, float("nan")],  # 5
            "A3": [7, 8, 9, 10],  # 8
        },
        dtype="float64",
    )

    expected = (
        pd.DataFrame(
            {
                "A1": [8, 18, 28, -2],
                "A2": [35, 45, 55, 5],
                "A3": [68, 72, 82, 2],
            },
            dtype="float64",
        )
        + (2 + 5 + 8) / 3
    )
    result = tmt_normalization(df, df_global, ["A1", "A2", "A3"])

    print(expected)
    print(result)
    pd.testing.assert_frame_equal(result, expected)


def test_batch_correction_with_subset_samples():
    # Define a small DataFrame and metadata
    df = pd.DataFrame(
        {
            "A1": [1.0, 3.0],
            "A2": [2.0, 4.0],
            "B1": [5.0, 7.0],
        }
    )
    metadata = pd.DataFrame(
        {
            "Sample": ["A1", "A2", "B1"],
            "Batch": ["X", "X", "Y"],
        }
    )
    # Only correct A1 and B1; A2 is in batch X but not used to compute batch means
    result = batch_correction(
        df,
        metadata,
        batch_correct_samples=["A1", "B1"],
        batch_col="Batch",
        sample_col="Sample",
    )
    # Compute expected by hand:
    # Batch X means from A1 only: [1,3] -> mean per row = [1,3], grand mean = 2 -> diffs = X: [-1,1]?
    # Actually batch_means:
    #  row0: X=1, Y=5 -> mean=3 -> diffs X: -2, Y: +2
    #  row1: X=3, Y=7 -> mean=5 -> diffs X: -2, Y: +2
    # Then subtract batch X diffs from A1 and A2, subtract batch Y diffs from B1
    expected = pd.DataFrame(
        {
            "A1": [1.0 - (-2.0), 3.0 - (-2.0)],
            "A2": [2.0 - (-2.0), 4.0 - (-2.0)],
            "B1": [5.0 - 2.0, 7.0 - 2.0],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_batch_correction_with_no_samples():
    # An empty batch_correct_samples list should default to all samples
    df = pd.DataFrame(
        {
            "S1": [10.0, 20.0],
            "S2": [30.0, 40.0],
        }
    )
    metadata = pd.DataFrame(
        {
            "Sample": ["S1", "S2"],
            "Batch": ["B1", "B2"],
        }
    )
    result = batch_correction(df, metadata, batch_correct_samples=[])
    # For B1 (S1) and B2 (S2), row means are [10,20] and [30,40], grand row-means are [20,30]
    # diffs: B1: [-10,-10], B2: [10,10] → subtract from each
    expected = pd.DataFrame(
        {
            "S1": [10.0 - (-10.0), 20.0 - (-10.0)],
            "S2": [30.0 - 10.0, 40.0 - 10.0],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_batch_correction_with_custom_column_names():
    # Test using non-default batch_col and sample_col names
    df = pd.DataFrame(
        {
            "X1": [2.0, 4.0, 6.0],
            "Y1": [1.0, 3.0, 5.0],
        }
    )
    meta_custom = pd.DataFrame(
        {
            "Smp": ["X1", "Y1"],
            "Bch": ["B1", "B1"],
        }
    )
    # Using the same batch for both samples, batch mean = row mean, so diffs = zero -> no change
    result = batch_correction(
        df,
        meta_custom,
        batch_col="Bch",
        sample_col="Smp",
    )
    pd.testing.assert_frame_equal(result, df)
