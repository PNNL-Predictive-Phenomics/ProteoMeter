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
