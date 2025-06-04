from __future__ import annotations

import pandas as pd

from proteometer.normalization import median_normalize_columns


def test_median_normalize_columns():
    df = pd.DataFrame(
        {"A1": [1, 2, 3], "A2": [4, 5, 6], "A3": [7, 8, 9]}, dtype="float64"
    )
    expected = pd.DataFrame(
        {"A1": [4, 5, 6], "A2": [4, 5, 6], "A3": [4, 5, 6]}, dtype="float64"
    )
    result = median_normalize_columns(df, ["A1", "A2", "A3"])
    pd.testing.assert_frame_equal(result, expected)
