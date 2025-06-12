import pandas as pd

from proteometer.utils import filter_missingness


def test_filter_missingness():
    df = pd.DataFrame(
        {
            "A1": [0, float("nan"), 2],
            "A2": [0, 1, float("nan")],
            "B1": [0, 1, 2],
            "B2": [0, 1, float("nan")],
            "C1": [float("nan"), 1, 2],
            "C2": [0, 1, 2],
            "C3": [0, 1, 2],
        }
    )
    groups = ["A", "B", "C"]
    group_cols = [["A1", "A2"], ["B1", "B2"], ["C1", "C2", "C3"]]

    expected_missing_part = pd.DataFrame(
        {
            "Total missingness": [1, 1, 2],
            "A missingness": [0, 1, 1],
            "B missingness": [0, 0, 1],
            "C missingness": [1, 0, 0],
            "missing_check": [0, 1, 2],
        }
    )
    expected_any = pd.concat([df, expected_missing_part], axis=1)
    expected_all = expected_any.loc[
        0:0, :
    ]  # pandas loc indexing is inclusive on both ends!?

    computed_any = filter_missingness(
        df, groups, group_cols, min_replicates_qc=2, method="any"
    )
    computed_all = filter_missingness(
        df, groups, group_cols, min_replicates_qc=2, method="all"
    )
    print()
    print(expected_any)
    print()
    print(computed_any)
    print()
    print(expected_all)
    print()
    print(computed_all)
    print()

    # note: expected == computed fails because of nan
    assert expected_any.equals(computed_any)
    assert expected_all.equals(computed_all)
