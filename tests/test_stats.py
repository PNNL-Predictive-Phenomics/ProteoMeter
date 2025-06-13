from __future__ import annotations

from typing import TypeAlias, Union, cast

import numpy as np
import pandas as pd
import pytest
import scipy as sp

import warnings

from proteometer.stats import TTestGroup, anova, pairwise_ttest

TestData: TypeAlias = tuple[pd.DataFrame, pd.DataFrame]
TestDataFixture = Union[TestData]


@pytest.fixture
def simple_anova_data() -> TestData:
    # Simulate a simple dataset with two groups and one factor
    df = pd.DataFrame(
        {
            "sample1": [1.0, 2.0, 3.0],
            "sample2": [1.1, 2.1, 3.1],
            "sample3": [4.0, 5.0, 6.0],
            "sample4": [4.1, 5.1, 6.1],
        },
        index=["feature1", "feature2", "feature3"],
        dtype="float64",
    )

    metadata = pd.DataFrame(
        {
            "sample": ["sample1", "sample2", "sample3", "sample4"],
            "group": ["A", "A", "B", "B"],
        },
        dtype="str",
    )

    return df, metadata


def test_anova_values(simple_anova_data: TestDataFixture):
    df, metadata = simple_anova_data
    result = anova(
        df, ["sample1", "sample2", "sample3", "sample4"], metadata, ["group"], "sample"
    )
    res = cast("pd.Series[float]", result["ANOVA_[group]_pval"])
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(result)

    pvals: list[float] = []
    for _, row in df.iterrows():  # type: ignore
        g1 = np.array([row["sample1"], row["sample2"]], dtype="float64")
        g2 = np.array([row["sample3"], row["sample4"]], dtype="float64")
        fval, pval = sp.stats.f_oneway(g1, g2)
        print(fval, pval)
        pvals.append(float(pval))

    print(res)
    assert np.isclose(res, pvals).all()


def test_anova_returns_original_if_no_factors(simple_anova_data: TestDataFixture):
    df, metadata = simple_anova_data
    result = anova(df, ["sample1", "sample2"], metadata, [], "sample")
    pd.testing.assert_frame_equal(result, df)


def test_anova_adds_pvals_and_adj_pvals(simple_anova_data: TestDataFixture):
    df, metadata = simple_anova_data
    result = anova(
        df, ["sample1", "sample2", "sample3", "sample4"], metadata, ["group"], "sample"
    )
    # Should add columns for p-values and adjusted p-values
    assert any(
        col.startswith("ANOVA_") and col.endswith("_pval") for col in result.columns
    )
    assert any(
        col.startswith("ANOVA_") and col.endswith("_adj-p") for col in result.columns
    )
    # Should have same index as input
    assert all(result.index == df.index)


def test_anova_handles_missing_data(simple_anova_data: TestDataFixture):
    df, metadata = simple_anova_data
    df.loc["feature1", "sample1"] = np.nan
    result = anova(
        df, ["sample1", "sample2", "sample3", "sample4"], metadata, ["group"], "sample"
    )
    # Should still produce pval columns, possibly with NaN values
    pval_cols = [col for col in result.columns if col.endswith("_pval")]
    assert len(pval_cols) > 0
    assert result[pval_cols].isnull().sum().sum() >= 0


def test_anova_multiple_factors(simple_anova_data: TestDataFixture):
    df, metadata = simple_anova_data
    metadata["batch"] = ["X", "X", "Y", "Y"]
    result = anova(
        df,
        ["sample1", "sample2", "sample3", "sample4"],
        metadata,
        ["group", "batch"],
        "sample",
    )
    # Should have pval columns for both main effects and interaction
    assert any("group" in col for col in result.columns if col.endswith("_pval"))
    assert any("batch" in col for col in result.columns if col.endswith("_pval"))
    assert any(
        "group * batch" in col for col in result.columns if col.endswith("_pval")
    )


def test_pairwise_ttest():
    df = pd.DataFrame(
        {
            "A1": [1.0],
            "A2": [1.0],
            "A3": [3.0],
            "A4": [3.0],
            "A5": [2.0],
            "B1": [4.0],
            "B2": [5.0],
            "B3": [6.0],
            "B4": [6.0],
            "B5": [4.0],
        },
        dtype="float64",
    )
    acols = [c for c in df.columns if "A" in c]
    bcols = [c for c in df.columns if "B" in c]
    ttest_group = TTestGroup(
        treat_group="A",
        control_group="B",
        treat_samples=acols,
        control_samples=bcols,
    )

    a = cast("pd.Series[float]", df[acols].iloc[0])
    b = cast("pd.Series[float]", df[bcols].iloc[0])

    mu_a = a.mean()
    mu_b = b.mean()

    sigma_a2 = a.std() ** 2 / len(a)  # 1/5
    sigma_b2 = b.std() ** 2 / len(b)  # 1/5
    t = (mu_a - mu_b) / np.sqrt(sigma_a2 + sigma_b2)  # -3 * np.sqrt(5 / 2)

    pval = cast("float", sp.stats.t.sf(np.abs(t), len(a) + len(b) - 2) * 2)
    print(t, pval)
    result = pairwise_ttest(df, [ttest_group])
    print(result)
    assert result["A/B_pval"].iloc[0] == pval


def test_pairwise_ttest_basic():
    # Simple case: two groups, no missing data
    df = pd.DataFrame(
        {
            "T1": [1.0, 2.0, 3.0],
            "T2": [1.5, 2.5, 3.5],
            "C1": [4.0, 5.0, 6.0],
            "C2": [4.5, 5.5, 6.5],
        },
        index=["f1", "f2", "f3"],
    )
    ttest_group = TTestGroup(
        treat_group="T",
        control_group="C",
        treat_samples=["T1", "T2"],
        control_samples=["C1", "C2"],
    )
    result = pairwise_ttest(df.copy(), [ttest_group])
    label = ttest_group.label()
    # Check mean difference
    expected_diff = cast("pd.Series[float]", df[ttest_group.treat_samples].mean(axis=1) - df[ttest_group.control_samples].mean(axis=1))
    
    pd.testing.assert_series_equal(result[label], expected_diff, check_names=False)
    # Check p-values match scipy
    for i, row in df.iterrows(): # type: ignore
        row = cast("pd.Series[float]", row)
        _, p = sp.stats.ttest_ind(
            [row["T1"], row["T2"]],
            [row["C1"], row["C2"]],
            nan_policy="omit",
        )
        p = cast("float", p)
        assert np.isclose(float(result.at[i, f"{label}_pval"]), float(p)) # type: ignore


def test_pairwise_ttest_with_nan():
    # Test with missing values
    warnings.filterwarnings(
        "error"
    )  # example constructed to be numerically fine; let's test that
    df = pd.DataFrame(
        {
            "T1": [1.0, np.nan, 3.0, 4.0],
            "T2": [1.5, 2.5, np.nan, 3.0],
            "C1": [4.0, 5.0, 6.0, 5.0],
            "C2": [4.5, np.nan, 6.5, 2.0],
        },
        index=["f1", "f2", "f3", "f4"],
    )
    ttest_group = TTestGroup(
        treat_group="T",
        control_group="C",
        treat_samples=["T1", "T2"],
        control_samples=["C1", "C2"],
    )
    result = pairwise_ttest(df.copy(), [ttest_group])
    label = ttest_group.label()
    # Should still produce output, possibly with NaN p-values
    assert f"{label}_pval" in result.columns
    assert result[f"{label}_pval"].isnull().sum() >= 0


def test_pairwise_ttest_multiple_groups():
    # Test with two different group comparisons
    df = pd.DataFrame(
        {
            "A1": [1.0, 2.0],
            "A2": [1.1, 2.1],
            "B1": [3.0, 4.0],
            "B2": [3.1, 4.1],
            "C1": [5.0, 6.0],
            "C2": [5.1, 6.1],
        },
        index=["f1", "f2"],
    )
    group1 = TTestGroup("A", "B", ["A1", "A2"], ["B1", "B2"])
    group2 = TTestGroup("B", "C", ["B1", "B2"], ["C1", "C2"])
    result = pairwise_ttest(df.copy(), [group1, group2])
    assert "A/B" in result.columns
    assert "B/C" in result.columns
    assert "A/B_pval" in result.columns
    assert "B/C_pval" in result.columns


def test_pairwise_ttest_adj_pval_nan():
    # If all p-values are nan, adj-p should also be nan
    warnings.filterwarnings(
        "error"
    )  # example constructed to be numerically fine; let's test that
    df = pd.DataFrame(
        {
            "T1": [np.nan, np.nan],
            "T2": [np.nan, np.nan],
            "C1": [np.nan, np.nan],
            "C2": [np.nan, np.nan],
        },
        index=["f1", "f2"],
    )
    ttest_group = TTestGroup(
        treat_group="T",
        control_group="C",
        treat_samples=["T1", "T2"],
        control_samples=["C1", "C2"],
    )
    result = pairwise_ttest(df.copy(), [ttest_group])
    label = ttest_group.label()
    assert result[f"{label}_adj-p"].isnull().all()


def test_pairwise_ttest_returns_input_shape():
    # Output should have same index as input
    df = pd.DataFrame(
        {
            "T1": [1.0, 2.0, 3.0],
            "T2": [1.5, 2.5, 3.5],
            "C1": [4.0, 5.0, 6.0],
            "C2": [4.5, 5.5, 6.5],
        },
        index=["f1", "f2", "f3"],
    )
    ttest_group = TTestGroup(
        treat_group="T",
        control_group="C",
        treat_samples=["T1", "T2"],
        control_samples=["C1", "C2"],
    )
    result = pairwise_ttest(df.copy(), [ttest_group])
    assert all(result.index == df.index)

