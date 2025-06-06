from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import pandas as pd

from proteometer.params import Params


def peptide_normalization_and_correction(
    global_pept: pd.DataFrame,
    mod_pept: pd.DataFrame,
    int_cols: list[str],
    metadata: pd.DataFrame,
    par: Params,
) -> pd.DataFrame:
    """
    Normalizes and applies batch correction to peptide data.

    Args:
        global_pept (pd.DataFrame): Global peptide data.
        mod_pept (pd.DataFrame): Modified peptide data.
        int_cols (list[str]): List of intensity column names.
        metadata (pd.DataFrame): Metadata for batch correction.
        par (Params): Parameters object containing experiment settings.

    Returns:
        pd.DataFrame: Normalized and batch-corrected peptide data.
    """
    if par.experiment_type == "TMT":
        mod_pept = tmt_normalization(mod_pept, global_pept, int_cols)
    else:
        mod_pept = median_normalization(mod_pept, int_cols)

    if par.batch_correction:
        mod_pept = batch_correction(
            mod_pept,
            metadata,
            par.batch_correct_samples,
            batch_col=par.metadata_batch_col,
            sample_col=par.metadata_sample_col,
        )

    return mod_pept


def tmt_normalization(
    df2transform: pd.DataFrame, global_pept: pd.DataFrame, int_cols: list[str]
) -> pd.DataFrame:
    """
    Performs TMT normalization on peptide data.

    Args:
        df2transform (pd.DataFrame): DataFrame to transform.
        global_pept (pd.DataFrame): Global peptide data for reference.
        int_cols (list[str]): List of intensity column names.

    Returns:
        pd.DataFrame: TMT-normalized DataFrame.
    """
    global_filtered = global_pept[global_pept[int_cols].isna().sum(axis=1) == 0].copy()
    global_medians = cast(
        "pd.Series[float]", global_filtered[int_cols].median(axis=0, skipna=True)
    )
    df_transformed = df2transform.copy()
    df_filtered = df2transform[df2transform[int_cols].isna().sum(axis=1) == 0].copy()
    df_medians = cast(
        "pd.Series[float]",
        (df_filtered[int_cols].median(axis=0, skipna=True).fillna(0)),
    )
    df_transformed[int_cols] = (
        df_transformed[int_cols].sub(global_medians, axis=1) + df_medians.mean()
    )
    return df_transformed


def median_normalization(
    df2transform: pd.DataFrame,
    int_cols: list[str],
    metadata_ori: pd.DataFrame | None = None,
    batch_correct_samples: Iterable[str] | pd.Series[str] | None = None,
    batch_col: str | None = None,
    sample_col: str = "Sample",
    skipna: bool = True,
    zero_center: bool = False,
) -> pd.DataFrame:
    """
    Performs median normalization on peptide data.

    Args:
        df2transform (pd.DataFrame): DataFrame to transform.
        int_cols (list[str]): List of intensity column names.
        metadata_ori (pd.DataFrame | None, optional): Metadata for batch correction. Defaults to None.
        batch_correct_samples (Iterable[str] | pd.Series[str] | None, optional): Samples to correct. Defaults to None.
        batch_col (str | None, optional): Batch column name. Defaults to None.
        sample_col (str, optional): Sample column name. Defaults to "Sample".
        skipna (bool, optional): Whether to skip NaN values. Defaults to True.
        zero_center (bool, optional): Whether to zero-center the data. Defaults to False.

    Returns:
        pd.DataFrame: Median-normalized DataFrame.
    """
    df_transformed = df2transform.copy()
    if batch_col is None or metadata_ori is None:
        if skipna:
            df_filtered = df_transformed[
                df_transformed[int_cols].isna().sum(axis=1) == 0
            ].copy()
        else:
            df_filtered = df_transformed.copy()

        if zero_center:
            median_correction_T = cast(
                "pd.Series[float]",
                df_filtered[int_cols].median(axis=0, skipna=True).fillna(0),
            )
        else:
            median_correction_T = cast(
                "pd.Series[float]",
                df_filtered[int_cols].median(axis=0, skipna=True).fillna(0)
                - df_filtered[int_cols].median(axis=0, skipna=True).fillna(0).mean(),
            )
        df_transformed[int_cols] = df_transformed[int_cols].sub(
            median_correction_T, axis=1
        )

        return df_transformed

    metadata = metadata_ori.copy()
    if batch_correct_samples is None:
        batch_correct_samples = cast("pd.Series[str]", metadata[sample_col])
    for batch in metadata[metadata[sample_col].isin(batch_correct_samples)][
        batch_col
    ].unique():
        int_cols_per_batch = cast(
            pd.Series[int], metadata[(metadata[batch_col] == batch)][sample_col]
        )
        if skipna:
            df_filtered = df_transformed[
                df_transformed[int_cols_per_batch].isna().sum(axis=1) == 0
            ].copy()
        else:
            df_filtered = df_transformed.copy()

        if zero_center:
            median_correction_T = cast(
                "pd.Series[float]",
                df_filtered[int_cols_per_batch].median(axis=0, skipna=True).fillna(0),
            )
        else:
            median_correction_T = cast(
                "pd.Series[float]",
                df_filtered[int_cols_per_batch].median(axis=0, skipna=True).fillna(0)
                - df_filtered[int_cols_per_batch]
                .median(axis=0, skipna=True)
                .fillna(0)
                .mean(),
            )
        df_transformed[int_cols_per_batch] = df_transformed[int_cols_per_batch].sub(
            median_correction_T, axis=1
        )

    return df_transformed


# Batch correction for PTM data
def batch_correction(
    df4batcor: pd.DataFrame,
    metadata_ori: pd.DataFrame,
    batch_correct_samples: Iterable[str] | pd.Series[str] | None = None,
    batch_col: str = "Batch",
    sample_col: str = "Sample",
) -> pd.DataFrame:
    """
    Applies batch correction to peptide data.

    Args:
        df4batcor (pd.DataFrame): DataFrame to correct.
        metadata_ori (pd.DataFrame): Metadata for batch correction.
        batch_correct_samples (Iterable[str] | pd.Series[str] | None, optional): Samples to correct. Defaults to None.
        batch_col (str, optional): Batch column name. Defaults to "Batch".
        sample_col (str, optional): Sample column name. Defaults to "Sample".

    Returns:
        pd.DataFrame: Batch-corrected DataFrame.
    """
    df = df4batcor.copy()
    metadata = metadata_ori.copy()
    if batch_correct_samples is None:
        batch_correct_samples = cast("pd.Series[str]", metadata[sample_col])
    batch_means_dict = {}
    for batch in metadata[metadata[sample_col].isin(batch_correct_samples)][
        batch_col
    ].unique():
        df_batch: pd.DataFrame = df[  # type: ignore
            metadata[
                (metadata[batch_col] == batch)
                & (metadata[sample_col].isin(batch_correct_samples))
            ][sample_col]
        ].copy()
        df_batch_means: pd.DataFrame = df_batch.mean(axis=1)  # type: ignore
        batch_means_dict.update({batch: df_batch_means})
    batch_means = cast("pd.Series[float]", pd.DataFrame(batch_means_dict).mean(axis=1))
    batch_means_diffs = batch_means.sub(batch_means, axis=0)
    metadata.index = metadata[sample_col].to_list()  # type: ignore

    batches = cast(
        Iterable[str],
        metadata[metadata[sample_col].isin(batch_correct_samples)][batch_col].unique(),
    )

    for batch in batches:
        int_cols_per_batch = cast(
            pd.Series[int], metadata[(metadata[batch_col] == batch)][sample_col]
        )
        df[int_cols_per_batch] = df[int_cols_per_batch].sub(
            batch_means_diffs[batch], axis=0
        )

    return df
