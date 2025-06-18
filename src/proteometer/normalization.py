from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import pandas as pd

from proteometer.params import Params


def peptide_normalization(
    global_pept: pd.DataFrame,
    mod_pept: pd.DataFrame,
    int_cols: list[str],
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

    return mod_pept


def peptide_batch_correction(
    mod_pept: pd.DataFrame,
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

    if par.batch_correction:
        mod_pept = batch_correction(
            mod_pept,
            metadata,
            par.batch_correct_samples,
            batch_col=par.metadata_batch_col,
            sample_col=par.metadata_sample_col,
        )

    return mod_pept


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
    df_transformed[int_cols] = (
        df_transformed[int_cols].sub(global_medians, axis=1) + global_medians.mean()
    )
    return df_transformed


def median_normalize_columns(
    df: pd.DataFrame,
    cols: list[str],
    skipna: bool = True,
    zero_center: bool = False,
) -> pd.DataFrame:
    """
    Performs median normalization on columns of a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to transform.
        cols (list[str]): List of column names to normalize.
        skipna (bool, optional): Whether to skip NaN values. Defaults to True.
        zero_center (bool, optional): Whether to zero-center the data. Defaults to False.

    Returns:
        pd.DataFrame: Median-normalized DataFrame.
    """
    if skipna:
        df_filtered = df[df[cols].isna().sum(axis=1) == 0].copy()
    else:
        df_filtered = df.copy()

    median_correction_T = cast(
        "pd.Series[float]",
        df_filtered[cols].median(axis=0, skipna=True).fillna(0),
    )
    if not zero_center:
        median_correction_T = median_correction_T.sub(median_correction_T.mean())
    df[cols] = df[cols].sub(median_correction_T, axis=1)

    return df


def median_normalization(
    df: pd.DataFrame,
    int_cols: list[str],
    metadata: pd.DataFrame | None = None,
    batch_correct_samples: Iterable[str] | pd.Series[str] | None = None,
    batch_col: str | None = None,
    sample_col: str = "Sample",
    skipna: bool = True,
    zero_center: bool = False,
) -> pd.DataFrame:
    """
    Performs median normalization on peptide data.

    Args:
        df (pd.DataFrame): DataFrame to transform.
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
    if batch_col is None or metadata is None:
        return median_normalize_columns(df, int_cols, skipna, zero_center)

    if batch_correct_samples is None or len(list(batch_correct_samples)) == 0:
        batch_correct_samples = cast("pd.Series[str]", metadata[sample_col])
    if not set(batch_correct_samples).issubset(cast("pd.Series[str]", metadata[sample_col])):
        batch_correct_samples = cast("pd.Series[str]", metadata[sample_col])
        Warning(
            f"Some samples provided for batch correction are not in metadata, using all samples in {sample_col} of metadata."
        )

    df_transformed = df.copy()
    for batch in metadata[metadata[sample_col].isin(batch_correct_samples)][
        batch_col
    ].unique():
        int_cols_per_batch = cast(
            "pd.Series[str]", metadata[(metadata[batch_col] == batch)][sample_col]
        )
        df_transformed = median_normalize_columns(
            df_transformed,
            int_cols_per_batch.to_list(),
            skipna,
            zero_center,
        )

    return df_transformed


# Batch correction for PTM data
def batch_correction(
    df4batcor: pd.DataFrame,
    metadata: pd.DataFrame,
    batch_correct_samples: Iterable[str] | pd.Series[str] | None = None,
    batch_col: str = "Batch",
    sample_col: str = "Sample",
) -> pd.DataFrame:
    """
    Applies batch correction to peptide data using row-mean centering.

    Args:
        df4batcor (pd.DataFrame): DataFrame to correct.
        metadata (pd.DataFrame): Metadata for batch correction.
        batch_correct_samples (Iterable[str] | pd.Series[str] | None, optional): Samples (column names) to correct. Defaults to None, in which case it is all samples as defined in metadata.
        batch_col (str, optional): Batch column name. Defaults to "Batch".
        sample_col (str, optional): Sample column name. Defaults to "Sample".

    Returns:
        pd.DataFrame: Batch-corrected DataFrame.
    """
    df = df4batcor.copy()
    if batch_correct_samples is None or len(list(batch_correct_samples)) == 0:
        batch_correct_samples = cast("pd.Series[str]", metadata[sample_col])
    if not set(batch_correct_samples).issubset(cast("pd.Series[str]", metadata[sample_col])):
        batch_correct_samples = cast("pd.Series[str]", metadata[sample_col])
        Warning(
            f"Some samples provided for batch correction are not in metadata, using all samples in {sample_col} of metadata."
        )

    batches = cast(
        Iterable[str],
        metadata[metadata[sample_col].isin(batch_correct_samples)][batch_col].unique(),
    )
    batch_means_dict = {}
    for batch in batches:
        df_batch: pd.DataFrame = df[
            metadata[
                (metadata[batch_col] == batch)
                & (metadata[sample_col].isin(batch_correct_samples))
            ][sample_col]
        ].copy()

        row_means_for_batch = cast("pd.Series[float]", df_batch.mean(axis=1))
        batch_means_dict.update({batch: row_means_for_batch})

    batch_means = pd.DataFrame(batch_means_dict)

    batch_means_diffs = batch_means.sub(batch_means.mean(axis=1), axis=0)

    for batch in batches:
        int_cols_per_batch = cast(
            "pd.Series[int]", metadata[(metadata[batch_col] == batch)][sample_col]
        )
        df[int_cols_per_batch] = df[int_cols_per_batch].sub(
            batch_means_diffs[batch], axis=0
        )

    return df
