# type: ignore
import pandas as pd


def tmt_normalization(df2transform, global_pept, int_cols):
    """_summary_

    Args:
        df2transform (_type_): _description_
    """
    global_filtered = global_pept[global_pept[int_cols].isna().sum(axis=1) == 0].copy()
    global_medians = global_filtered[int_cols].median(axis=0, skipna=True)
    df_transformed = df2transform.copy()
    df_filtered = df2transform[df2transform[int_cols].isna().sum(axis=1) == 0].copy()
    df_medians = df_filtered[int_cols].median(axis=0, skipna=True).fillna(0)
    df_transformed[int_cols] = (
        df_transformed[int_cols].sub(global_medians, axis=1) + df_medians.mean()
    )
    return df_transformed


def median_normalization(
    df2transform,
    int_cols,
    metadata_ori=None,
    batch_correct_samples=None,
    batch_col=None,
    sample_col="Sample",
    skipna=True,
    zero_center=False,
):
    """_summary_

    Args:
        df2transform (_type_): _description_
        int_cols (_type_): _description_

    Returns:
        _type_: _description_
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
            median_correction_T = (
                df_filtered[int_cols].median(axis=0, skipna=True).fillna(0)
            )
        else:
            median_correction_T = (
                df_filtered[int_cols].median(axis=0, skipna=True).fillna(0)
                - df_filtered[int_cols].median(axis=0, skipna=True).fillna(0).mean()
            )
        df_transformed[int_cols] = df_transformed[int_cols].sub(
            median_correction_T, axis=1
        )
        # df_transformed = df_transformed.replace([np.inf, -np.inf], np.nan)
        return df_transformed

    metadata = metadata_ori.copy()
    if batch_correct_samples is None:
        batch_correct_samples = metadata[sample_col].to_list()
    for batch in metadata[metadata[sample_col].isin(batch_correct_samples)][
        batch_col
    ].unique():
        int_cols_per_batch = metadata[(metadata[batch_col] == batch)][sample_col]
        if skipna:
            df_filtered = df_transformed[
                df_transformed[int_cols_per_batch].isna().sum(axis=1) == 0
            ].copy()
        else:
            df_filtered = df_transformed.copy()

        if zero_center:
            median_correction_T = (
                df_filtered[int_cols_per_batch].median(axis=0, skipna=True).fillna(0)
            )
        else:
            median_correction_T = (
                df_filtered[int_cols_per_batch].median(axis=0, skipna=True).fillna(0)
                - df_filtered[int_cols_per_batch]
                .median(axis=0, skipna=True)
                .fillna(0)
                .mean()
            )
        df_transformed[int_cols_per_batch] = df_transformed[int_cols_per_batch].sub(
            median_correction_T, axis=1
        )
    # df_transformed = df_transformed.replace([np.inf, -np.inf], np.nan)
    return df_transformed


# %%
# Batch correction for PTM data
def batch_correction(
    df4batcor,
    metadata_ori,
    batch_correct_samples=None,
    batch_col="Batch",
    sample_col="Sample",
):
    df = df4batcor.copy()
    metadata = metadata_ori.copy()
    if batch_correct_samples is None:
        batch_correct_samples = metadata[sample_col].to_list()
    batch_means = {}
    for batch in metadata[metadata[sample_col].isin(batch_correct_samples)][
        batch_col
    ].unique():
        df_batch = df[
            metadata[
                (metadata[batch_col] == batch)
                & (metadata[sample_col].isin(batch_correct_samples))
            ][sample_col]
        ].copy()
        # df_batch = df_batch[df_batch.isna().sum(axis=1) <= 0].copy()
        df_batch_means = df_batch.mean(axis=1).fillna(0)
        # print(f"Batch {batch} means: {df_batch_means}")
        # print(f"Batch {batch} mean: {df_batch_means.mean()}")
        batch_means.update({batch: df_batch_means})
    batch_means = pd.DataFrame(batch_means)
    batch_means_diffs = batch_means.sub(batch_means.mean(axis=1), axis=0)
    metadata.index = metadata[sample_col].to_list()
    for batch in metadata[metadata[sample_col].isin(batch_correct_samples)][
        batch_col
    ].unique():
        int_cols_per_batch = metadata[(metadata[batch_col] == batch)][sample_col]
        df[int_cols_per_batch] = df[int_cols_per_batch].sub(
            batch_means_diffs[batch], axis=0
        )
    # df = df.replace([np.inf, -np.inf], np.nan)
    return df
