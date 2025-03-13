# type: ignore
import numpy as np


# Rolling up the quantification by using log2 medians and plus the log2 of peptide number
def rollup(
    df_ori,
    int_cols,
    rollup_col,
    id_col="id",
    id_separator="@",
    multiply_rollup_counts=True,
    ignore_NA=True,
    rollup_func="median",
):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = df_ori.reset_index(drop=True)
    info_cols = [col for col in df.columns if col not in int_cols]

    df[id_col] = df[id_col] + id_separator + df[rollup_col]

    df[int_cols] = 2 ** df[int_cols]
    df[int_cols] = df[int_cols].fillna(0)

    agg_methods_1 = {i: lambda x: x.iloc[0] for i in info_cols}
    if multiply_rollup_counts:
        if ignore_NA:
            if rollup_func.lower() == "median":
                agg_methods_2 = {
                    i: lambda x: np.log2(len(x)) + x.median() for i in int_cols
                }
            elif rollup_func.lower() == "mean":
                agg_methods_2 = {
                    i: lambda x: np.log2(len(x)) + x.mean() for i in int_cols
                }
            elif rollup_func.lower() == "sum":
                agg_methods_2 = {
                    i: lambda x: np.log2(np.nansum(2 ** (x.replace(0, np.nan))))
                    for i in int_cols
                }
            else:
                ValueError(
                    "The rollup function is not recognized. Please choose from the following: median, mean, sum"
                )
        else:
            if rollup_func.lower() == "median":
                agg_methods_2 = {
                    i: lambda x: np.log2(x.notna().sum()) + x.median() for i in int_cols
                }
            elif rollup_func.lower() == "mean":
                agg_methods_2 = {
                    i: lambda x: np.log2(x.notna().sum()) + x.mean() for i in int_cols
                }
            elif rollup_func.lower() == "sum":
                agg_methods_2 = {
                    i: lambda x: np.log2(np.nansum(2 ** (x.replace(0, np.nan))))
                    for i in int_cols
                }
            else:
                ValueError(
                    "The rollup function is not recognized. Please choose from the following: median, mean, sum"
                )
    else:
        if rollup_func.lower() == "median":
            agg_methods_2 = {i: lambda x: x.median() for i in int_cols}
        elif rollup_func.lower() == "mean":
            agg_methods_2 = {i: lambda x: x.mean() for i in int_cols}
        elif rollup_func.lower() == "sum":
            agg_methods_2 = {
                i: lambda x: np.log2(np.nansum(2 ** (x.replace(0, np.nan))))
                for i in int_cols
            }
        else:
            ValueError(
                "The rollup function is not recognized. Please choose from the following: median, mean, sum"
            )
    df = df.groupby(id_col, as_index=False).agg({**agg_methods_1, **agg_methods_2})
    df[int_cols] = np.log2(df[int_cols])
    df[int_cols] = df[int_cols].replace([np.inf, -np.inf], np.nan)
    df.index = df[id_col].to_list()
    return df


def rollup_to_site(
    df_ori,
    int_cols,
    uniprot_col,
    peptide_col,
    residue_col,
    residue_sep=";",
    id_col="id",
    id_separator="@",
    site_col="Site",
    multiply_rollup_counts=True,
    ignore_NA=True,
    rollup_func="sum",
):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = df_ori.reset_index(drop=True)
    info_cols = [col for col in df.columns if col not in int_cols]

    df[residue_col] = df[residue_col].str.split(residue_sep)
    df = df.explode(residue_col)
    df[id_col] = df[uniprot_col] + id_separator + df[residue_col]

    info_cols_wo_peptide_col = [col for col in info_cols if col != peptide_col]
    agg_methods_0 = {peptide_col: lambda x: "; ".join(x)}
    agg_methods_1 = {i: lambda x: x.iloc[0] for i in info_cols_wo_peptide_col}
    if multiply_rollup_counts:
        if ignore_NA:
            if rollup_func.lower() == "median":
                agg_methods_2 = {
                    i: lambda x: np.log2(len(x)) + x.median() for i in int_cols
                }
            elif rollup_func.lower() == "mean":
                agg_methods_2 = {
                    i: lambda x: np.log2(len(x)) + x.mean() for i in int_cols
                }
            elif rollup_func.lower() == "sum":
                agg_methods_2 = {
                    i: lambda x: np.log2(np.nansum(2 ** (x.replace(0, np.nan))))
                    for i in int_cols
                }
            else:
                ValueError(
                    "The rollup function is not recognized. Please choose from the following: median, mean, sum"
                )
        else:
            if rollup_func.lower() == "median":
                agg_methods_2 = {
                    i: lambda x: np.log2(x.notna().sum()) + x.median() for i in int_cols
                }
            elif rollup_func.lower() == "mean":
                agg_methods_2 = {
                    i: lambda x: np.log2(x.notna().sum()) + x.mean() for i in int_cols
                }
            elif rollup_func.lower() == "sum":
                agg_methods_2 = {
                    i: lambda x: np.log2(np.nansum(2 ** (x.replace(0, np.nan))))
                    for i in int_cols
                }
            else:
                ValueError(
                    "The rollup function is not recognized. Please choose from the following: median, mean, sum"
                )
    else:
        if rollup_func.lower() == "median":
            agg_methods_2 = {i: lambda x: x.median() for i in int_cols}
        elif rollup_func.lower() == "mean":
            agg_methods_2 = {i: lambda x: x.mean() for i in int_cols}
        elif rollup_func.lower() == "sum":
            agg_methods_2 = {
                i: lambda x: np.log2(np.nansum(2 ** (x.replace(0, np.nan))))
                for i in int_cols
            }
        else:
            ValueError(
                "The rollup function is not recognized. Please choose from the following: median, mean, sum"
            )
    df = df.groupby(id_col, as_index=False).agg(
        {**agg_methods_0, **agg_methods_1, **agg_methods_2}
    )
    df[site_col] = df[id_col].to_list()
    df[int_cols] = df[int_cols].replace([np.inf, -np.inf], np.nan)
    df.index = df[id_col].to_list()
    return df


# Rolling up the quantification by summing up the raw intensities and then log2 transformation
def rollup_by_sum(df_ori, int_cols, rollup_col, id_col="id", id_separator="@"):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    raise DeprecationWarning("Deprecated")
    df = df_ori.reset_index(drop=True)
    info_cols = [col for col in df.columns if col not in int_cols]

    df[id_col] = df[id_col] + id_separator + df[rollup_col]

    df[int_cols] = 2 ** df[int_cols]
    df[int_cols] = df[int_cols].fillna(0)

    agg_methods_1 = {i: lambda x: x.iloc[0] for i in info_cols}
    agg_methods_2 = {i: "sum" for i in int_cols}
    df = df.groupby(id_col, as_index=False).agg({**agg_methods_1, **agg_methods_2})
    df[int_cols] = np.log2(df[int_cols])
    df[int_cols] = df[int_cols].replace([np.inf, -np.inf], np.nan)
    df.index = df[id_col].to_list()
    return df


# Rolling up the quantification by using log2 medians and plus the log2 of peptide number
def rollup_by_median(
    df_ori,
    int_cols,
    rollup_col,
    id_col="id",
    id_separator="@",
    multiply_rollup_counts=True,
    ignore_NA=True,
):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    raise DeprecationWarning("Deprecated")
    df = df_ori.reset_index(drop=True)
    info_cols = [col for col in df.columns if col not in int_cols]

    df[id_col] = df[id_col] + id_separator + df[rollup_col]

    df[int_cols] = 2 ** df[int_cols]
    df[int_cols] = df[int_cols].fillna(0)

    agg_methods_1 = {i: lambda x: x.iloc[0] for i in info_cols}
    if multiply_rollup_counts:
        if ignore_NA:
            agg_methods_2 = {
                i: lambda x: np.log2(len(x)) + x.median() for i in int_cols
            }
        else:
            agg_methods_2 = {
                i: lambda x: np.log2(x.notna().sum()) + x.median() for i in int_cols
            }
    else:
        agg_methods_2 = {i: lambda x: x.median() for i in int_cols}
    df = df.groupby(id_col, as_index=False).agg({**agg_methods_1, **agg_methods_2})
    df[int_cols] = np.log2(df[int_cols])
    df[int_cols] = df[int_cols].replace([np.inf, -np.inf], np.nan)
    df.index = df[id_col].to_list()
    return df


# Rolling up the quantification by using log2 medians and plus the log2 of peptide number
def rollup_by_mean(
    df_ori,
    int_cols,
    rollup_col,
    id_col="id",
    id_separator="@",
    multiply_rollup_counts=True,
    ignore_NA=True,
):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    raise DeprecationWarning("Deprecated")
    df = df_ori.reset_index(drop=True)
    info_cols = [col for col in df.columns if col not in int_cols]

    df[id_col] = df[id_col] + id_separator + df[rollup_col]

    df[int_cols] = 2 ** df[int_cols]
    df[int_cols] = df[int_cols].fillna(0)

    agg_methods_1 = {i: lambda x: x.iloc[0] for i in info_cols}
    if multiply_rollup_counts:
        if ignore_NA:
            agg_methods_2 = {i: lambda x: np.log2(len(x)) + x.mean() for i in int_cols}
        else:
            agg_methods_2 = {
                i: lambda x: np.log2(x.notna().sum()) + x.mean() for i in int_cols
            }
    else:
        agg_methods_2 = {i: lambda x: x.mean() for i in int_cols}
    df = df.groupby(id_col, as_index=False).agg({**agg_methods_1, **agg_methods_2})
    df[int_cols] = np.log2(df[int_cols])
    df[int_cols] = df[int_cols].replace([np.inf, -np.inf], np.nan)
    df.index = df[id_col].to_list()
    return df
