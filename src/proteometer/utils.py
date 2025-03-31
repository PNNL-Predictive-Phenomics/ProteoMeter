# type: ignore


def generate_index(df, prot_col, level_col=None, id_separator="@", id_col="id"):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    if level_col is None:
        df[id_col] = df[prot_col]
    else:
        df[id_col] = df[prot_col] + id_separator + df[level_col]
    df.index = df[id_col].to_list()
    return df


def check_missingness(df, groups, group_cols):
    """_summary_

    Args:
        df (_type_): _description_
    """
    df["Total missingness"] = 0
    for name, cols in zip(groups, group_cols):
        df[f"{name} missingness"] = df[cols].isna().sum(axis=1)
        df["Total missingness"] = df["Total missingness"] + df[f"{name} missingness"]
    return df


def filter_missingness(df, groups, group_cols, missing_thr=0.0):
    """_summary_

    Args:
        df (_type_): _description_
        groups (_type_): _description_
        group_cols (_type_): _description_
        missing_thr (float, optional): _description_. Defaults to 0.0.

    Returns:
        _type_: _description_
    """
    df = check_missingness(df, groups, group_cols)

    df["missing_check"] = 0
    for name, cols in zip(groups, group_cols):
        df["missing_check"] = df["missing_check"] + (
            df[f"{name} missingness"] > missing_thr * len(cols)
        ).astype(int)
    df_w = df[~(df["missing_check"] > 0)].copy()
    return df_w
