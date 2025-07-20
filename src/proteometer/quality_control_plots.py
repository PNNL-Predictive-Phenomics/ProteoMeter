from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.markers import MarkerStyle
    from numpy import float64


# fig, axs = plt.subplots(1, 3, figsize=(12, 4))
# xscale = lip_prok[comparisons].abs().max(axis=None) * 1.1
# yscale = -np.log10(lip_prok[[c + sig_type for c in comparisons]].min().min()) * 1.1

# for ax, comparison in zip(axs, comparisons):
#     sig_mult = lip_prok[f"{comparison}"] * (
#         lip_prok[f"{comparison}{sig_type}"] < sig_thresh
#     )

#     p = ax.scatter(
#         lip_prok[f"{comparison}"],
#         -np.log10(lip_prok[f"{comparison}{sig_type}"]),
#         c=sig_mult,
#         cmap="coolwarm",
#         vmax=xscale / 2,
#         vmin=-xscale / 2,
#         s=10,
#     )
#     ax.axhline(-np.log10(sig_thresh), color="black", linestyle="--", alpha=0.5)
#     ax.set_xlim(-xscale, xscale)
#     ax.set_ylim(0, yscale)
#     ax.grid()
#     ax.set_xlabel(f"Log2FC {comparison}")
#     ax.set_ylabel("-Log10 adj-p-Value")
# fig.suptitle("ProK Sites", fontsize=16, y=1.01)
# plt.show()


def volcano_plot(
    df: pd.DataFrame,
    comparison: str,
    ax: Axes | None = None,
    sig_type: str = "adj-p",
    sig_thresh: float = 0.1,
    max_color_value: float | None = None,
) -> Axes:
    """Plots a volcano plot of the data.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        comparison (str): The comparison to plot.
        ax (Axes | None, optional): Matplotlib Axes object to draw the volcano plot on.
            If `None`, a new Axes object is created. Defaults to `None`.
        sig_type (str, optional): The type of significance to use. Defaults to "adj-p".
        sig_thresh (float, optional): The significance threshold to use. Defaults to 0.1.
        max_color_value (float | None, optional): Value at which the color scale should stop (symmetrical about zero).
            If None, the maximum absolute value in the data is used.

    Returns:
        Axes: The matplotlib Axes object with the plotted volcano plot.
    """
    if ax is None:
        _, ax = plt.subplots()
    log2fc = cast("pd.Series[float]", df[f"{comparison}"])
    significance = cast("pd.Series[float]", df[f"{comparison}_{sig_type}"])
    sig_mult = log2fc * (significance < sig_thresh)  # type: ignore

    cscale = log2fc.abs().max() if max_color_value is None else max_color_value

    ax.scatter(
        log2fc,
        -np.log10(significance),
        c=sig_mult,  # type: ignore
        cmap="coolwarm",
        vmax=cscale,
        vmin=-cscale,
        s=10,
    )
    xscale = log2fc.abs().max() * 1.1
    yscale = -np.log10(significance.min()) * 1.1

    ax.set_xlim(-xscale, xscale)
    ax.set_ylim(0, yscale)

    ax.axhline(-np.log10(sig_thresh), color="black", linestyle="--", alpha=0.5)
    ax.grid()
    ax.set_xlabel(f"Log2FC {comparison}")
    ax.set_ylabel(f"-Log10 {sig_type}")

    return ax


def biplot(
    df: pd.DataFrame,
    int_cols: list[str],
    group_cols: list[list[str]],
    ax: Axes | None = None,
) -> Axes:
    """Plots a biplot of the data.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        int_cols (list[str]): List of columns to plot.
        group_cols (list[list[str]]): List of lists of columns to group by.
        ax (Axes | None, optional): Matplotlib Axes object to draw the biplot on.
            If `None`, a new Axes object is created. Defaults to `None`.

    Returns:
        Axes: The matplotlib Axes object with the plotted biplot.
    """
    if ax is None:
        _, ax = plt.subplots()
    mat = df[int_cols].T
    scaler = StandardScaler()
    scaler.fit(mat)
    mat_scaled = cast("npt.NDArray[float64]", scaler.transform(mat))
    pca = PCA()
    x = pca.fit_transform(mat_scaled)  # type: ignore
    v1, v2, *_ = pca.explained_variance_ratio_
    score = x[:, 0:2]  # type: ignore
    xs = score[:, 0]  # type: ignore
    ys = score[:, 1]  # type: ignore
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())

    for g_ind, group in enumerate(group_cols):
        inds = [i for i, g in enumerate(df[int_cols].columns) if g in group]
        xvals = score[inds, 0] * scalex
        yvals = score[inds, 1] * scaley
        color = f"C{g_ind}"
        sns.kdeplot(
            x=xvals,
            y=yvals,
            ax=ax,
            color=color,
            fill=True,
            alpha=0.2,
            levels=[0.1, 0.2, 0.5, 1],
        )
        for i in inds:
            ptx, pty = xs[i] * scalex, ys[i] * scaley
            ax.scatter(
                [ptx],
                [pty],
                c=color,
                marker=cast("MarkerStyle", "."),
                s=100,
            )
            ax.annotate(f"{int_cols[i]}",
                        (ptx, pty),
                        fontsize=12,
                        ha="center",
                        va="center")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(f"PC1 ({v1:.2%})")
    ax.set_ylabel(f"PC2 ({v2:.2%})")
    ax.grid()
    return ax


def correlation_plot(
    df: pd.DataFrame, int_cols: list[str], ax: Axes | None = None
) -> Axes:
    """Plots a correlation heatmap of the data.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        int_cols (list[str]): List of columns to plot.
        ax (Axes | None, optional): Matplotlib Axes object to draw the correlation heatmap on.
            If `None`, a new Axes object is created. Defaults to `None`.

    Returns:
        Axes: The matplotlib Axes object with the plotted correlation heatmap.
    """
    if ax is None:
        _, ax = plt.subplots()
    sns.heatmap(
        df[int_cols].corr(), ax=ax, vmin=0.75, vmax=1, cbar_kws={"label": "Correlation"}
    )
    ax.set_xticks([i + 0.5 for i in range(len(int_cols))])
    ax.set_yticks([i + 0.5 for i in range(len(int_cols))])
    ax.set_xticklabels(int_cols, rotation=90)
    ax.set_yticklabels(int_cols, rotation=0)
    return ax
