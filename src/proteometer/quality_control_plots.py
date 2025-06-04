from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.markers import MarkerStyle
    from numpy import float64


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
    mat_scaled = cast(npt.NDArray[float64], scaler.transform(mat))
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
                marker=cast(MarkerStyle, rf"${i}$"),
                s=100,
                label=f"{df[int_cols].columns[i]}",
            )
            ax.scatter([ptx], [pty], c="k", marker=cast(MarkerStyle, "."), s=10)
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
