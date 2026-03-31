from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.axes import Axes
    from matplotlib.markers import MarkerStyle
    from numpy import float64


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
    sig_mult = log2fc * (significance < sig_thresh)

    cscale = log2fc.abs().max() if max_color_value is None else max_color_value

    ax.scatter(
        log2fc,
        -np.log10(significance),
        c=sig_mult,
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
    use_sample_names: bool = False,
) -> Axes:
    """Plots a biplot of the data.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        int_cols (list[str]): List of columns to plot.
        group_cols (list[list[str]]): List of lists of columns to group by.
        ax (Axes | None, optional): Matplotlib Axes object to draw the biplot on.
            If `None`, a new Axes object is created. Defaults to `None`.
        use_sample_names (bool, optional): If True, uses sample names for annotations.
            Defaults to `False` with index numbers and legend to label points.

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
    x = pca.fit_transform(mat_scaled)
    v1, v2, *_ = pca.explained_variance_ratio_
    score = x[:, 0:2]
    xs = score[:, 0]
    ys = score[:, 1]
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
        if use_sample_names:
            for i in inds:
                ptx, pty = xs[i] * scalex, ys[i] * scaley
                ax.scatter(
                    [ptx],
                    [pty],
                    c=color,
                    marker=cast("MarkerStyle", "."),
                    s=100,
                )
                ax.annotate(
                    f"{int_cols[i]}", (ptx, pty), fontsize=12, ha="center", va="center"
                )
        else:
            for i in inds:
                ptx, pty = xs[i] * scalex, ys[i] * scaley
                ax.scatter(
                    [ptx],
                    [pty],
                    c=color,
                    marker=cast("MarkerStyle", rf"${i}$"),
                    s=100,
                    label=f"{df[int_cols].columns[i]}",
                )
                ax.scatter([ptx], [pty], c="k", marker=cast("MarkerStyle", "."), s=10)
    if not use_sample_names:
        ax.legend(ncols=2, bbox_to_anchor=(1.4, 0.5), loc="center right")
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


def plot_peptide_coverage(
    intensities: Iterable[float],
    pept_start_positions: Iterable[int],
    pept_end_positions: Iterable[int],
    sequence: str,
    linewidth: float = 5,
    n_ticklabels: int | None = 10,
    set_xlim_to_sequence: bool = True,
    zero_center_color: bool = False,
    ax: Axes | None = None,
) -> tuple[Axes, ScalarMappable]:
    """Plots the coverage of peptides over a protein sequence.

    Args:
        intensities (Iterable[float]): An iterable of intensity values corresponding to each peptide. (Determines peptide color)
        pept_start_positions (Iterable[int]): An iterable of start positions of peptides (1-indexed).
        pept_end_positions (Iterable[int]): An iterable of end positions of peptides (1-indexed).
        sequence (str): The full protein sequence.
        linewidth (float, optional): The width of the lines representing peptides. Defaults to 5.
        n_ticklabels (int | None, optional): Number of tick labels to display on the x-axis. If `None`, tick labels are not altered. Defaults to 10.
        set_xlim_to_sequence (bool, optional): Whether to set the x-axis limits to the sequence length. Defaults to True.
            If False, x-axis limits are not altered (useful if you want to overlay this plot on another plot with shared x-axis).
        zero_center_color (bool, optional): Whether to center the color scale at zero and use a diverging colormap. Defaults to False.
        ax (Axes | None, optional): Matplotlib Axes object to draw the peptide coverage plot on.
            If `None`, a new Axes object is created. Defaults to `None`.

    Returns:
        tuple[Axes, ScalarMappable]: A tuple containing the matplotlib Axes object with the plotted peptide coverage
            and a ScalarMappable for the color mapping (e.g., you can create a colorbar via `fig.colorbar(scalar_mappable, ax=ax)`).

    """
    if ax is None:
        _, ax = plt.subplots()

    intensity_array = np.array(intensities)

    depths = np.zeros(len(sequence))
    if zero_center_color:
        cmap = plt.get_cmap("berlin")
        max_abs_intensity = np.nanmax(np.abs(intensity_array))
        pnorm = Normalize(vmin=-max_abs_intensity, vmax=max_abs_intensity)
    else:
        cmap = plt.get_cmap("viridis")
        pnorm = Normalize(
            vmin=np.nanmin(intensity_array), vmax=np.nanmax(intensity_array)
        )
    for start, end, intensity in zip(
        pept_start_positions, pept_end_positions, intensities, strict=True
    ):
        if start < 0 or end > len(sequence):
            raise ValueError(
                f"Peptide positions out of bounds: start={start}, end={end}, sequence length={len(sequence)}"
            )
        if start >= end:
            raise ValueError(
                f"Peptide start position must be less than end position: start={start}, end={end}"
            )
        if np.isnan(intensity):
            continue

        cur_depth = depths[start:end].max()
        depths[start:end] += 1
        ax.hlines(
            cur_depth + 1,
            start - 0.5,
            end + 0.5,
            color=cmap(pnorm(intensity)),
            linewidth=linewidth,
        )

    if n_ticklabels == 0:
        ax.set_xticks([])
    elif n_ticklabels:
        tick_interval = np.ceil(len(sequence) / n_ticklabels).astype("int")
        xtick_positions = list(range(1, len(sequence) + 1, tick_interval))
        xtick_labels = [
            f"{s}{i}" for s, i in zip(sequence[::tick_interval], xtick_positions)
        ]
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, rotation=90)

    if set_xlim_to_sequence:
        ax.set_xlim(1, len(sequence))

    return ax, ScalarMappable(cmap=cmap, norm=pnorm)
