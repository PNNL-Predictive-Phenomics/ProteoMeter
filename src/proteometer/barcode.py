from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from proteometer.lip import select_lytic_sites

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd
    from matplotlib.typing import ColorType


def plot_barcode(
    pal: Collection[ColorType],
    ticklabel: list[str] | None = None,
    barcode_name: str | None = None,
    ax: Axes | None = None,
    size: tuple[int, int] = (10, 2),
) -> Axes:
    """Plot a color-coded barcode.

    Args:
        pal (Collection[ColorType]): A collection of colors for the barcode.
        ticklabel (list[str] | None, optional): Labels for the ticks on the x-axis. Defaults to None.
        barcode_name (str | None, optional): Name label for the barcode on the y-axis. Defaults to None.
        ax (Axes | None, optional): Matplotlib Axes object to draw the barcode on. If None, a new Axes is created. Defaults to None.
        size (tuple[int, int], optional): Size of the figure (width, height). Defaults to (10, 2).

    Returns:
        Axes: The matplotlib Axes object with the plotted barcode.
    """
    n = len(pal)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=size)
    ax.imshow(
        np.arange(n).reshape(1, n),
        cmap=mcolors.ListedColormap(list(pal)),
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_yticks([0])
    if barcode_name:
        ax.set_yticklabels([barcode_name])
    if ticklabel:
        tick_interval = np.ceil(n / len(ticklabel)).astype("int")
        ax.set_xticks(np.arange(0, n, tick_interval))  # type: ignore
        ax.set_xticklabels(ticklabel)
    return ax


def get_barcode(
    fc_bar: pd.DataFrame, color_levels: int = 20, fc_bar_max: float | None = None
) -> list[ColorType]:
    """Get a color-coded barcode for a given DataFrame of fold changes.

    Args:
        fc_bar (pd.DataFrame): DataFrame with columns "FC_DIFF", "FC_TYPE", and "Res".
        color_levels (int, optional): Number of colors in the palette. Defaults to 20.
        fc_bar_max (float | None, optional): Maximum fold change value. Defaults to None.

    Returns:
        list[ColorType]: A list of colors for the barcode.
    """

    both_pal_vals = sns.color_palette("Greens", color_levels)
    up_pal_vals = sns.color_palette("Reds", color_levels)
    down_pal_vals = sns.color_palette("Blues", color_levels)
    insig_pal_vals = sns.color_palette("Greys", color_levels)
    if fc_bar_max is None:
        fc_bar_max = cast("float", fc_bar["FC_DIFF"].abs().max())
    bar_code: list[ColorType] = []
    for i in range(fc_bar.shape[0]):
        if fc_bar.iloc[i, 1] == "both":
            bar_code.append(
                both_pal_vals[
                    int(np.ceil(abs(fc_bar.iloc[i, 0]) / fc_bar_max * color_levels) - 1)  # type: ignore
                ]
            )
        elif fc_bar.iloc[i, 1] == "up":
            bar_code.append(
                up_pal_vals[
                    int(np.ceil(abs(fc_bar.iloc[i, 0]) / fc_bar_max * color_levels) - 1)  # type: ignore
                ]
            )
        elif fc_bar.iloc[i, 1] == "down":
            bar_code.append(
                down_pal_vals[
                    int(np.ceil(abs(fc_bar.iloc[i, 0]) / fc_bar_max * color_levels) - 1)  # type: ignore
                ]
            )
        elif fc_bar.iloc[i, 1] == "insig":
            bar_code.append(
                insig_pal_vals[
                    int(np.ceil(abs(fc_bar.iloc[i, 0]) / fc_bar_max * color_levels) - 1)  # type: ignore
                ]
            )
        else:
            bar_code.append((0, 0, 0))
    return bar_code


# This function is to plot the barcode of a protein with fold changes at single site level
def plot_pept_barcode(
    pept_df: pd.DataFrame,
    pairwise_ttest_name: str,
    sequence: str,
    max_vis_fc: float = 3.0,
    color_levels: int = 20,
    sig_type: str = "adj-p",
    sig_thr: float = 0.05,
    ax: Axes | None = None,
) -> Axes:
    """
    Plot the barcode of a protein with fold changes at tryptic peptide level.

    When peptides overlap, the one with the largest effect size (fold change over significance value) is shown.

    Args:
        pept_df (pd.DataFrame): DataFrame with peptide-level data.
        pairwise_ttest_name (str): Name of the column with pairwise t-test p-values.
        sequence (str): The sequence of the protein.
        output_file_name (str | None, optional): If not None, save the figure to the given file. Defaults to None.

        max_vis_fc (float, optional): The maximum fold change value to visualize. Defaults to 3.0.
        color_levels (int, optional): The number of colors in the palette. Defaults to 20.
        sig_type (str, optional): The type of significance test. Defaults to "pval".
        sig_thr (float, optional): The significance threshold. Defaults to 0.05.
        ax (Axes | None, optional): Matplotlib Axes object to draw the barcode on. If None, a new Axes is created. Defaults to None.

    Returns:
        Axes: The matplotlib Axes object with the plotted barcode.
    """
    seq_len = len(sequence)
    tryptic = pept_df[pept_df["pept_type"] == "Tryptic"].copy()
    semi = pept_df[pept_df["pept_type"] == "Semi-tryptic"].copy()
    protein_id = str(pept_df["Protein"].iloc[0])  # type: ignore
    if not (semi.shape[0] > 0 or tryptic.shape[0] > 0):
        raise ValueError(
            "The peptide dataframe is empty with either tryptic or semi-tryptic peptides. Please check the input dataframe."
        )

    # both_pal_vals = sns.color_palette("Greens", color_levels)
    up_pal_vals = cast("list[ColorType]", sns.color_palette("Reds", color_levels))
    down_pal_vals = cast("list[ColorType]", sns.color_palette("Blues", color_levels))
    insig_pal_vals = cast("list[ColorType]", sns.color_palette("Greys", color_levels))

    fc_diff_names = [aa + str(i + 1) for i, aa in enumerate(list(sequence))]
    fc_diff_max = cast("float", pept_df[pairwise_ttest_name].abs().max())
    tryptic_bar_code: list[ColorType] = [(1.0, 1.0, 1.0)] * len(fc_diff_names)
    tryptic["effect"] = (
        tryptic[pairwise_ttest_name].abs()
        / tryptic[f"{pairwise_ttest_name}_{sig_type}"]
    )

    for _, row in tryptic.sort_values("effect", ascending=True).iterrows():  # type: ignore
        fc = cast("float", row[pairwise_ttest_name])
        if np.isnan(fc):
            continue
        sig = cast("float", row[f"{pairwise_ttest_name}_{sig_type}"])
        start = cast("int", row["pept_start"])
        end = cast("int", row["pept_end"])
        disc = (
            np.ceil(
                min(
                    abs(fc),
                    max_vis_fc + 0.1,
                )
                / fc_diff_max
                * color_levels
            ).astype("int")
            - 1
        )
        if sig < sig_thr and fc > 0:
            for i in range(start - 1, end - 1):
                tryptic_bar_code[i] = up_pal_vals[disc]
        elif sig < sig_thr and fc < 0:
            for i in range(start - 1, end - 1):
                tryptic_bar_code[i] = down_pal_vals[disc]
        else:
            for i in range(start - 1, end - 1):
                tryptic_bar_code[i] = insig_pal_vals[disc]

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 2))
    plot_barcode(
        tryptic_bar_code,
        barcode_name=protein_id,
        ticklabel=[
            fc_diff_names[j]
            for j in cast(
                "Iterable[int]",
                np.arange(0, seq_len, np.ceil(seq_len / 10).astype("int")),
            )
        ],
        ax=ax,
    )
    return ax


# This function is to plot the barcode of a protein with fold changes at lytic site level
def plot_site_barcode(
    site_df: pd.DataFrame,
    pairwise_ttest_name: str,
    sequence: str,
    output_file_name: str | None = None,
    uniprot_id: str = "Protein ID (provided by user)",
    max_vis_fc: float = 3.0,
    color_levels: int = 20,
    site_type_col: str = "Type",
    sig_type: str = "pval",
    sig_thr: float = 0.05,
) -> Figure:
    """
    Plot the barcode of a protein with fold changes at lytic site level.

    Args:
        site_df (pd.DataFrame): DataFrame with site-level data.
        sequence (str): The sequence of the protein.
        pairwise_ttest_name (str): Name of the column with pairwise t-test p-values.
        output_file_name (str | None, optional): If not None, save the figure to the given file. Defaults to None.
        uniprot_id (str, optional): The UniProt ID of the protein. Defaults to "Protein ID (provided by user)".
        max_vis_fc (float, optional): The maximum fold change value to visualize. Defaults to 3.0.
        color_levels (int, optional): The number of colors in the palette. Defaults to 20.
        site_type_col (str, optional): The column name for site type. Defaults to "Lytic site type".
        sig_type (str, optional): The type of significance test. Defaults to "pval".
        sig_thr (float, optional): The significance threshold. Defaults to 0.05.

    Returns:
        Figure: The matplotlib Figure object with the plotted barcode.

    Raises:
        ValueError: If there is no trypsin or prok data.
    """
    seq_len = len(sequence)
    trypsin = select_lytic_sites(site_df, "Tryp", site_type_col)
    prok = select_lytic_sites(site_df, "ProK", site_type_col)

    if not (prok.shape[0] > 0 or trypsin.shape[0] > 0):
        raise ValueError(
            "The digestion site dataframe is empty with either trypsin or prok sites. Please check the input dataframe."
        )

    up_pal_vals = sns.color_palette("Reds", color_levels)
    down_pal_vals = sns.color_palette("Blues", color_levels)
    insig_pal_vals = sns.color_palette("Greys", color_levels)

    fc_diff_names = [aa + str(i + 1) for i, aa in enumerate(list(sequence))]
    fc_diff_max = cast("float", site_df[pairwise_ttest_name].abs().max())
    trypsin_bar_code = [(1.0, 1.0, 1.0)] * len(fc_diff_names)
    prok_bar_code = [(1.0, 1.0, 1.0)] * len(fc_diff_names)
    if trypsin.shape[0] > 0:
        trypsin_fc_diff = trypsin[
            [
                "Site",
                "Pos",
                pairwise_ttest_name,
                f"{pairwise_ttest_name}_{sig_type}",
            ]
        ].copy()
        trypsin_fc_diff.index = trypsin_fc_diff["Site"].to_list()  # type: ignore
        for i in range(trypsin_fc_diff.shape[0]):
            tfd1 = int(cast("int", trypsin_fc_diff.iloc[i, 1]))  # pos
            tfd2 = cast("float", trypsin_fc_diff.iloc[i, 2])  # fc
            tfd3 = cast("float", trypsin_fc_diff.iloc[i, 3])  # (adj) pval
            if np.isnan(tfd2) or np.isnan(tfd3):
                continue
            if tfd2 > 0:
                if tfd3 < sig_thr:
                    trypsin_bar_code[tfd1 - 1] = up_pal_vals[
                        np.ceil(
                            min(abs(tfd2), max_vis_fc + 0.1)
                            / fc_diff_max
                            * color_levels
                        ).astype("int")
                        - 1
                    ]
                else:
                    trypsin_bar_code[tfd1 - 1] = insig_pal_vals[
                        np.ceil(
                            min(
                                abs(tfd2),
                                max_vis_fc + 0.1,
                            )
                            / fc_diff_max
                            * color_levels
                        ).astype("int")
                        - 1
                    ]
            else:
                if tfd3 < sig_thr:
                    trypsin_bar_code[tfd1 - 1] = down_pal_vals[
                        np.ceil(
                            min(
                                abs(tfd2),
                                max_vis_fc + 0.1,
                            )
                            / fc_diff_max
                            * color_levels
                        ).astype("int")
                        - 1
                    ]
                else:
                    trypsin_bar_code[tfd1 - 1] = insig_pal_vals[
                        np.ceil(
                            min(
                                abs(tfd2),
                                max_vis_fc + 0.1,
                            )
                            / fc_diff_max
                            * color_levels
                        ).astype("int")
                        - 1
                    ]
    if prok.shape[0] > 0:
        prok_fc_diff = prok[
            [
                "Site",
                "Pos",
                pairwise_ttest_name,
                f"{pairwise_ttest_name}_{sig_type}",
            ]
        ].copy()
        prok_fc_diff.index = prok_fc_diff["Site"].to_list()  # type: ignore
        for i in range(prok_fc_diff.shape[0]):
            pfd1 = int(cast("int", prok_fc_diff.iloc[i, 1]))
            pfd2 = cast("float", prok_fc_diff.iloc[i, 2])
            pfd3 = cast("float", prok_fc_diff.iloc[i, 3])
            if np.isnan(pfd2) or np.isnan(pfd3):
                continue
            if pfd2 > 0:
                if pfd3 < sig_thr:
                    prok_bar_code[pfd1 - 1] = up_pal_vals[
                        np.ceil(
                            min(abs(pfd2), max_vis_fc + 0.1)
                            / fc_diff_max
                            * color_levels
                        ).astype("int")
                        - 1
                    ]
                else:
                    prok_bar_code[pfd1 - 1] = insig_pal_vals[
                        np.ceil(
                            min(abs(pfd2), max_vis_fc + 0.1)
                            / fc_diff_max
                            * color_levels
                        ).astype("int")
                        - 1
                    ]
            else:
                if pfd3 < sig_thr:
                    prok_bar_code[pfd1 - 1] = down_pal_vals[
                        np.ceil(
                            min(abs(pfd2), max_vis_fc + 0.1)
                            / fc_diff_max
                            * color_levels
                        ).astype("int")
                        - 1
                    ]
                else:
                    prok_bar_code[pfd1 - 1] = insig_pal_vals[
                        np.ceil(
                            min(abs(pfd2), max_vis_fc + 0.1)
                            / fc_diff_max
                            * color_levels
                        ).astype("int")
                        - 1
                    ]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(2, 1, 1)
    plot_barcode(
        trypsin_bar_code,
        barcode_name=uniprot_id + "_trypsin_site",
        ticklabel=[
            fc_diff_names[j]
            for j in cast(
                "Iterable[int]",
                np.arange(0, seq_len, np.ceil(seq_len / 10).astype("int")),
            )
        ],
        ax=ax,
    )
    ax = fig.add_subplot(2, 1, 2)
    plot_barcode(
        prok_bar_code,
        barcode_name=uniprot_id + "_prok_site",
        ticklabel=[
            fc_diff_names[j]
            for j in cast(
                "Iterable[int]",
                np.arange(0, seq_len, np.ceil(seq_len / 10).astype("int")),
            )
        ],
        ax=ax,
    )
    fig.tight_layout()
    if output_file_name is not None:
        fig.savefig(f"{output_file_name}_{uniprot_id}_digestion_site_barcodes.pdf")

    return fig
