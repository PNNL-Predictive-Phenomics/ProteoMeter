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
    """Plot the values in a color palette as a horizontal array.
    Parameters
    ----------
    pal : sequence of matplotlib colors
        colors, i.e. as returned by seaborn.color_palette()
    size :
        figure size of plot
    ax :
        an existing axes to use
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
    # The proper way to set no ticks
    # ax.yaxis.set_major_locator(ticker.NullLocator())
    # ax.set_xticks(np.arange(n) - .5)
    # ax.set_xticks(np.arange(n))
    # Ensure nice border between colors
    # ax.set_xticklabels(["" for _ in range(n)])
    if ticklabel:
        tick_interval = np.ceil(n / len(ticklabel)).astype("int")
        ax.set_xticks(np.arange(0, n, tick_interval))  # type: ignore
        ax.set_xticklabels(ticklabel)
    return ax


def get_barcode(
    fc_bar: pd.DataFrame, color_levels: int = 20, fc_bar_max: float | None = None
) -> list[ColorType]:
    # fc_bar = copy.deepcopy(res_fc_diff[["FC_DIFF", "FC_TYPE", "Res"]])
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
    save2file: str | None = None,
    uniprot_id: str = "Protein ID (provided by user)",
    max_vis_fc: float = 3.0,
    color_levels: int = 20,
    sig_type: str = "pval",
    sig_thr: float = 0.05,
) -> Figure:
    """_summary_

    Args:
        pept_df (_type_): _description_
        sequence (_type_): _description_
        save2file (_type_, optional): _description_. Defaults to None.
        max_vis_fc (int, optional): _description_. Defaults to 3.
        color_levels (int, optional): _description_. Defaults to 20.
    """
    seq_len = len(sequence)
    tryptic = pept_df[pept_df["pept_type"] == "Tryptic"].copy()
    semi = pept_df[pept_df["pept_type"] == "Semi-tryptic"].copy()
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
    semi_bar_code: list[ColorType] = [(1.0, 1.0, 1.0)] * len(fc_diff_names)
    if tryptic.shape[0] > 0:
        tryptic_fc_diff = tryptic[
            [
                "Site",
                "Pos",
                pairwise_ttest_name,
                f"{pairwise_ttest_name}_{sig_type}",
            ]
        ].copy()
        tryptic_fc_diff.index = tryptic_fc_diff["Site"].to_list()  # type: ignore
        for i in range(tryptic_fc_diff.shape[0]):
            tfd1 = cast("int", tryptic_fc_diff.iloc[i, 1])
            tfd2 = cast("float", tryptic_fc_diff.iloc[i, 2])
            tfd3 = cast("float", tryptic_fc_diff.iloc[i, 3])

            if tfd2 > 0:
                if tfd3 < sig_thr:
                    tryptic_bar_code[tfd1 - 1] = cast(
                        "ColorType",
                        up_pal_vals[
                            np.ceil(
                                min(abs(tfd2), max_vis_fc + 0.1)
                                / fc_diff_max
                                * color_levels
                            ).astype("int")
                            - 1
                        ],
                    )
                else:
                    tryptic_bar_code[tfd1 - 1] = insig_pal_vals[
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
                    tryptic_bar_code[tfd1 - 1] = down_pal_vals[
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
                    tryptic_bar_code[tfd1 - 1] = insig_pal_vals[
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
    if semi.shape[0] > 0:
        semi_fc_diff = semi[
            [
                "Site",
                "Pos",
                pairwise_ttest_name,
                f"{pairwise_ttest_name}_{sig_type}",
            ]
        ].copy()
        semi_fc_diff.index = semi_fc_diff["Site"].to_list()  # type: ignore
        for i in range(semi_fc_diff.shape[0]):
            sfd1 = cast("int", semi_fc_diff.iloc[i, 1])
            sfd2 = cast("float", semi_fc_diff.iloc[i, 2])
            sfd3 = cast("float", semi_fc_diff.iloc[i, 3])
            if sfd2 > 0:
                if sfd3 < sig_thr:
                    semi_bar_code[sfd1 - 1] = up_pal_vals[
                        np.ceil(
                            min(abs(sfd2), max_vis_fc + 0.1)
                            / fc_diff_max
                            * color_levels
                        ).astype("int")
                        - 1
                    ]
                else:
                    semi_bar_code[sfd1 - 1] = insig_pal_vals[
                        np.ceil(
                            min(abs(sfd2), max_vis_fc + 0.1)
                            / fc_diff_max
                            * color_levels
                        ).astype("int")
                        - 1
                    ]
            else:
                if sfd3 < sig_thr:
                    semi_bar_code[sfd1 - 1] = down_pal_vals[
                        np.ceil(
                            min(abs(sfd2), max_vis_fc + 0.1)
                            / fc_diff_max
                            * color_levels
                        ).astype("int")
                        - 1
                    ]
                else:
                    semi_bar_code[sfd1 - 1] = insig_pal_vals[
                        np.ceil(
                            min(abs(sfd2), max_vis_fc + 0.1)
                            / fc_diff_max
                            * color_levels
                        ).astype("int")
                        - 1
                    ]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(2, 1, 1)
    plot_barcode(
        tryptic_bar_code,
        barcode_name=uniprot_id + "_tryptic",
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
        semi_bar_code,
        barcode_name=uniprot_id + "_semi-tryptic",
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
    if save2file is not None:
        fig.savefig(f"{save2file}_{uniprot_id}_any_tryptic_barcodes.pdf")

    return fig


# This function is to plot the barcode of a protein with fold changes at lytic site level
def plot_site_barcode(
    site_df: pd.DataFrame,
    sequence: str,
    pairwise_ttest_name: str,
    save2file: str | None = None,
    uniprot_id: str = "Protein ID (provided by user)",
    max_vis_fc: float = 3.0,
    color_levels: int = 20,
    site_type_col: str = "Lytic site type",
    sig_type: str = "pval",
    sig_thr: float = 0.05,
) -> Figure:
    seq_len = len(sequence)
    trypsin = select_lytic_sites(site_df, "trypsin", site_type_col)
    prok = select_lytic_sites(site_df, "prok", site_type_col)
    if not (prok.shape[0] > 0 or trypsin.shape[0] > 0):
        raise ValueError(
            "The digestion site dataframe is empty with either trypsin or prok sites. Please check the input dataframe."
        )

    # both_pal_vals = sns.color_palette("Greens", color_levels)
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
            tfd1 = cast("int", trypsin_fc_diff.iloc[i, 1])
            tfd2 = cast("float", trypsin_fc_diff.iloc[i, 2])
            tfd3 = cast("float", trypsin_fc_diff.iloc[i, 3])
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
            pfd1 = cast("int", prok_fc_diff.iloc[i, 1])
            pfd2 = cast("float", prok_fc_diff.iloc[i, 2])
            pfd3 = cast("float", prok_fc_diff.iloc[i, 3])
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
    if save2file is not None:
        fig.savefig(f"{save2file}_{uniprot_id}_digestion_site_barcodes.pdf")

    return fig
