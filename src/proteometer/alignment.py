from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.typing import ColorType

from proteometer.lip import select_tryptic_pattern


def get_df_for_pept_alignment_plot(
    pept_df: pd.DataFrame,
    prot_seq: str,
    pairwise_ttest_name: str,
    tryptic_pattern: str = "all",
    peptide_col: str = "Sequence",
    clean_pept_col: str = "clean_pept",
    max_vis_fc: float = 3.0,
    id_separator: str = "@",
) -> pd.DataFrame:
    """
    Generates a DataFrame for visualizing peptide alignment with fold changes.

    Args:
        pept_df (pd.DataFrame): Input DataFrame containing peptide information.
        prot_seq (str): Protein sequence to align peptides against.
        pairwise_ttest_name (str): Column name in `pept_df` containing fold change values.
        tryptic_pattern (str, optional): Tryptic pattern to filter peptides. Defaults to "all".
        peptide_col (str, optional): Column name for peptide sequences in `pept_df`. Defaults to "Sequence".
        clean_pept_col (str, optional): Column name for cleaned peptide sequences in `pept_df`. Defaults to "clean_pept".
        max_vis_fc (float, optional): Maximum fold change value for visualization. Defaults to 3.0.
        id_separator (str, optional): Separator for peptide ID formatting. Defaults to "@".

    Returns:
        pd.DataFrame: A DataFrame with fold changes aligned to the protein sequence.
    """
    seq_len = len(prot_seq)
    protein = select_tryptic_pattern(
        pept_df,
        prot_seq,
        tryptic_pattern=tryptic_pattern,
        peptide_col=peptide_col,
        clean_pept_col=clean_pept_col,
    )
    if protein.shape[0] <= 0:
        raise ValueError(
            f"The {tryptic_pattern} peptide dataframe is empty. Please check the input dataframe."
        )
    else:
        # protein.reset_index(drop=True, inplace=True)
        protein["pept_id"] = [
            str(cast(int, protein["pept_start"].to_list()[i])).zfill(4)
            + "-"
            + str(cast(int, protein["pept_end"].to_list()[i])).zfill(4)
            + id_separator
            + pept
            for i, pept in enumerate(cast("list[str]", protein[peptide_col].to_list()))
        ]
        # protein.index = protein["pept_id"]
        ceiled_fc = [
            max_vis_fc if i > max_vis_fc else -max_vis_fc if i < -max_vis_fc else i
            for i in cast("list[float]", protein[pairwise_ttest_name].to_list())
        ]
        foldchanges = np.zeros((protein.shape[0], seq_len))
        for i in range(len(foldchanges)):
            foldchanges[
                i,
                (protein["pept_start"].to_list()[i] - 1) : (
                    protein["pept_end"].to_list()[i] - 1
                ),
            ] = ceiled_fc[i]
        fc_df = (
            pd.DataFrame(
                foldchanges,
                index=protein["pept_id"],
                columns=[aa + str(i + 1) for i, aa in enumerate(list(prot_seq))],
            )
            .sort_index()
            .replace({0: np.nan})
        )
        return fc_df


# Plot the peptide alignment with the fold changes
def plot_pept_alignment(
    pept_df: pd.DataFrame,
    prot_seq: str,
    pairwise_ttest_name: str,
    save2file: str | None = None,
    tryptic_pattern: str = "all",
    peptide_col: str = "Sequence",
    clean_pept_col: str = "clean_pept",
    max_vis_fc: float = 3.0,
    color_map: str | list[ColorType] | Colormap | None = "coolwarm",
) -> Figure:
    """
    Plots a heatmap of peptide alignment with fold changes.

    Args:
        pept_df (pd.DataFrame): Input DataFrame containing peptide information.
        prot_seq (str): Protein sequence to align peptides against.
        pairwise_ttest_name (str): Column name in `pept_df` containing fold change values.
        save2file (str | None, optional): File path to save the plot. If None, the plot is displayed. Defaults to None.
        tryptic_pattern (str, optional): Tryptic pattern to filter peptides. Defaults to "all".
        peptide_col (str, optional): Column name for peptide sequences in `pept_df`. Defaults to "Sequence".
        clean_pept_col (str, optional): Column name for cleaned peptide sequences in `pept_df`. Defaults to "clean_pept".
        max_vis_fc (float, optional): Maximum fold change value for visualization. Defaults to 3.0.
        color_map (str | list[ColorType] | Colormap | None, optional): Colormap for the heatmap. Defaults to "coolwarm".

    Returns:
        Figure: The plot object.
    """
    seq_len = len(prot_seq)
    fc_df = get_df_for_pept_alignment_plot(
        pept_df,
        prot_seq,
        pairwise_ttest_name,
        tryptic_pattern=tryptic_pattern,
        peptide_col=peptide_col,
        clean_pept_col=clean_pept_col,
        max_vis_fc=max_vis_fc,
    )

    fig = plt.figure(
        figsize=(
            min(max(np.floor(seq_len / 3), 5), 10),
            min(max(np.floor(pept_df.shape[0] / 5), 3), 6),
        )
    )
    sns.heatmap(fc_df, center=0, cmap=color_map)
    fig.tight_layout()
    if save2file is not None:
        fig.savefig(f"{save2file}_{tryptic_pattern}_pept_alignments_with_FC.pdf")

    return fig
