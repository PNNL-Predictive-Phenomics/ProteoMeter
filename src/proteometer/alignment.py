from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Colormap
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
) -> pd.DataFrame | None:
    """_summary_

    Args:
        pept_df (_type_): _description_
        prot_seq (_type_): _description_
        pept_type (str, optional): _description_. Defaults to "all".
        peptide_col (str, optional): _description_. Defaults to "Sequence".
        max_vis_fc (int, optional): _description_. Defaults to 3.
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
        print(
            f"The {tryptic_pattern} peptide dataframe is empty. Please check the input dataframe."
        )
        return None
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
):
    """_summary_

    Args:
        pept_df (_type_): _description_
        prot_seq (_type_): _description_
        save2file (_type_, optional): _description_. Defaults to None.
        pept_type (str, optional): _description_. Defaults to "all".
        peptide_col (str, optional): _description_. Defaults to "Sequence".
        color_map (str, optional): _description_. Defaults to "coolwarm".
        max_vis_fc (int, optional): _description_. Defaults to 3.
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
    if fc_df is not None:
        plt.figure(
            figsize=(
                min(max(np.floor(seq_len / 3), 5), 10),
                min(max(np.floor(pept_df.shape[0] / 5), 3), 6),
            )
        )
        sns.heatmap(fc_df, center=0, cmap=color_map)
        plt.tight_layout()
        if save2file is not None:
            plt.savefig(f"{save2file}_{tryptic_pattern}_pept_alignments_with_FC.pdf")
            plt.close()
            return None
        else:
            return plt
    else:
        print(
            f"The {tryptic_pattern} peptide dataframe is empty. Please check the input dataframe."
        )
        return None
