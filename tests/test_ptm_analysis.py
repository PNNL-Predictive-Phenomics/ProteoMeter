import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from proteometer.ptm_analysis import ptm_analysis, ptm_analysis_return_all


class DummyParams:
    def __init__(self):
        self.metadata_file = "dummy_metadata.tsv"
        self.global_prot_file = "dummy_global_prot.tsv"
        self.global_pept_file = "dummy_global_pept.tsv"
        self.ptm_pept_files = ["dummy_ptm_pept1.tsv", "dummy_ptm_pept2.tsv"]
        self.uniprot_col = "Protein"
        self.peptide_col = "Peptide"
        self.residue_col = "Residue"
        self.id_col = "ID"
        self.id_separator = "|"
        self.site_col = "Site"
        self.ptm_names = ["PTM1", "PTM2"]
        self.log2_scale = False
        self.min_replicates_qc = 1
        self.anova_factors = ["Group"]
        self.metadata_sample_col = "Sample"
        self.batch_correction = True
        self.batch_correct_samples = ["Sample1", "Sample2"]
        self.metadata_batch_col = "Batch"
        self.abundance_correction = True
        

tmp_path = Path('data')  # Temporary path for dummy files


@pytest.fixture
def dummy_params(tmp_path: Path):
    # Create dummy files
    metadata = pd.DataFrame({
        "Sample": ["S1", "S2"],
        "Group": ["A", "B"],
        "Batch": ["B1", "B2"]
    })
    global_prot = pd.DataFrame({
        "Protein": ["P1", "P2"],
        "S1": [1.0, 2.0],
        "S2": [3.0, 4.0]
    })
    global_pept = pd.DataFrame({
        "Protein": ["P1", "P2"],
        "Peptide": ["pepa", "pepb"],
        "S1": [1.1, 2.1],
        "S2": [3.1, 4.1]
    })
    ptm_pept1 = pd.DataFrame({
        "Protein": ["P1"],
        "Peptide": ["pepa"],
        "Residue": ["R1"],
        "S1": [1.2],
        "S2": [3.2]
    })
    ptm_pept2 = pd.DataFrame({
        "Protein": ["P2"],
        "Peptide": ["pepb"],
        "Residue": ["R2"],
        "S1": [2.2],
        "S2": [4.2]
    })

    metadata_file = tmp_path / "metadata.tsv"
    global_prot_file = tmp_path / "global_prot.tsv"
    global_pept_file = tmp_path / "global_pept.tsv"
    ptm_pept_file1 = tmp_path / "ptm_pept1.tsv"
    ptm_pept_file2 = tmp_path / "ptm_pept2.tsv"

    metadata.to_csv(metadata_file, sep="\t", index=False)
    global_prot.to_csv(global_prot_file, sep="\t", index=False)
    global_pept.to_csv(global_pept_file, sep="\t", index=False)
    ptm_pept1.to_csv(ptm_pept_file1, sep="\t", index=False)
    ptm_pept2.to_csv(ptm_pept_file2, sep="\t", index=False)

    params = DummyParams()
    params.metadata_file = str(metadata_file)
    params.global_prot_file = str(global_prot_file)
    params.global_pept_file = str(global_pept_file)
    params.ptm_pept_files = [str(ptm_pept_file1), str(ptm_pept_file2)]
    return params

@patch("proteometer.ptm_analysis.parse_metadata")
@patch("proteometer.ptm_analysis.generate_index")
@patch("proteometer.ptm_analysis.stats")
@patch("proteometer.ptm_analysis.filter_missingness")
@patch("proteometer.ptm_analysis.abundance")
@patch("proteometer.ptm_analysis.normalization")
@patch("proteometer.ptm_analysis.rollup")
@patch("proteometer.ptm_analysis.ptm")
def test_ptm_analysis_basic(
    mock_ptm, mock_rollup, mock_normalization, mock_abundance, # type: ignore
    mock_filter_missingness, mock_stats, mock_generate_index, mock_parse_metadata, dummy_params # type: ignore
):
    # Setup parse_metadata mocks
    mock_parse_metadata.int_columns.return_value = ["S1", "S2"]
    mock_parse_metadata.anova_columns.return_value = ["Group"]
    mock_parse_metadata.group_columns.return_value = (["Group"], [["A", "B"]])
    mock_parse_metadata.t_test_groups.return_value = [MagicMock()]

    # Setup generate_index to return the DataFrame (accepts any args)
    mock_generate_index.side_effect = lambda *args, **kwargs: args[0]  # type: ignore

    # Setup normalization.peptide_normalization to just pass through the modified peptide DF
    mock_normalization.peptide_normalization.side_effect = (
        lambda global_pept, mod_pept, *args, **kwargs: mod_pept  # type: ignore
    )

    # Setup stats.log2_transformation to return input DataFrame
    mock_stats.log2_transformation.side_effect = lambda df, cols: df  # type: ignore

    # Setup filter_missingness to return input DataFrame
    mock_filter_missingness.side_effect = lambda df, *args, **kwargs: df  # type: ignore[return-value]

    # Setup abundance.global_prot_normalization_and_stats to return input DataFrame
    mock_abundance.global_prot_normalization_and_stats.side_effect = lambda **kwargs: kwargs["global_prot"]  # type: ignore[return-value]

    # Setup rollup.rollup_to_site to return input DataFrame
    def rollup_to_site_side_effect(*args, **kwargs) -> pd.DataFrame: # type: ignore
        return args[0] # type: ignore
    mock_rollup.rollup_to_site.side_effect = rollup_to_site_side_effect

    # Setup normalization.batch_correction to return input DataFrame
    mock_normalization.batch_correction.side_effect = lambda *args, **kwargs: args[0] # type: ignore

    # Setup abundance.prot_abund_correction to return input DataFrame
    mock_abundance.prot_abund_correction.side_effect = lambda *args, **kwargs: args[0] # type: ignore

    # Setup stats.anova and stats.pairwise_ttest to return input DataFrame
    mock_stats.anova.side_effect = lambda df, *args, **kwargs: df # type: ignore
    mock_stats.pairwise_ttest.side_effect = lambda df, *args, **kwargs: df # type: ignore

    # Setup ptm.combine_multi_ptms to return the first DataFrame in the dict
    mock_ptm.combine_multi_ptms.side_effect = lambda dct, par: list(dct.values())[0] # type: ignore

    all_ptms, global_prot = ptm_analysis(dummy_params) # type: ignore

    assert isinstance(all_ptms, pd.DataFrame)
    assert isinstance(global_prot, pd.DataFrame)

@patch("proteometer.ptm_analysis.parse_metadata")
@patch("proteometer.ptm_analysis.generate_index")
@patch("proteometer.ptm_analysis.stats")
@patch("proteometer.ptm_analysis.filter_missingness")
@patch("proteometer.ptm_analysis.abundance")
@patch("proteometer.ptm_analysis.normalization")
@patch("proteometer.ptm_analysis.rollup")
@patch("proteometer.ptm_analysis.ptm")
def test_ptm_analysis_no_batch_no_abund(
    mock_ptm, mock_rollup, mock_normalization, mock_abundance, # type: ignore
    mock_filter_missingness, mock_stats, mock_generate_index, mock_parse_metadata, dummy_params # type: ignore
):
    # Disable batch and abundance correction
    dummy_params.batch_correction = False
    dummy_params.abundance_correction = False

    # Setup parse_metadata mocks
    mock_parse_metadata.int_columns.return_value = ["S1", "S2"]
    mock_parse_metadata.anova_columns.return_value = ["Group"]
    mock_parse_metadata.group_columns.return_value = (["Group"], [["A", "B"]])
    mock_parse_metadata.t_test_groups.return_value = [MagicMock()]

    mock_generate_index.side_effect = lambda df, *args, **kwargs: df # type: ignore
    mock_normalization.peptide_normalization.side_effect = lambda **kwargs: kwargs["mod_pept"] # type: ignore
    mock_stats.log2_transformation.side_effect = lambda df, cols: df # type: ignore
    mock_filter_missingness.side_effect = lambda df, *args, **kwargs: df # type: ignore
    mock_abundance.global_prot_normalization_and_stats.side_effect = lambda **kwargs: kwargs["global_prot"] # type: ignore
    def rollup_to_site_side_effect(data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame: # type: ignore
        return data
    mock_rollup.rollup_to_site.side_effect = rollup_to_site_side_effect
    mock_stats.anova.side_effect = lambda df, *args, **kwargs: df # type: ignore
    mock_stats.pairwise_ttest.side_effect = lambda df, *args, **kwargs: df # type: ignore
    mock_ptm.combine_multi_ptms.side_effect = lambda dct, par: list(dct.values())[0] # type: ignore

    all_ptms, global_prot = ptm_analysis(dummy_params) # type: ignore

    assert isinstance(all_ptms, pd.DataFrame)
    assert isinstance(global_prot, pd.DataFrame)


@patch("proteometer.ptm_analysis.parse_metadata")
@patch("proteometer.ptm_analysis.generate_index")
@patch("proteometer.ptm_analysis.stats")
@patch("proteometer.ptm_analysis.filter_missingness")
@patch("proteometer.ptm_analysis.abundance")
@patch("proteometer.ptm_analysis.normalization")
@patch("proteometer.ptm_analysis.rollup")
@patch("proteometer.ptm_analysis.ptm")
def test_ptm_analysis_no_abundance_correction(
    mock_ptm, mock_rollup, mock_normalization, mock_abundance, # type: ignore
    mock_filter_missingness, mock_stats, mock_generate_index, mock_parse_metadata, dummy_params # type: ignore
):

    dummy_params.abundance_correction = False

    # Setup parse_metadata mocks
    mock_parse_metadata.int_columns.return_value = ["S1", "S2"]
    mock_parse_metadata.anova_columns.return_value = ["Group"]
    mock_parse_metadata.group_columns.return_value = (["Group"], [["A", "B"]])
    mock_parse_metadata.t_test_groups.return_value = [MagicMock()]

    mock_generate_index.side_effect = lambda df, *args, **kwargs: df # type: ignore
    mock_normalization.peptide_normalization.side_effect = (
        lambda global_pept, mod_pept, *args, **kwargs: mod_pept  # type: ignore
    )
    # stub batch_correction to passthrough a DataFrame
    mock_normalization.batch_correction.side_effect = lambda df, *args, **kwargs: df  # type: ignore
    # stub prot_abund_correction to passthrough a DataFrame
    mock_abundance.prot_abund_correction.side_effect = lambda df, *args, **kwargs: df  # type: ignore

    mock_stats.log2_transformation.side_effect = lambda df, cols: df # type: ignore
    mock_filter_missingness.side_effect = lambda df, *args, **kwargs: df # type: ignore
    mock_abundance.global_prot_normalization_and_stats.side_effect = lambda **kwargs: kwargs["global_prot"] # type: ignore
    mock_rollup.rollup_to_site.side_effect = lambda *args, **kwargs: args[0] # type: ignore
    mock_stats.anova.side_effect = lambda df, *args, **kwargs: df # type: ignore
    mock_stats.pairwise_ttest.side_effect = lambda df, *args, **kwargs: df # type: ignore
    mock_ptm.combine_multi_ptms.side_effect = lambda dct, par: list(dct.values())[0] # type: ignore

    all_ptms, global_prot = ptm_analysis(dummy_params) # type: ignore

    assert isinstance(all_ptms, pd.DataFrame)
    assert isinstance(global_prot, pd.DataFrame)


@patch("proteometer.ptm_analysis.parse_metadata")
@patch("proteometer.ptm_analysis.generate_index")
@patch("proteometer.ptm_analysis.stats")
@patch("proteometer.ptm_analysis.filter_missingness")
@patch("proteometer.ptm_analysis.abundance")
@patch("proteometer.ptm_analysis.normalization")
@patch("proteometer.ptm_analysis.rollup")
@patch("proteometer.ptm_analysis.ptm")
def test_ptm_analysis_return_all_basic(
    mock_ptm, mock_rollup, mock_normalization, mock_abundance, # type: ignore
    mock_filter_missingness, mock_stats, mock_generate_index, mock_parse_metadata, dummy_params # type: ignore
):
    # Setup parse_metadata mocks
    mock_parse_metadata.int_columns.return_value = ["S1", "S2"]
    mock_parse_metadata.anova_columns.return_value = ["Group"]
    mock_parse_metadata.group_columns.return_value = (["Group"], [["A", "B"]])
    mock_parse_metadata.t_test_groups.return_value = [MagicMock()]

    # Setup generate_index to return the DataFrame (accepts any args)
    mock_generate_index.side_effect = lambda *args, **kwargs: args[0]  # type: ignore

    # Setup normalization.peptide_normalization to just pass through the modified peptide DF
    mock_normalization.peptide_normalization.side_effect = (
        lambda global_pept, mod_pept, *args, **kwargs: mod_pept  # type: ignore
    )

    # Setup stats.log2_transformation to return input DataFrame
    mock_stats.log2_transformation.side_effect = lambda df, cols: df  # type: ignore

    # Setup filter_missingness to return input DataFrame
    mock_filter_missingness.side_effect = lambda df, *args, **kwargs: df  # type: ignore[return-value]

    # Setup abundance.global_prot_normalization_and_stats to return input DataFrame
    mock_abundance.global_prot_normalization_and_stats.side_effect = lambda **kwargs: kwargs["global_prot"]  # type: ignore[return-value]

    # Setup rollup.rollup_to_site to return input DataFrame
    def rollup_to_site_side_effect(*args, **kwargs) -> pd.DataFrame: # type: ignore
        return args[0] # type: ignore
    mock_rollup.rollup_to_site.side_effect = rollup_to_site_side_effect

    # Setup normalization.batch_correction to return input DataFrame
    mock_normalization.batch_correction.side_effect = lambda *args, **kwargs: args[0] # type: ignore

    # Setup abundance.prot_abund_correction to return input DataFrame
    mock_abundance.prot_abund_correction.side_effect = lambda *args, **kwargs: args[0] # type: ignore

    # Setup stats.anova and stats.pairwise_ttest to return input DataFrame
    mock_stats.anova.side_effect = lambda df, *args, **kwargs: df # type: ignore
    mock_stats.pairwise_ttest.side_effect = lambda df, *args, **kwargs: df # type: ignore

    # Setup ptm.combine_multi_ptms to return the first DataFrame in the dict
    mock_ptm.combine_multi_ptms.side_effect = lambda dct, par: list(dct.values())[0] # type: ignore

    all_ptms, global_prot, all_ptms_uncorrected = ptm_analysis_return_all(dummy_params) # type: ignore

    assert isinstance(all_ptms, pd.DataFrame)
    assert isinstance(global_prot, pd.DataFrame)
    assert isinstance(all_ptms_uncorrected, pd.DataFrame)

@patch("proteometer.ptm_analysis.parse_metadata")
@patch("proteometer.ptm_analysis.generate_index")
@patch("proteometer.ptm_analysis.stats")
@patch("proteometer.ptm_analysis.filter_missingness")
@patch("proteometer.ptm_analysis.abundance")
@patch("proteometer.ptm_analysis.normalization")
@patch("proteometer.ptm_analysis.rollup")
@patch("proteometer.ptm_analysis.ptm")
def test_ptm_analysis_return_all_no_batch_no_abund(
    mock_ptm, mock_rollup, mock_normalization, mock_abundance, # type: ignore
    mock_filter_missingness, mock_stats, mock_generate_index, mock_parse_metadata, dummy_params # type: ignore
):
    # Disable batch and abundance correction
    dummy_params.batch_correction = False
    dummy_params.abundance_correction = False

    # Setup parse_metadata mocks
    mock_parse_metadata.int_columns.return_value = ["S1", "S2"]
    mock_parse_metadata.anova_columns.return_value = ["Group"]
    mock_parse_metadata.group_columns.return_value = (["Group"], [["A", "B"]])
    mock_parse_metadata.t_test_groups.return_value = [MagicMock()]

    mock_generate_index.side_effect = lambda df, *args, **kwargs: df # type: ignore
    mock_normalization.peptide_normalization.side_effect = (
        lambda global_pept, mod_pept, *args, **kwargs: mod_pept  # type: ignore
    )
    # stub batch_correction (will be skipped when batch=False)
    mock_normalization.batch_correction.side_effect = lambda df, *args, **kwargs: df  # type: ignore
    # stub prot_abund_correction to passthrough a DataFrame
    mock_abundance.prot_abund_correction.side_effect = lambda df, *args, **kwargs: df  # type: ignore

    mock_stats.log2_transformation.side_effect = lambda df, cols: df # type: ignore
    mock_filter_missingness.side_effect = lambda df, *args, **kwargs: df # type: ignore
    mock_abundance.global_prot_normalization_and_stats.side_effect = lambda **kwargs: kwargs["global_prot"] # type: ignore
    def rollup_to_site_side_effect(data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame: # type: ignore
        return data
    mock_rollup.rollup_to_site.side_effect = rollup_to_site_side_effect
    mock_stats.anova.side_effect = lambda df, *args, **kwargs: df # type: ignore
    mock_stats.pairwise_ttest.side_effect = lambda df, *args, **kwargs: df # type: ignore
    mock_ptm.combine_multi_ptms.side_effect = lambda dct, par: list(dct.values())[0] # type: ignore

    all_ptms, global_prot, all_ptms_uncorrected = ptm_analysis_return_all(dummy_params) # type: ignore

    assert isinstance(all_ptms, pd.DataFrame)
    assert isinstance(global_prot, pd.DataFrame)
    assert isinstance(all_ptms_uncorrected, pd.DataFrame)


@patch("proteometer.ptm_analysis.parse_metadata")
@patch("proteometer.ptm_analysis.generate_index")
@patch("proteometer.ptm_analysis.stats")
@patch("proteometer.ptm_analysis.filter_missingness")
@patch("proteometer.ptm_analysis.abundance")
@patch("proteometer.ptm_analysis.normalization")
@patch("proteometer.ptm_analysis.rollup")
@patch("proteometer.ptm_analysis.ptm")
def test_ptm_analysis_return_all_no_abundance_correction(
    mock_ptm, mock_rollup, mock_normalization, mock_abundance, # type: ignore
    mock_filter_missingness, mock_stats, mock_generate_index, mock_parse_metadata, dummy_params # type: ignore
):

    dummy_params.abundance_correction = False

    # Setup parse_metadata mocks
    mock_parse_metadata.int_columns.return_value = ["S1", "S2"]
    mock_parse_metadata.anova_columns.return_value = ["Group"]
    mock_parse_metadata.group_columns.return_value = (["Group"], [["A", "B"]])
    mock_parse_metadata.t_test_groups.return_value = [MagicMock()]

    mock_generate_index.side_effect = lambda df, *args, **kwargs: df # type: ignore
    mock_normalization.peptide_normalization.side_effect = (
        lambda global_pept, mod_pept, *args, **kwargs: mod_pept  # type: ignore
    )
    # stub batch_correction to passthrough a DataFrame
    mock_normalization.batch_correction.side_effect = lambda df, *args, **kwargs: df  # type: ignore
    # stub prot_abund_correction to passthrough a DataFrame
    mock_abundance.prot_abund_correction.side_effect = lambda df, *args, **kwargs: df  # type: ignore

    mock_stats.log2_transformation.side_effect = lambda df, cols: df # type: ignore
    mock_filter_missingness.side_effect = lambda df, *args, **kwargs: df # type: ignore
    mock_abundance.global_prot_normalization_and_stats.side_effect = lambda **kwargs: kwargs["global_prot"] # type: ignore
    mock_rollup.rollup_to_site.side_effect = lambda *args, **kwargs: args[0] # type: ignore
    mock_stats.anova.side_effect = lambda df, *args, **kwargs: df # type: ignore
    mock_stats.pairwise_ttest.side_effect = lambda df, *args, **kwargs: df # type: ignore
    mock_ptm.combine_multi_ptms.side_effect = lambda dct, par: list(dct.values())[0] # type: ignore

    all_ptms, global_prot, all_ptms_uncorrected = ptm_analysis_return_all(dummy_params) # type: ignore

    assert isinstance(all_ptms, pd.DataFrame)
    assert isinstance(global_prot, pd.DataFrame)
    assert isinstance(all_ptms_uncorrected, pd.DataFrame)


