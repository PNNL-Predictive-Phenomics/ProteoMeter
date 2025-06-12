import warnings

from proteometer.params import Params
from proteometer.peptide import nip_off_pept
from proteometer.ptm import get_ptm_pos_in_pept
from proteometer.ptm_analysis import ptm_analysis


def test_ptm_analysis():
    warnings.filterwarnings("error")
    par = Params("tests/data/test_config_ptm.toml")
    dfs = ptm_analysis(par)
    warnings.resetwarnings()
    for df in dfs:
        assert df is not None
        print(df)
        print(df.columns)


def test_ptm_position():
    SEQ = "K.G@VSEK@.D"

    expected_nip = "G@VSEK@"
    assert nip_off_pept(SEQ) == expected_nip

    expected_pos = [1, 5]  # G@ in position 1, K@ in position 5
    assert get_ptm_pos_in_pept(SEQ, ptm_label="@") == expected_pos
