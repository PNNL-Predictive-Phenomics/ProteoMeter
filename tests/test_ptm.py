from proteometer.params import Params
from proteometer.ptm_analysis import ptm_analysis


def test_ptm_analysis():
    par = Params("tests/data/test_config_ptm.toml")
    df = ptm_analysis(par)

    assert df is not None
