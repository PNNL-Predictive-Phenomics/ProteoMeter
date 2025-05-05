import warnings

from proteometer.params import Params
from proteometer.ptm_analysis import ptm_analysis


def test_ptm_analysis():
    warnings.filterwarnings("error")
    par = Params("tests/data/test_config_ptm.toml")
    df = ptm_analysis(par)
    warnings.resetwarnings()
    assert df is not None
    print(df)
    print(df.columns)
