import warnings

from proteometer.params import Params
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
