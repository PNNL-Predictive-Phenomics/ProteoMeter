import warnings

from proteometer.lip_analysis import lip_analysis
from proteometer.params import Params


def test_lip_simple():
    warnings.filterwarnings(
        "error"
    )  # example constructed to be numerically fine; let's test that
    par = Params("tests/data/test_config_lip.toml")
    dfs = lip_analysis(par)
    warnings.resetwarnings()
    for df in dfs:
        assert df is not None
        print(df)
        print(df.columns)
