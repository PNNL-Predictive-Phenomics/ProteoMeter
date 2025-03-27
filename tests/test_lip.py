from proteometer.lip_analysis import lip_analysis
from proteometer.params import Params


def test_lip_analysis():
    par = Params("tests/data/test_config_lip.toml")
    df = lip_analysis(par)

    assert df is not None
