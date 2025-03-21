from proteometer.params import Params


def test_params_lip():
    par = Params("tests/data/test_config_lip.toml")
    assert par is not None


def test_params_ptm():
    par = Params("tests/data/test_config_ptm.toml")
    assert par is not None
