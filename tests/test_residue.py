from proteometer.residue import get_res_names, get_res_pos


def test_get_res_names_single_residue():
    residues = ["A123"]
    expected = [["A123"]]
    assert get_res_names(residues) == expected


def test_get_res_names_multiple_residues_in_string():
    residues = ["A123B456"]
    expected = [["A123", "B456"]]
    assert get_res_names(residues) == expected


def test_get_res_names_with_lowercase_and_hyphen():
    residues = ["A123a-B456c"]
    expected = [["A123a-", "B456c"]]
    assert get_res_names(residues) == expected


def test_get_res_names_multiple_strings():
    residues = ["A123B456", "C789d-E101f"]
    expected = [["A123", "B456"], ["C789d-", "E101f"]]
    assert get_res_names(residues) == expected


def test_get_res_names_empty_string():
    residues = [""]
    expected: list[list[str]] = [[]]
    assert get_res_names(residues) == expected


def test_get_res_names_no_matches():
    residues = ["xyz", "123", "-"]
    expected: list[list[str]] = [[], [], []]
    assert get_res_names(residues) == expected


def test_get_res_names_mixed_valid_and_invalid():
    residues = ["A123", "xyz", "B456c", ""]
    expected = [["A123"], [], ["B456c"], []]
    assert get_res_names(residues) == expected


def test_get_res_pos_single_residue():
    residues = ["A123"]
    expected = [[123]]
    assert get_res_pos(residues) == expected


def test_get_res_pos_multiple_residues_in_string():
    residues = ["A123B456"]
    expected = [[123, 456]]
    assert get_res_pos(residues) == expected


def test_get_res_pos_with_lowercase_and_hyphen():
    residues = ["A123a-B456c"]
    expected = [[123, 456]]
    assert get_res_pos(residues) == expected


def test_get_res_pos_multiple_strings():
    residues = ["A123B456", "C789d-E101f"]
    expected = [[123, 456], [789, 101]]
    assert get_res_pos(residues) == expected


def test_get_res_pos_empty_string():
    residues = [""]
    expected: list[list[str]] = [[]]
    assert get_res_pos(residues) == expected


def test_get_res_pos_no_matches():
    residues = ["xyz", "-", "abc"]
    expected: list[list[str]] = [[], [], []]
    assert get_res_pos(residues) == expected


def test_get_res_pos_mixed_valid_and_invalid():
    residues = ["A123", "xyz", "B456c", ""]
    expected = [[123], [], [456], []]
    assert get_res_pos(residues) == expected
