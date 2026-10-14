import pandas as pd
import pytest
from modules.importer_functions import import_observed_shifts
from pathlib import Path

TEST_DATA = Path(__file__).parent / 'test_data'

@pytest.mark.parametrize(
    'filetype',
    ["csv", "json", "shiftx", "SPARTA+", "", "banana", "tinkywinky"]
)
def test_invalid_filetypes(filetype, capsys):
    result = import_observed_shifts(filename='irrelevant', filetype=filetype)

    captured = capsys.readouterr()
    assert result == None
    assert captured.out.strip() == (
        f'import_obs_shifts: invalid filetype \'{filetype}\'.'
    )

@pytest.mark.parametrize(
    "filetype, filename, expected_filename",
    [
        ("ccpn", 'ccpn_resonances.txt', 'ccpn_snaps_internal'),
        ("nef", 'P3a_L273R.nef', 'nef_snaps_internal'),
        ('nmrpipe', 'nmrpipe_shifts.tab', 'nmrpipe_snaps_internal'),
        ('sparky', 'sparky_shifts.txt', 'sparky_snaps_internal'),
     ]
)
def test_file_formats(filetype, filename, expected_filename):
    result = import_observed_shifts(filename=TEST_DATA / filename, filetype=filetype)

    # result.to_pickle(expected_filename)

    expected = pd.read_pickle(TEST_DATA / expected_filename)

    pd.testing.assert_frame_equal(
        result,
        expected,
        check_exact=False,
        check_dtype=False,
        rtol=1e-7,
        atol=1e-8,
    )

@pytest.mark.parametrize(
    "filetype, filename",
    [('ccpn', 'P3a_L273R.nef'), ('nef', 'nmrpipe_shifts')]
)
def test_incorrect_file(filetype, filename):
    with pytest.raises(Exception):
        result = import_observed_shifts(filename=TEST_DATA / filename, filetype=filetype)
