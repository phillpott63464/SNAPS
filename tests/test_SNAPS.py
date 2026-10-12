import pytest
from pathlib import Path

from SNAPS import runSNAPS

TEST_DATA = Path(__file__).parent / 'test_data'


@pytest.mark.parametrize(
    'measured_shifts, expected_shifts, config_file, shift_type, expected_file',
    [
        (
            '../data/P3a_L273R/ccpn_resonances.txt',
            '../data/P3a_L273R/shiftx2.cs',
            '../config/config_yaml_2.txt',
            'ccpn',
            TEST_DATA / 'p3a_ccpn_shiftx2_test_results',
        )
    ],
)
def test_SNAPS(
    tmp_path,
    measured_shifts,
    expected_shifts,
    config_file,
    shift_type,
    expected_file,
):
    out_file = tmp_path / 'out.txt'
    log_file = tmp_path / 'log.txt'

    runSNAPS(
        [
            measured_shifts,
            expected_shifts,
            str(out_file),
            '-c',
            config_file,
            '-l',
            str(log_file),
            '--shift_type',
            shift_type,
        ]
    )

    OUTPUTTED = out_file.read_text()
    EXPECTED = expected_file.read_text()

    assert OUTPUTTED == EXPECTED
