import pytest
from pathlib import Path

from SNAPS import runSNAPS

TEST_DATA = Path(__file__).parent / 'test_data'


def test_SNAPS(tmp_path):
    out_file = tmp_path / 'out.txt'
    log_file = tmp_path / 'log.txt'
    expected_file = TEST_DATA / 'p3a_ccpn_shiftx2_test_results'

    runSNAPS(
        [
            '../data/P3a_L273R/ccpn_resonances.txt',
            '../data/P3a_L273R/shiftx2.cs',
            str(out_file),
            '-c',
            '../config/config_yaml_2.txt',
            '-l',
            str(log_file),
            '--shift_type',
            'ccpn',
        ]
    )

    OUTPUTTED = out_file.read_text()
    EXPECTED = expected_file.read_text()

    assert OUTPUTTED == EXPECTED
