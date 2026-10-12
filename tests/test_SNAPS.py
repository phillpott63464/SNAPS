import pytest
from pathlib import Path

from SNAPS import runSNAPS

TEST_DATA = Path(__file__).parent / 'test_data'

#### ALWAYS RUN TESTS USING PYTHONHASHSEED=0 uv run pytest
#### Assignment results depend on hash randomisation. Haven't looked into a fix yet. All test data was generated with hashseed=0


@pytest.mark.parametrize(
    'measured_shifts, expected_shifts, config_file, shift_type, expected_file, prediction_type',
    [
        (
            'data/P3a_L273R/ccpn_resonances.txt',
            'data/P3a_L273R/shiftx2.cs',
            'config/config_yaml_2.txt',
            'ccpn',
            TEST_DATA / 'p3a_ccpn_shiftx2_test_results',
            'shiftx2',
        ),
        (
            'data/P3a_L273R/nmrpipe_shifts.tab',
            'data/P3a_L273R/shiftx2.cs',
            'config/config_yaml_2.txt',
            'nmrpipe',
            TEST_DATA / 'p3a_nmrpipe_shiftx2_test_results',
            'shiftx2',
        ),
        (
            'data/P3a_L273R/P3a_L273R.nef',
            'data/P3a_L273R/shiftx2.cs',
            'config/config_yaml_2.txt',
            'nef',
            TEST_DATA / 'p3a_nef_shiftx2_test_results',
            'shiftx2',
        ),
        (
            'data/P3a_L273R/ccpn_resonances.txt',
            'data/P3a_L273R/P3a_L273R_sparta.tab',
            'config/config_yaml_2.txt',
            'ccpn',
            TEST_DATA / 'p3a_ccpn_sparta_test_results',
            'sparta+',
        ),
        (
            'data/P3a_L273R/nmrpipe_shifts.tab',
            'data/P3a_L273R/P3a_L273R_sparta.tab',
            'config/config_yaml_2.txt',
            'nmrpipe',
            TEST_DATA / 'p3a_nmrpipe_sparta_test_results',
            'sparta+',
        ),
        (
            'data/P3a_L273R/P3a_L273R.nef',
            'data/P3a_L273R/P3a_L273R_sparta.tab',
            'config/config_yaml_2.txt',
            'nef',
            TEST_DATA / 'p3a_nef_sparta_test_results',
            'sparta+',
        ),
        (
            'data/P3a_L273R/sparky_shifts.txt',
            'data/P3a_L273R/P3a_L273R_sparta.tab',
            'config/config_yaml_2.txt',
            'sparky',
            TEST_DATA / 'p3a_sparky_sparta_test_results',
            'sparta+',
        ),
    ],
)
def test_SNAPS(
    tmp_path,
    measured_shifts,
    expected_shifts,
    config_file,
    shift_type,
    expected_file,
    prediction_type,
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
            '--pred_type',
            prediction_type,
        ]
    )

    OUTPUTTED = out_file.read_text()
    EXPECTED = expected_file.read_text()

    assert OUTPUTTED == EXPECTED
