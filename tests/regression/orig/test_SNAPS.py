import pytest
from pathlib import Path

# noinspection unresolved-references
from SNAPS import run_snaps
import pandas as pd
import shutil
import os
import re
import numpy as np

TEST_DATA = Path(__file__).resolve().parents[2] / 'test_data'

@pytest.mark.parametrize(
    "measured_shifts, expected_shifts, config_file, shift_type, expected_file, prediction_type",
    [
        (
            str(TEST_DATA / "ccpn_resonances.txt"),
            str(TEST_DATA / "shiftx2.cs"),
            str(TEST_DATA / "config_yaml_2.txt"),
            "ccpn",
            str(TEST_DATA / "p3a_ccpn_shiftx2_test_results"),
            "shiftx2",
        ),
        (
            str(TEST_DATA / "nmrpipe_shifts.tab"),
            str(TEST_DATA / "shiftx2.cs"),
            str(TEST_DATA / "config_yaml_2.txt"),
            "nmrpipe",
            str(TEST_DATA / "p3a_nmrpipe_shiftx2_test_results"),
            "shiftx2",
        ),
        (
            str(TEST_DATA / "P3a_L273R.nef"),
            str(TEST_DATA / "shiftx2.cs"),
            str(TEST_DATA / "config_yaml_2.txt"),
            "nef",
            str(TEST_DATA / "p3a_nef_shiftx2_test_results"),
            "shiftx2",
        ),
        (
            str(TEST_DATA / "ccpn_resonances.txt"),
            str(TEST_DATA / "P3a_L273R_sparta.tab"),
            str(TEST_DATA / "config_yaml_2.txt"),
            "ccpn",
            str(TEST_DATA / "p3a_ccpn_sparta_test_results"),
            "sparta+",
        ),
        (
            str(TEST_DATA / "nmrpipe_shifts.tab"),
            str(TEST_DATA / "P3a_L273R_sparta.tab"),
            str(TEST_DATA / "config_yaml_2.txt"),
            "nmrpipe",
            str(TEST_DATA / "p3a_nmrpipe_sparta_test_results"),
            "sparta+",
        ),
        (
            str(TEST_DATA / "P3a_L273R.nef"),
            str(TEST_DATA / "P3a_L273R_sparta.tab"),
            str(TEST_DATA / "config_yaml_2.txt"),
            "nef",
            str(TEST_DATA / "p3a_nef_sparta_test_results"),
            "sparta+",
        ),
        (
            str(TEST_DATA / "sparky_shifts.txt"),
            str(TEST_DATA / "P3a_L273R_sparta.tab"),
            str(TEST_DATA / "config_yaml_2.txt"),
            "sparky",
            str(TEST_DATA / "p3a_sparky_sparta_test_results"),
            "sparta+",
        ),
    ],
    ids=[
        "p3a_ccpn_shiftx2",
        "p3a_nmrpipe_shiftx2",
        "p3a_nef_shiftx2",
        "p3a_ccpn_sparta",
        "p3a_nmrpipe_sparta",
        "p3a_nef_sparta",
        "p3a_sparky_sparta",
    ],
)
def test_run_snaps(
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

    run_snaps(
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
            '--out_type',
            'snaps',
        ]
    )

    outputted = pd.read_csv(out_file, sep=r'\s+')
    expected = pd.read_csv(expected_file, sep=r'\s+')

    """
    Tests:
    - All headers present
    - Mismatch cost is not greater than expected
    - If mismatch cost is same as expected, assignment is same
    - If mismatch cost is less than expected, we don't care about what the assignment says
    """

    try:
        assert list(outputted.columns.values) == list(expected.columns.values)
    except:
        raise AssertionError('Column headers do not match')

    outputted_cost = sum(
        (outputted[key] - outputted[f'{key}_pred']).abs().sum()
        for key in ['CA', 'HA', 'H', 'CB', 'C', 'N']
    )
    expected_cost = sum(
        (expected[key] - expected[f'{key}_pred']).abs().sum()
        for key in ['CA', 'HA', 'H', 'CB', 'C', 'N']
    )

    if outputted_cost > expected_cost:
        raise AssertionError('Assignment has gotten worse')
    elif outputted_cost == expected_cost:
        try:
            assert list(outputted['Res_name']) == list(expected['Res_name'])
        except:
            raise AssertionError('Assignment has changed with same cost')
    elif outputted_cost < expected_cost:
        if list(outputted['Res_name']) == list(expected['Res_name']):
            raise AssertionError(
                'Assignment has not changed, but cost has - issue with data imports?'
            )
        raise Warning(
            'Assignment has reduced in cost - this is probably an improvement! But double check.'
        )
