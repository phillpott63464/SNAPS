from refactor.SNAPS import run_pipeline

from refactor.SNAPSconfig import build_config
import pytest

from pathlib import Path

TEST_DATA = Path(__file__).parent / 'test_data'

import pytest
from pathlib import Path

# noinspection unresolved-references
from SNAPS import run_snaps
import pandas as pd
import shutil
import os
import re
import numpy as np
from pandas.testing import assert_frame_equal

TEST_DATA = Path(__file__).parent / 'test_data'


@pytest.mark.parametrize(
    'measured_shifts, expected_shifts, config_file, shift_type, prediction_type',
    [
        (
            'data/P3a_L273R/ccpn_resonances.txt',
            'data/P3a_L273R/shiftx2.cs',
            'config/config_yaml_2.txt',
            'ccpn',
            'shiftx2',
        ),
        (
            'data/P3a_L273R/nmrpipe_shifts.tab',
            'data/P3a_L273R/shiftx2.cs',
            'config/config_yaml_2.txt',
            'nmrpipe',
            'shiftx2',
        ),
        (
            'data/P3a_L273R/P3a_L273R.nef',
            'data/P3a_L273R/shiftx2.cs',
            'config/config_yaml_2.txt',
            'nef',
            'shiftx2',
        ),
        (
            'data/P3a_L273R/ccpn_resonances.txt',
            'data/P3a_L273R/P3a_L273R_sparta.tab',
            'config/config_yaml_2.txt',
            'ccpn',
            'sparta+',
        ),
        (
            'data/P3a_L273R/nmrpipe_shifts.tab',
            'data/P3a_L273R/P3a_L273R_sparta.tab',
            'config/config_yaml_2.txt',
            'nmrpipe',
            'sparta+',
        ),
        (
            'data/P3a_L273R/P3a_L273R.nef',
            'data/P3a_L273R/P3a_L273R_sparta.tab',
            'config/config_yaml_2.txt',
            'nef',
            'sparta+',
        ),
        (
            'data/P3a_L273R/sparky_shifts.txt',
            'data/P3a_L273R/P3a_L273R_sparta.tab',
            'config/config_yaml_2.txt',
            'sparky',
            'sparta+',
        ),
    ],
    ids=[
        'p3a_ccpn_shiftx2',
        'p3a_nmrpipe_shiftx2',
        'p3a_nef_shiftx2',
        'p3a_ccpn_sparta',
        'p3a_nmrpipe_sparta',
        'p3a_nef_sparta',
        'p3a_sparky_sparta',
    ],
)
def test_run_snaps(
    tmp_path,
    measured_shifts,
    expected_shifts,
    config_file,
    shift_type,
    prediction_type,
):
    out_file_legacy = tmp_path / 'out_legacy.txt'
    out_file_refactor = tmp_path / 'out_refactor.txt'
    log_file = tmp_path / 'log.txt'

    run_snaps(
        [
            measured_shifts,
            expected_shifts,
            str(out_file_legacy),
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

    config = build_config(
        config_file=Path(config_file),
        cli_overrides={
            'obs_file': Path(measured_shifts),
            'pred_file': Path(expected_shifts),
            'output_file': out_file_refactor,
        },
    )

    run_pipeline(config)

    assert_frame_equal(
        pd.read_csv(out_file_refactor), pd.read_csv(out_file_legacy)
    )
