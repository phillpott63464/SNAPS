"""
Regression testing the old one to the new one to make sure that
the expected outputs are the same
"""

from modules.SNAPSimporter import SNAPSImporter
from modules.importer_functions import import_observed_shifts # original function (stateless version)
from modules.assigner_functions import import_predicted_shifts
import pytest
import pandas as pd
from pathlib import Path
from pandas.testing import assert_frame_equal
import logging

TEST_DATA = Path(__file__).parent / 'test_data'

@pytest.fixture
def logger():
    return logging.getLogger("test_SNAPSimporter")

@pytest.mark.parametrize(
    'file, format',
    [
        ('P3a_L273R.nef', 'nef'),
        ('ccpn_resonances.txt', 'ccpn'),
        ('nmrpipe_shifts.tab', 'nmrpipe'),
        ('sparky_shifts.txt', 'sparky'),
    ]
)
def test_SNAPSimporterObserved(file, format):
    importer = SNAPSImporter()
    file_path = TEST_DATA / file
    result = importer.import_data(
        pred_file=TEST_DATA / 'shiftx2.cs', # Irrelevant to these tests
        obs_file=file_path,
        pred_format='shiftx2', # Irrelevant to these tests
        obs_format=format,
    )

    expected_result = import_observed_shifts(filename = file_path, filetype=format)

    assert_frame_equal(result.obs_shifts, expected_result)

@pytest.mark.parametrize(
    'file, format',
    [
        ('shiftx2.cs', 'shiftx2'),
        ('P3a_L273R_sparta.tab', 'sparta+'),
    ]
)
def test_SNAPSimporterPredicted(file, format, logger):
    importer = SNAPSImporter()
    file_path = TEST_DATA / file
    result = importer.import_data(
        pred_file=file_path,
        obs_file=TEST_DATA / 'P3a_L273R.nef',
        pred_format=format,
        obs_format='nef',
    )

    expected_result, _ = import_predicted_shifts(filename = file_path, filetype=format, logger=logger)

    assert_frame_equal(result.pred_shifts, expected_result)

def test_SNAPSimporterExtensionMapping(logger):
    importer = SNAPSImporter()
    result = importer.import_data(
        pred_file=TEST_DATA / 'shiftx2.cs',
        obs_file=TEST_DATA / 'P3a_L273R.nef',
        pred_format=None,
        obs_format=None,
    )

    expected_preds, _ = import_predicted_shifts(filename = TEST_DATA/ 'shiftx2.cs', filetype = 'shiftx2', logger=logger)
    expected_obs = import_observed_shifts(filename = TEST_DATA/ 'P3a_L273R.nef', filetype = 'nef')

    assert_frame_equal(expected_obs, result.obs_shifts)
    assert_frame_equal(expected_preds, result.pred_shifts)