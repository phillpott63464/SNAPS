"""
Regression testing the old one to the new one to make sure that
the expected outputs are the same
"""

from refactor.SNAPSimporter import SNAPSImporter
from refactor.SNAPSconfig import build_config
from modules.importer_functions import (
    import_observed_shifts,
)   # original function (stateless version)
from modules.assigner_functions import import_predicted_shifts
import pytest
import pandas as pd
from pathlib import Path
from pandas.testing import assert_frame_equal
import logging

TEST_DATA = Path(__file__).parent / 'test_data'


@pytest.fixture
def logger():
    return logging.getLogger('test_SNAPSimporter')


@pytest.mark.parametrize(
    'file, format, saved_result',
    [
        ('P3a_L273R.nef', 'nef', 'nef_snaps_internal'),
        ('ccpn_resonances.txt', 'ccpn', 'ccpn_snaps_internal'),
        ('nmrpipe_shifts.tab', 'nmrpipe', 'nmrpipe_snaps_internal'),
        ('sparky_shifts.txt', 'sparky', 'sparky_snaps_internal'),
    ],
)
def test_SNAPSimporterObserved(file, format, saved_result):
    importer = SNAPSImporter()
    file_path = TEST_DATA / file
    result = importer.import_data(
        pred_file=TEST_DATA / 'shiftx2.cs',  # Irrelevant to these tests
        obs_file=file_path,
        pred_format='shiftx2',  # Irrelevant to these tests
        obs_format=format,
    )

    expected_result = import_observed_shifts(
        filename=file_path, filetype=format
    )

    assert_frame_equal(result.obs_shifts, expected_result)


@pytest.mark.parametrize(
    'file, format, saved_result',
    [
        ('shiftx2.cs', 'shiftx2', 'shiftx2_snaps_internal'),
        ('P3a_L273R_sparta.tab', 'sparta+', 'sparta_snaps_internal'),
    ],
)
def test_SNAPSimporterPredicted(file, format, logger, saved_result):
    importer = SNAPSImporter()
    file_path = TEST_DATA / file
    result = importer.import_data(
        pred_file=file_path,
        obs_file=TEST_DATA / 'P3a_L273R.nef',
        pred_format=format,
        obs_format='nef',
    )

    expected_result, _ = import_predicted_shifts(
        filename=file_path, filetype=format, logger=logger
    )

    assert_frame_equal(result.pred_shifts, expected_result)


def test_SNAPSimporterExtensionMapping(logger):
    importer = SNAPSImporter()
    result = importer.import_data(
        pred_file=TEST_DATA / 'shiftx2.cs',
        obs_file=TEST_DATA / 'P3a_L273R.nef',
        pred_format=None,
        obs_format=None,
    )

    expected_preds, _ = import_predicted_shifts(
        filename=TEST_DATA / 'shiftx2.cs', filetype='shiftx2', logger=logger
    )
    expected_obs = import_observed_shifts(
        filename=TEST_DATA / 'P3a_L273R.nef', filetype='nef'
    )

    assert_frame_equal(expected_obs, result.obs_shifts)
    assert_frame_equal(expected_preds, result.pred_shifts)


def test_SNAPSimporterLogsSelectedReaders(caplog):
    configured_logger = logging.getLogger('test.snaps_importer_logging')
    importer = SNAPSImporter(
        config=build_config(cli_overrides={'logger': configured_logger})
    )

    with caplog.at_level(logging.INFO, logger=configured_logger.name):
        importer.import_data(
            pred_file=TEST_DATA / 'shiftx2.cs',
            obs_file=TEST_DATA / 'P3a_L273R.nef',
            pred_format='shiftx2',
            obs_format='nef',
        )

    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == configured_logger.name
    ] == [
        'Observed shifts reader=nef',
        'Predicted shifts reader=shiftx2',
    ]


def test_SNAPSimporterCanRead(logger):
    # So this one tests if it can read files just from their signatures
    importer = SNAPSImporter()
    result = importer.import_data(
        pred_file=TEST_DATA / 'P3a_L273R_sparta.tab',
        obs_file=TEST_DATA
        / 'nmrpipe_shifts.tab',  # See how these are both the same file extension
        pred_format=None,
        obs_format=None,
    )

    expected_preds, _ = import_predicted_shifts(
        filename=TEST_DATA / 'P3a_L273R_sparta.tab',
        filetype='sparta+',
        logger=logger,
    )
    expected_obs = import_observed_shifts(
        filename=TEST_DATA / 'nmrpipe_shifts.tab', filetype='nmrpipe'
    )

    assert_frame_equal(expected_obs, result.obs_shifts)
    assert_frame_equal(expected_preds, result.pred_shifts)
