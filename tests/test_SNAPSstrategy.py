from modules.SNAPSstrategy import create_strategy
from modules.SNAPSimporter import SNAPSImporter
from modules.assigner_functions import prepare_obs_preds, calc_log_prob_matrix
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
    'pred_file, obs_file',
    [
        ('shiftx2.cs', 'P3a_L273R.nef'),
        ('P3a_L273R_sparta.tab', 'ccpn_resonances.txt'),
        ('P3a_L273R_sparta.tab', 'nmrpipe_shifts.tab'),
        ('shiftx2.cs', 'sparky_shifts.txt'),
    ],
)
def test_SNAPSstrategy_preprocess(pred_file, obs_file, logger):
    importer = SNAPSImporter()
    imported = importer.import_data(
        pred_file=TEST_DATA / pred_file,
        obs_file=TEST_DATA / obs_file,
    )

    strategy = create_strategy('iterative')

    result_obs, result_preds = strategy.preprocess_obs_preds(
        imported.obs_shifts, imported.pred_shifts
    )

    expected_obs, expected_preds = prepare_obs_preds(
        imported.obs_shifts, imported.pred_shifts, logger
    )

    assert_frame_equal(result_obs, expected_obs)
    assert_frame_equal(result_preds, expected_preds)

@pytest.mark.parametrize(
    'pred_file, obs_file',
    [
        ('shiftx2.cs', 'P3a_L273R.nef'),
        ('P3a_L273R_sparta.tab', 'ccpn_resonances.txt'),
        ('P3a_L273R_sparta.tab', 'nmrpipe_shifts.tab'),
        ('shiftx2.cs', 'sparky_shifts.txt'),
    ],
)
def test_SNAPstrategy_log_prob_matrix(pred_file, obs_file, logger):
    importer = SNAPSImporter()
    imported = importer.import_data(
        pred_file=TEST_DATA / pred_file,
        obs_file=TEST_DATA / obs_file,
    )
    strategy = create_strategy('iterative')
    obs, preds = strategy.preprocess_obs_preds(
        imported.obs_shifts, imported.pred_shifts
    )

    log_prob_matrix = strategy.calc_log_prob_matrix(
        obs,
        preds,
        pred_correction=False,
        delta_correlation=True,
        delta_correlation_mean_corrected_file='config/dd_mean.csv',
        delta_correlation_cov_corrected_file='config/dd_cov.csv',
        delta_correlation_mean_file='config/d_mean.csv',
        delta_correlation_cov_file='config/d_cov.csv',
        logger=logger,
    )

    expected_log_prob_matrix = calc_log_prob_matrix(
        obs=obs,
        preds=preds,
        pred_correction=False,
        delta_correlation=True,
        delta_correlation_mean_corrected_file='config/dd_mean.csv',
        delta_correlation_cov_corrected_file='config/dd_cov.csv',
        delta_correlation_mean_file='config/d_mean.csv',
        delta_correlation_cov_file='config/d_cov.csv',
        logger=logger,
    )

    assert_frame_equal(log_prob_matrix, expected_log_prob_matrix)
