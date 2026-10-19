from modules.SNAPSstrategy import create_strategy
from modules.SNAPSimporter import SNAPSImporter
from modules.assigner_functions import prepare_obs_preds, calc_log_prob_matrix, find_best_assignment, make_assign_df, import_predicted_shifts
from modules.importer_functions import import_observed_shifts
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

    strategy = create_strategy('direct')

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
    strategy = create_strategy('direct')
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

@pytest.mark.parametrize(
    'pred_file, obs_file',
    [
        ('shiftx2.cs', 'P3a_L273R.nef'),
        ('P3a_L273R_sparta.tab', 'ccpn_resonances.txt'),
        ('P3a_L273R_sparta.tab', 'nmrpipe_shifts.tab'),
        ('shiftx2.cs', 'sparky_shifts.txt'),
    ],
)
def test_SNAPstrategy_find_best_assignment(pred_file, obs_file, logger):
    importer = SNAPSImporter()
    imported = importer.import_data(
        pred_file=TEST_DATA / pred_file,
        obs_file=TEST_DATA / obs_file,
    )
    strategy = create_strategy('direct')
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

    matching = strategy.find_best_assignment(
        score_matrix=log_prob_matrix,
        maximise=True,
        logger=logger,
    )

    expected = find_best_assignment(
        score_matrix=log_prob_matrix,
        maximise=True,
        logger=logger,
    )
    assert_frame_equal(matching, expected)


@pytest.mark.parametrize(
    'pred_file, obs_file',
    [
        ('shiftx2.cs', 'P3a_L273R.nef'),
        ('P3a_L273R_sparta.tab', 'ccpn_resonances.txt'),
        ('P3a_L273R_sparta.tab', 'nmrpipe_shifts.tab'),
        ('shiftx2.cs', 'sparky_shifts.txt'),
    ],
)
def test_SNAPstrategy_make_assign_df(pred_file, obs_file, logger):
    importer = SNAPSImporter()
    imported = importer.import_data(
        pred_file=TEST_DATA / pred_file,
        obs_file=TEST_DATA / obs_file,
    )
    strategy = create_strategy('direct')
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

    matching = strategy.find_best_assignment(
        score_matrix=log_prob_matrix,
        maximise=True,
        logger=logger,
    )

    assign_df = strategy.make_assign_df(
        obs=obs,
        preds=preds,
        log_prob_matrix=log_prob_matrix,
        matching=matching,
        logger=logger,
    )

    expected = make_assign_df(
        obs=obs,
        preds=preds,
        log_prob_matrix=log_prob_matrix,
        matching=matching,
        logger=logger,
    )

    assert_frame_equal(assign_df, expected)

@pytest.mark.parametrize(
    'pred_file, obs_file, pred_format, obs_format',
    [
        ('shiftx2.cs', 'P3a_L273R.nef', 'shiftx2', 'nef'),
        ('P3a_L273R_sparta.tab', 'ccpn_resonances.txt', 'sparta+', 'ccpn'),
        ('P3a_L273R_sparta.tab', 'nmrpipe_shifts.tab', 'sparta+', 'nmrpipe'),
        ('shiftx2.cs', 'sparky_shifts.txt', 'shiftx2', 'sparky'),
    ],
)
def test_SNAPstragety_direct(pred_file, obs_file, pred_format, obs_format, logger):
    # FULLY DEFAULT PATHWAY
    obs = import_observed_shifts(filename = TEST_DATA / obs_file, filetype=obs_format)
    preds, _ = import_predicted_shifts(filename = TEST_DATA / pred_file, filetype = pred_format, logger=logger)
    obs, preds = prepare_obs_preds(obs, preds, logger)
    log_prob_matrix = calc_log_prob_matrix(
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
    matching = find_best_assignment(log_prob_matrix, maximise=True, logger=logger)
    assign_df = make_assign_df(
            obs=obs,
            preds=preds,
            log_prob_matrix=log_prob_matrix,
            matching=matching,
            logger=logger,
        )

    importer = SNAPSImporter()
    strategy = create_strategy('direct')
    imported = importer.import_data(
        pred_file=TEST_DATA / pred_file,
        obs_file=TEST_DATA / obs_file,
    )

    assigned = strategy.assign(data=imported)

    assert_frame_equal(assigned.assign_df, assign_df)
    assert_frame_equal(assigned.assignments, matching)
    assert_frame_equal(assigned.log_prob_matrix, log_prob_matrix)
