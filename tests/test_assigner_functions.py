# Tests for functions I have wrapped outside of SNAPS_assigner.py
# Basically making everything stateless for easier testing and to make the dataflow clearer rather than relying
# On previous operations being done in order on self.preds constantly
# This'll then make it easier to add custom dataclasses for data transfer in a legible way
import logging


import logging
import pytest
import pandas as pd
from bokeh.sampledata.mtb import obiszow_mtb_xcm
from modules.assigner_functions import (
    import_predicted_shifts,
    prepare_obs_preds,
    calc_log_prob_matrix,
    find_best_assignment,
    make_assign_df,
    calc_mismatch_matrix,
    add_consistency_info,
)
from modules.importer_functions import import_observed_shifts
from refactor.SNAPSimporter import SNAPSImporter
from pathlib import Path
from pandas.testing import assert_frame_equal

TEST_DATA = Path(__file__).parent / 'test_data'
NAPS_SHIFTS = Path(__file__).parents[1] / 'data/P3a_L273R/naps_shifts.txt'


@pytest.fixture
def logger():
    return logging.getLogger('test_predicted_shifts')


@pytest.mark.parametrize(
    'filetype',
    ['csv', 'json', 'shiftx', 'SPARTA+', '', 'banana', 'tinkywinky'],
)
def test_invalid_filetypes(filetype, logger, caplog):
    """Only 'shiftx2' and 'sparta+' are supported."""
    with caplog.at_level(logging.ERROR):
        result = import_predicted_shifts('dummy.csv', filetype, None, logger)

    assert result is None
    assert f"Invalid predicted shift type: '{filetype}'" in caplog.text
    assert 'Allowed' in caplog.text
    assert "'shiftx2' or 'sparta+'" in caplog.text


@pytest.mark.parametrize('filetype', ['shiftx2', 'sparta+'])
def test_missing_file(filetype, logger):
    """Missing input files should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):   # this is implicitly an assertion
        import_predicted_shifts(
            'this_file_does_not_exist.txt',
            filetype,
            None,
            logger,
        )


@pytest.mark.parametrize(
    'filetype, filename, snapshot',
    [
        ('shiftx2', 'shiftx2.cs', 'shiftx2_snaps_internal'),
        ('sparta+', 'P3a_L273R_sparta.tab', 'sparta_snaps_internal'),
    ],
)
def test_predicted_shifts(logger, filetype, filename, snapshot):
    preds, _ = import_predicted_shifts(
        filename=TEST_DATA / filename,
        filetype=filetype,
        logger=logger,
    )

    # preds.to_pickle(TEST_DATA / snapshot)

    expected = pd.read_pickle(
        TEST_DATA / snapshot,
    )

    pd.testing.assert_frame_equal(
        preds,
        expected,
        check_exact=False,
        rtol=1e-7,
        atol=1e-8,
    )


def test_prepare_obs_preds(logger):
    importer = SNAPSImporter()
    imported = importer.import_data(
        pred_file=TEST_DATA / 'shiftx2.cs',
        obs_file=TEST_DATA / 'P3a_L273R.nef',
    )

    result_obs, result_preds = prepare_obs_preds(
        imported.obs_shifts, imported.pred_shifts, logger
    )

    # result_obs.to_pickle(TEST_DATA / 'prepare_obs.pickle')
    # result_preds.to_pickle(TEST_DATA / 'prepare_preds.pickle')

    expected_obs = pd.read_pickle(TEST_DATA / 'prepare_obs.pickle')
    expected_preds = pd.read_pickle(TEST_DATA / 'prepare_preds.pickle')

    assert_frame_equal(result_obs, expected_obs)
    assert_frame_equal(result_preds, expected_preds)


def test_calc_log_prob_matrix(logger):
    importer = SNAPSImporter()
    imported = importer.import_data(
        pred_file=TEST_DATA / 'shiftx2.cs',
        obs_file=TEST_DATA / 'P3a_L273R.nef',
    )

    obs, preds = prepare_obs_preds(
        imported.obs_shifts, imported.pred_shifts, logger
    )
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

    expected_result = pd.read_pickle(TEST_DATA / 'log_prog.pickle')

    assert_frame_equal(log_prob_matrix, expected_result)


def test_find_best_assignment(logger):
    importer = SNAPSImporter()
    imported = importer.import_data(
        pred_file=TEST_DATA / 'shiftx2.cs',
        obs_file=TEST_DATA / 'P3a_L273R.nef',
    )

    obs, preds = prepare_obs_preds(
        imported.obs_shifts, imported.pred_shifts, logger
    )

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

    matching = find_best_assignment(
        log_prob_matrix, maximise=True, logger=logger
    )

    expected = pd.read_pickle(TEST_DATA / 'matching.pickle')

    assert_frame_equal(matching, expected)


def test_make_assign_df(logger):
    importer = SNAPSImporter()
    imported = importer.import_data(
        pred_file=TEST_DATA / 'shiftx2.cs',
        obs_file=TEST_DATA / 'P3a_L273R.nef',
    )

    obs, preds = prepare_obs_preds(
        imported.obs_shifts, imported.pred_shifts, logger
    )

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

    matching = find_best_assignment(
        log_prob_matrix, maximise=True, logger=logger
    )

    print('obs index:', obs.index.names)
    print('preds index:', preds.index.names)
    print('matching index:', matching.index.names)

    print('obs columns:', obs.columns.tolist())
    print('preds columns:', preds.columns.tolist())
    print('matching columns:', matching.columns.tolist())

    assign_df = make_assign_df(
        obs=obs,
        preds=preds,
        log_prob_matrix=log_prob_matrix,
        matching=matching,
        logger=logger,
    )

    # assign_df.to_pickle(TEST_DATA / 'assign_df.pickle')

    expected = pd.read_pickle(TEST_DATA / 'assign_df.pickle')

    assert_frame_equal(assign_df, expected)


def test_calc_mismatch_matrix(logger):
    observed = import_observed_shifts(NAPS_SHIFTS, 'snaps')

    mismatch, consistent_links = calc_mismatch_matrix(
        observed,
        threshold=0.2,
        logger=logger,
    )

    assert mismatch.shape == (103, 103)
    assert consistent_links.shape == (103, 103)
    assert mismatch.loc[' 237AlaH', ' 238MetH'] == pytest.approx(0.057)
    assert consistent_links.loc[' 237AlaH', ' 238MetH'] == 2
    assert consistent_links.to_numpy().max() == 2


def test_add_consistency_info(logger):
    observed = import_observed_shifts(NAPS_SHIFTS, 'snaps')
    preds, _ = import_predicted_shifts(
        TEST_DATA / 'shiftx2.cs',
        'shiftx2',
        logger=logger,
    )
    mismatch, consistent_links = calc_mismatch_matrix(
        observed,
        threshold=0.2,
        logger=logger,
    )
    matching = pd.DataFrame(
        {
            'Res_name': preds.index[:3],
            'SS_name': observed.index[:3],
        }
    )
    for atom in ('C', 'CA', 'CB', 'C_m1', 'CA_m1', 'CB_m1'):
        matching[atom] = 0.0

    result = add_consistency_info(
        matching,
        preds,
        mismatch,
        consistent_links,
        threshold=0.2,
        logger=logger,
    )

    assert {
        'Max_mismatch_m1',
        'Max_mismatch_p1',
        'Num_good_links_m1',
        'Num_good_links_p1',
        'Confidence',
    }.issubset(result.columns)
    assert result['Confidence'].tolist() == ['High', 'High', 'High']
    assert result.loc[0, 'Num_good_links_p1'] == 2
    assert result.loc[1, 'Num_good_links_m1'] == 2
