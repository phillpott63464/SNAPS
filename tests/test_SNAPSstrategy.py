from refactor.SNAPSstrategy import create_strategy
from refactor.SNAPSimporter import SNAPSImporter
from modules.assigner_functions import prepare_obs_preds, calc_log_prob_matrix, find_best_assignment, make_assign_df, import_predicted_shifts
from modules.importer_functions import import_observed_shifts
from refactor.SNAPSconfig import build_config
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from pandas.testing import assert_frame_equal
import logging
from scipy.stats import norm

TEST_DATA = Path(__file__).parent / 'test_data'


@pytest.fixture
def logger():
    return logging.getLogger('test_SNAPSimporter')


@pytest.fixture
def unit_strategy():
    config = build_config(
        cli_overrides={
            'atom_set': {'H', 'N'},
            'atom_sd': {'H': 0.5, 'N': 1.0},
            'delta_correlation': False,
        }
    )
    return create_strategy('direct', config)


def test_strategy_factory_uses_built_config(unit_strategy):
    assert unit_strategy.name == 'direct'
    assert unit_strategy.config.atom_set == {'H', 'N'}
    assert unit_strategy.config.atom_sd == {'H': 0.5, 'N': 1.0}
    assert unit_strategy.config.delta_correlation is False


def test_strategy_uses_logger_from_built_config(caplog):
    configured_logger = logging.getLogger('test.configured_strategy')
    config = build_config(
        cli_overrides={
            'logger': configured_logger,
            'delta_correlation': False,
        }
    )
    strategy = create_strategy('direct', config)
    scores = pd.DataFrame(
        [[1.0]],
        index=pd.Index(['SS1'], name='SS_name'),
        columns=pd.Index(['R1'], name='Res_name'),
    )

    with caplog.at_level(logging.INFO, logger=configured_logger.name):
        strategy.find_best_assignment(scores)

    assert strategy.config.logger is configured_logger
    assert 'Started linear assignment' in caplog.text


def test_find_best_assignment_logs_constraint_warnings(unit_strategy, caplog):
    scores = pd.DataFrame(
        [[10.0, 1.0, 2.0], [2.0, 9.0, 3.0], [4.0, 5.0, 8.0]],
        index=pd.Index(['SS1', 'SS2', 'SS3'], name='SS_name'),
        columns=pd.Index(['R1', 'R2', 'R3'], name='Res_name'),
    )
    inc = pd.DataFrame({
        'SS_name': ['SS1', 'SS1', 'SS3'],
        'Res_name': ['R1', 'R2', 'R3'],
    })
    exc = pd.DataFrame({'SS_name': ['SS3'], 'Res_name': ['R2']})

    with caplog.at_level(logging.WARNING, logger=unit_strategy.config.logger.name):
        unit_strategy.find_best_assignment(
            scores,
            inc=inc,
            exc=exc,
            dummy_rows=set(),
            dummy_cols=set(),
        )

    warnings = [record.getMessage() for record in caplog.records]
    assert 'Warning: entries in inc conflict with one another - dropping conflicts' in warnings
    assert 'Some values in exc are also found in inc, so are redundant.' in warnings


def test_calc_log_prob_matrix_logs_completion(unit_strategy, caplog):
    obs = pd.DataFrame(
        {'H': [1.0], 'Dummy_SS': [False]},
        index=pd.Index(['SS1'], name='SS_name'),
    )
    preds = pd.DataFrame({
        'Res_name': ['R1'],
        'Res_type': ['A'],
        'Dummy_res': [False],
        'H': [1.0],
    }, index=['R1'])
    preds.index.name = 'Res_name'

    with caplog.at_level(logging.INFO, logger=unit_strategy.config.logger.name):
        unit_strategy.calc_log_prob_matrix(obs, preds)

    assert 'Calculated log probability matrix (1x1)' in caplog.text


def test_calc_log_prob_matrix_logs_prediction_correction(
    tmp_path, caplog
):
    correction_file = tmp_path / 'linear_model.csv'
    pd.DataFrame(
        {
            'Atom_type': ['H'],
            'Res_type': ['A'],
            'Grad': [0.0],
            'Offset': [0.0],
        },
        index=['H_A'],
    ).to_csv(correction_file)
    config = build_config(
        cli_overrides={
            'logger': logging.getLogger('test.prediction_correction'),
            'atom_set': {'H'},
            'atom_sd': {'H': 1.0},
            'delta_correlation': False,
            'pred_correction': True,
            'pred_correction_file': str(correction_file),
        }
    )
    strategy = create_strategy('direct', config)
    obs = pd.DataFrame(
        {'H': [1.0], 'Dummy_SS': [False]},
        index=pd.Index(['SS1'], name='SS_name'),
    )
    preds = pd.DataFrame({
        'Res_name': ['R1'],
        'Res_type': ['A'],
        'Dummy_res': [False],
        'H': [1.0],
    }, index=['R1'])
    preds.index.name = 'Res_name'

    with caplog.at_level(logging.INFO, logger=config.logger.name):
        strategy.calc_log_prob_matrix(obs, preds)

    assert f'Imported pred_correction info from {correction_file}' in caplog.text
    assert 'Calculating corrections to predicted shifts' in caplog.text


def test_calc_log_prob_matrix_logs_correlated_error_processing(tmp_path, caplog):
    mean_file = tmp_path / 'd_mean.csv'
    cov_file = tmp_path / 'd_cov.csv'
    pd.DataFrame({1: [0.0]}, index=['H']).to_csv(mean_file, header=False)
    pd.DataFrame([[1.0]], index=['H'], columns=['H']).to_csv(cov_file)
    config = build_config(
        cli_overrides={
            'logger': logging.getLogger('test.correlated_errors'),
            'atom_set': {'H'},
            'atom_sd': {'H': 1.0},
            'delta_correlation': True,
            'delta_correlation_mean_file': str(mean_file),
            'delta_correlation_cov_file': str(cov_file),
        }
    )
    strategy = create_strategy('direct', config)
    obs = pd.DataFrame(
        {'H': [1.0], 'Dummy_SS': [False]},
        index=pd.Index(['SS1'], name='SS_name'),
    )
    preds = pd.DataFrame({
        'Res_name': ['R1'],
        'Res_type': ['A'],
        'Dummy_res': [False],
        'H': [1.0],
    }, index=['R1'])
    preds.index.name = 'Res_name'

    with caplog.at_level(logging.INFO, logger=config.logger.name):
        strategy.calc_log_prob_matrix(obs, preds)

    assert (
        f'Imported delta_correlation info from {mean_file} and {cov_file}'
        in caplog.text
    )
    assert 'Accounting for correlated prediction errors' in caplog.text


def test_calc_log_prob_matrix_logs_corrected_correlated_files(tmp_path, caplog):
    correction_file = tmp_path / 'linear_model.csv'
    corrected_mean_file = tmp_path / 'dd_mean.csv'
    corrected_cov_file = tmp_path / 'dd_cov.csv'
    pd.DataFrame(
        {
            'Atom_type': ['H'],
            'Res_type': ['A'],
            'Grad': [0.0],
            'Offset': [0.0],
        },
        index=['H_A'],
    ).to_csv(correction_file)
    pd.DataFrame({1: [0.0]}, index=['H']).to_csv(
        corrected_mean_file, header=False
    )
    pd.DataFrame([[1.0]], index=['H'], columns=['H']).to_csv(corrected_cov_file)
    config = build_config(
        cli_overrides={
            'logger': logging.getLogger('test.corrected_correlated_errors'),
            'atom_set': {'H'},
            'atom_sd': {'H': 1.0},
            'delta_correlation': True,
            'pred_correction': True,
            'pred_correction_file': str(correction_file),
            'delta_correlation_mean_corrected_file': str(corrected_mean_file),
            'delta_correlation_cov_corrected_file': str(corrected_cov_file),
        }
    )
    strategy = create_strategy('direct', config)
    obs = pd.DataFrame(
        {'H': [1.0], 'Dummy_SS': [False]},
        index=pd.Index(['SS1'], name='SS_name'),
    )
    preds = pd.DataFrame({
        'Res_name': ['R1'],
        'Res_type': ['A'],
        'Dummy_res': [False],
        'H': [1.0],
    }, index=['R1'])
    preds.index.name = 'Res_name'

    with caplog.at_level(logging.INFO, logger=config.logger.name):
        strategy.calc_log_prob_matrix(obs, preds)

    assert (
        f'Imported delta_correlation info from {corrected_mean_file} and '
        f'{corrected_cov_file}' in caplog.text
    )


def test_find_best_assignment_logs_fixed_and_excluded_assignments(
    unit_strategy, caplog
):
    scores = pd.DataFrame(
        [[10.0, 1.0], [2.0, 9.0]],
        index=pd.Index(['SS1', 'SS2'], name='SS_name'),
        columns=pd.Index(['R1', 'R2'], name='Res_name'),
    )
    inc = pd.DataFrame({'SS_name': ['SS1'], 'Res_name': ['R1']})
    exc = pd.DataFrame({'SS_name': ['SS2'], 'Res_name': ['R2']})

    with caplog.at_level(logging.INFO, logger=unit_strategy.config.logger.name):
        unit_strategy.find_best_assignment(
            scores,
            inc=inc,
            exc=exc,
            dummy_rows=set(),
            dummy_cols=set(),
        )

    assert '1 assignments were fixed, 1 remain to be assigned' in caplog.text
    assert 'Penalised 1 excluded row,column pairs' in caplog.text


def test_find_best_assignment_logs_all_dummy_debug_message(unit_strategy, caplog):
    scores = pd.DataFrame(
        [[1.0]],
        index=pd.Index(['DSS_1'], name='SS_name'),
        columns=pd.Index(['DR_1'], name='Res_name'),
    )

    with caplog.at_level(logging.DEBUG, logger=unit_strategy.config.logger.name):
        result = unit_strategy.find_best_assignment(
            scores,
            dummy_rows={'DSS_1'},
            dummy_cols={'DR_1'},
            return_none_all_dummy=True,
        )

    assert result is None
    assert 'Score matrix includes only dummy rows/columns' in caplog.text


def test_make_assign_df_logs_invalid_matching_warning(unit_strategy, caplog):
    obs = pd.DataFrame({'SS_name': ['SS1'], 'H': [1.0]})
    preds = pd.DataFrame({
        'Res_name': ['R1'],
        'Res_N': [1],
        'Res_type': ['A'],
        'Dummy_res': [False],
        'H': [1.0],
    })
    log_prob_matrix = pd.DataFrame(
        [[0.0]],
        index=pd.Index(['SS1'], name='SS_name'),
        columns=pd.Index(['R1'], name='Res_name'),
    )

    with caplog.at_level(logging.WARNING, logger=unit_strategy.config.logger.name):
        with pytest.raises(KeyError):
            unit_strategy.make_assign_df(
                obs=obs,
                preds=preds,
                log_prob_matrix=log_prob_matrix,
                matching=pd.DataFrame({'invalid': ['value']}),
            )

    assert (
        'Cannot make assignment dataframe, as matching dataframe does not have '
        'the correct columns'
    ) in caplog.text


def test_preprocessing_uses_configured_atom_schema_and_dummy_names(logger):
    config = build_config(
        cli_overrides={
            'atom_set': {'H'},
            'supported_atom_types': {'H', 'N'},
            'dummy_predicted_prefix': 'PRED_',
            'dummy_observed_prefix': 'OBS_',
        }
    )
    strategy = create_strategy('direct', config)
    obs = pd.DataFrame({
        'SS_name': ['SS1', 'SS2'],
        'H': [1.0, 2.0],
        'N': [10.0, 20.0],
    })
    preds = pd.DataFrame({
        'Res_N': [1, 2],
        'Res_type': ['A', 'P'],
        'Res_name': ['R1', 'R2'],
        'Res_name_m1': [np.nan, 'R1'],
        'Res_name_p1': ['R2', np.nan],
        'H': [1.0, 2.0],
        'N': [10.0, 20.0],
    })

    result_obs, result_preds = strategy.preprocess_obs_preds(obs, preds)

    assert result_preds['Res_name'].tolist() == ['R1', 'PRED_1']
    assert result_obs['SS_name'].tolist() == ['SS1', 'SS2']
    assert 'N' not in result_obs.columns
    assert 'N' not in result_preds.columns


def test_log_probability_defaults_come_from_config(logger):
    config = build_config(
        cli_overrides={
            'atom_set': {'H'},
            'atom_sd': {'H': 2.0},
            'delta_correlation': False,
            'default_probability': 0.2,
            'shift_scale_factor': 2.0,
        }
    )
    strategy = create_strategy('direct', config)
    obs = pd.DataFrame(
        {'H': [np.nan], 'Dummy_SS': [False]},
        index=pd.Index(['SS1'], name='SS_name'),
    )
    preds = pd.DataFrame(
        {'H': [2.0], 'Dummy_res': [False]},
        index=pd.Index(['R1'], name='Res_name'),
    )

    result = strategy.calc_log_prob_matrix(obs, preds)

    assert result.loc['SS1', 'R1'] == pytest.approx(np.log10(0.2))


def test_calc_log_prob_matrix_calculates_independent_atom_probabilities(
        unit_strategy, logger
):
    obs = pd.DataFrame(
        {'H': [1.0], 'N': [10.0], 'Dummy_SS': [False]},
        index=pd.Index(['SS1'], name='SS_name'),
    )
    preds = pd.DataFrame(
        {'H': [1.5], 'N': [10.0], 'Dummy_res': [False]},
        index=pd.Index(['R1'], name='Res_name'),
    )

    result = unit_strategy.calc_log_prob_matrix(obs, preds)
    expected = norm.logpdf(0.5, scale=0.5) + norm.logpdf(0.0, scale=1.0)

    assert result.index.name == 'SS_name'
    assert result.columns.name == 'Res_name'
    assert result.loc['SS1', 'R1'] == pytest.approx(expected)


def test_calc_log_prob_matrix_penalises_missing_shifts(unit_strategy, logger):
    obs = pd.DataFrame(
        {'H': [np.nan], 'N': [10.0], 'Dummy_SS': [False]},
        index=pd.Index(['SS1'], name='SS_name'),
    )
    preds = pd.DataFrame(
        {'H': [1.0], 'N': [10.0], 'Dummy_res': [False]},
        index=pd.Index(['R1'], name='Res_name'),
    )

    result = unit_strategy.calc_log_prob_matrix(obs, preds)
    expected = np.log10(0.01) + norm.logpdf(0.0, scale=1.0)

    assert result.loc['SS1', 'R1'] == pytest.approx(expected)


def test_find_best_assignment_maximises_scores(unit_strategy, logger):
    scores = pd.DataFrame(
        [[10.0, 1.0], [2.0, 9.0]],
        index=pd.Index(['SS1', 'SS2'], name='SS_name'),
        columns=pd.Index(['R1', 'R2'], name='Res_name'),
    )

    result = unit_strategy.find_best_assignment(scores)

    assert set(zip(result['SS_name'], result['Res_name'])) == {
        ('SS1', 'R1'),
        ('SS2', 'R2'),
    }


def test_find_best_assignment_honours_required_assignment(
        unit_strategy, logger
):
    scores = pd.DataFrame(
        [[10.0, 1.0], [2.0, 9.0]],
        index=pd.Index(['SS1', 'SS2'], name='SS_name'),
        columns=pd.Index(['R1', 'R2'], name='Res_name'),
    )
    required = pd.DataFrame({'SS_name': ['SS1'], 'Res_name': ['R2']})

    result = unit_strategy.find_best_assignment(
        scores,
        inc=required,
    )

    assert set(zip(result['SS_name'], result['Res_name'])) == {
        ('SS1', 'R2'),
        ('SS2', 'R1'),
    }


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
    )

    expected_log_prob_matrix = calc_log_prob_matrix(
        obs=obs,
        preds=preds,
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
    )

    matching = strategy.find_best_assignment(
        score_matrix=log_prob_matrix,
        maximise=True,
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
    )

    matching = strategy.find_best_assignment(
        score_matrix=log_prob_matrix,
        maximise=True,
    )

    assign_df = strategy.make_assign_df(
        obs=obs,
        preds=preds,
        log_prob_matrix=log_prob_matrix,
        matching=matching,
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
