from refactor.SNAPSconfig import build_config, SNAPSConfig
import pytest


@pytest.fixture
def complete_config(tmp_path):
    file_path = tmp_path / 'config.yaml'
    data = """
# Atom settings
atom_set:
  - H
  - N
  - CA
  - CB

atom_sd:
  H: 0.20
  N: 1.25
  HA: 0.15
  CA: 0.50
  CB: 0.60
  C: 0.65

# Assignment settings
seq_link_threshold: 0.60
iterate_until_consistent: false
use_ss_class_info: true
delta_correlation: false
pred_correction: true
pred_correction_file: custom/linear_model.csv
delta_correlation_mean_corrected_file: custom/dd_mean.csv
delta_correlation_cov_corrected_file: custom/dd_cov.csv
delta_correlation_mean_file: custom/d_mean.csv
delta_correlation_cov_file: custom/d_cov.csv
supported_atom_types:
- H
- N
- CA
default_probability: 0.05
shift_scale_factor: 1.25
pred_offset: 7
maximise_assignments: false
dummy_predicted_prefix: PRED_
dummy_observed_prefix: OBS_

# Strategy
strategy: bayesian
    """

    with open(file_path, 'w') as f:
        f.write(data)

    return file_path


@pytest.fixture
def partial_config(tmp_path):
    file_path = tmp_path / 'config.yaml'
    data = """
seq_link_threshold: 0.70
    """
    with open(file_path, 'w') as f:
        f.write(data)

    return file_path


@pytest.fixture
def out_of_range_config(tmp_path):
    file_path = tmp_path / 'config.yaml'
    data = """
seq_link_threshold: 47
    """
    with open(file_path, 'w') as f:
        f.write(data)

    return file_path


def test_default():
    config = build_config()

    assert sorted(config.atom_set) == sorted(
        {'H', 'N', 'HA', 'CA', 'CB', 'C', 'C_m1', 'CA_m1', 'CB_m1'}
    )
    assert config.atom_sd == {
        'H': 0.1726,
        'N': 1.1169,
        'HA': 0.1231,
        'CA': 0.4412,
        'CB': 0.5163,
        'C': 0.5256,
    }
    assert sorted(config.supported_atom_types) == sorted(
        {'H', 'N', 'C', 'CA', 'CB', 'C_m1', 'CA_m1', 'CB_m1', 'HA'}
    )
    assert config.default_probability == 0.01
    assert config.shift_scale_factor == 1.0
    assert config.pred_offset == 0
    assert config.maximise_assignments is True
    assert config.dummy_predicted_prefix == 'DR_'
    assert config.dummy_observed_prefix == 'DSS_'
    assert config.seq_link_threshold == 0.2
    assert config.iterate_until_consistent == True
    assert config.use_ss_class_info == False
    assert config.delta_correlation == True
    assert config.pred_correction == False
    assert config.pred_correction_file == 'config/lin_model_shiftx2.csv'
    assert config.delta_correlation_mean_corrected_file == 'config/dd_mean.csv'
    assert config.delta_correlation_cov_corrected_file == 'config/dd_cov.csv'
    assert config.delta_correlation_mean_file == 'config/d_mean.csv'
    assert config.delta_correlation_cov_file == 'config/d_cov.csv'
    assert config.strategy == 'direct'


def test_complete(complete_config):
    config = build_config(config_file=complete_config)

    assert sorted(config.atom_set) == sorted({'H', 'N', 'CA', 'CB'})
    assert config.atom_sd == {
        'H': 0.20,
        'N': 1.25,
        'HA': 0.15,
        'CA': 0.50,
        'CB': 0.60,
        'C': 0.65,
    }
    assert sorted(config.supported_atom_types) == ['CA', 'H', 'N']
    assert config.default_probability == 0.05
    assert config.shift_scale_factor == 1.25
    assert config.pred_offset == 7
    assert config.maximise_assignments is False
    assert config.dummy_predicted_prefix == 'PRED_'
    assert config.dummy_observed_prefix == 'OBS_'
    assert config.seq_link_threshold == 0.60
    assert config.iterate_until_consistent == False
    assert config.use_ss_class_info == True
    assert config.delta_correlation == False
    assert config.pred_correction == True
    assert config.pred_correction_file == 'custom/linear_model.csv'
    assert config.delta_correlation_mean_corrected_file == 'custom/dd_mean.csv'
    assert config.delta_correlation_cov_corrected_file == 'custom/dd_cov.csv'
    assert config.delta_correlation_mean_file == 'custom/d_mean.csv'
    assert config.delta_correlation_cov_file == 'custom/d_cov.csv'
    assert config.strategy == 'bayesian'


def test_partial(partial_config):
    config = build_config(config_file=partial_config)

    assert sorted(config.atom_set) == sorted(
        {'H', 'N', 'HA', 'CA', 'CB', 'C', 'C_m1', 'CA_m1', 'CB_m1'}
    )
    assert config.atom_sd == {
        'H': 0.1726,
        'N': 1.1169,
        'HA': 0.1231,
        'CA': 0.4412,
        'CB': 0.5163,
        'C': 0.5256,
    }
    assert config.seq_link_threshold == 0.70
    assert config.pred_offset == 0
    assert config.iterate_until_consistent == True
    assert config.use_ss_class_info == False
    assert config.delta_correlation == True
    assert config.pred_correction == False
    assert config.strategy == 'direct'


def test_out_of_range(out_of_range_config):
    with pytest.raises(ValueError):
        config = build_config(config_file=out_of_range_config)


@pytest.fixture
def complete_cli_overrides():
    return {
        'atom_set': ['H', 'N', 'CA', 'CB'],
        'atom_sd': {
            'H': 0.20,
            'N': 1.25,
            'HA': 0.15,
            'CA': 0.50,
            'CB': 0.60,
            'C': 0.65,
        },
        'supported_atom_types': ['H', 'N', 'CA'],
        'default_probability': 0.05,
        'shift_scale_factor': 1.25,
        'pred_offset': 10,
        'maximise_assignments': False,
        'dummy_predicted_prefix': 'PRED_',
        'dummy_observed_prefix': 'OBS_',
        'seq_link_threshold': 0.60,
        'iterate_until_consistent': False,
        'use_ss_class_info': True,
        'delta_correlation': False,
        'pred_correction': True,
        'pred_correction_file': 'cli/linear_model.csv',
        'delta_correlation_mean_corrected_file': 'cli/dd_mean.csv',
        'delta_correlation_cov_corrected_file': 'cli/dd_cov.csv',
        'delta_correlation_mean_file': 'cli/d_mean.csv',
        'delta_correlation_cov_file': 'cli/d_cov.csv',
        'strategy': 'bayesian',
    }


@pytest.fixture
def partial_cli_overrides():
    return {'seq_link_threshold': 0.70}


@pytest.fixture
def out_of_range_cli_overrides():
    return {'seq_link_threshold': 47}


def test_cli_default():
    config = build_config(cli_overrides={})

    assert sorted(config.atom_set) == sorted(
        {'H', 'N', 'HA', 'CA', 'CB', 'C', 'C_m1', 'CA_m1', 'CB_m1'}
    )
    assert config.atom_sd == {
        'H': 0.1726,
        'N': 1.1169,
        'HA': 0.1231,
        'CA': 0.4412,
        'CB': 0.5163,
        'C': 0.5256,
    }
    assert config.seq_link_threshold == 0.2
    assert config.iterate_until_consistent == True
    assert config.use_ss_class_info == False
    assert config.delta_correlation == True
    assert config.pred_correction == False
    assert config.strategy == 'direct'


def test_cli_complete(complete_cli_overrides):
    config = build_config(cli_overrides=complete_cli_overrides)

    assert sorted(config.atom_set) == sorted({'H', 'N', 'CA', 'CB'})
    assert config.atom_sd == {
        'H': 0.20,
        'N': 1.25,
        'HA': 0.15,
        'CA': 0.50,
        'CB': 0.60,
        'C': 0.65,
    }
    assert sorted(config.supported_atom_types) == ['CA', 'H', 'N']
    assert config.default_probability == 0.05
    assert config.shift_scale_factor == 1.25
    assert config.pred_offset == 10
    assert config.maximise_assignments is False
    assert config.dummy_predicted_prefix == 'PRED_'
    assert config.dummy_observed_prefix == 'OBS_'
    assert config.seq_link_threshold == 0.60
    assert config.iterate_until_consistent == False
    assert config.use_ss_class_info == True
    assert config.delta_correlation == False
    assert config.pred_correction == True
    assert config.pred_correction_file == 'cli/linear_model.csv'
    assert config.delta_correlation_mean_corrected_file == 'cli/dd_mean.csv'
    assert config.delta_correlation_cov_corrected_file == 'cli/dd_cov.csv'
    assert config.delta_correlation_mean_file == 'cli/d_mean.csv'
    assert config.delta_correlation_cov_file == 'cli/d_cov.csv'
    assert config.strategy == 'bayesian'


def test_cli_partial(partial_cli_overrides):
    config = build_config(cli_overrides=partial_cli_overrides)

    assert sorted(config.atom_set) == sorted(
        {'H', 'N', 'HA', 'CA', 'CB', 'C', 'C_m1', 'CA_m1', 'CB_m1'}
    )
    assert config.atom_sd == {
        'H': 0.1726,
        'N': 1.1169,
        'HA': 0.1231,
        'CA': 0.4412,
        'CB': 0.5163,
        'C': 0.5256,
    }
    assert config.seq_link_threshold == 0.70
    assert config.pred_offset == 0
    assert config.iterate_until_consistent == True
    assert config.use_ss_class_info == False
    assert config.delta_correlation == True
    assert config.pred_correction == False
    assert config.strategy == 'direct'


def test_cli_out_of_range(out_of_range_cli_overrides):
    with pytest.raises(ValueError):
        build_config(cli_overrides=out_of_range_cli_overrides)


def test_cli_overrides_file(complete_config):
    """CLI overrides take priority over YAML configuration."""
    overrides = {
        'seq_link_threshold': 0.80,
        'strategy': 'iterative',
        'iterate_until_consistent': True,
        'pred_correction': False,
        'pred_correction_file': 'override/linear_model.csv',
        'delta_correlation_mean_corrected_file': 'override/dd_mean.csv',
        'delta_correlation_cov_corrected_file': 'override/dd_cov.csv',
        'delta_correlation_mean_file': 'override/d_mean.csv',
        'delta_correlation_cov_file': 'override/d_cov.csv',
    }

    config = build_config(config_file=complete_config, cli_overrides=overrides)

    # Values overridden by CLI
    assert config.seq_link_threshold == 0.80
    assert config.strategy == 'iterative'
    assert config.iterate_until_consistent == True
    assert config.pred_correction == False
    assert config.pred_correction_file == 'override/linear_model.csv'
    assert (
        config.delta_correlation_mean_corrected_file == 'override/dd_mean.csv'
    )
    assert config.delta_correlation_cov_corrected_file == 'override/dd_cov.csv'
    assert config.delta_correlation_mean_file == 'override/d_mean.csv'
    assert config.delta_correlation_cov_file == 'override/d_cov.csv'

    # Values retained from YAML
    assert sorted(config.atom_set) == sorted(['H', 'N', 'CA', 'CB'])
    assert config.atom_sd == {
        'H': 0.20,
        'N': 1.25,
        'HA': 0.15,
        'CA': 0.50,
        'CB': 0.60,
        'C': 0.65,
    }
    assert config.use_ss_class_info == True
    assert config.delta_correlation == False


def test_cli_none_overrides():
    """None values should not override defaults."""
    overrides = {
        'seq_link_threshold': None,
        'strategy': None,
        'pred_correction': None,
        'use_ss_class_info': True,
    }

    config = build_config(cli_overrides=overrides)

    assert config.seq_link_threshold == 0.2
    assert config.strategy == 'direct'
    assert config.pred_correction == False
    assert config.use_ss_class_info == True
