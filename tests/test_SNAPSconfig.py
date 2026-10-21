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

    assert config.atom_set == sorted(['H', 'N', 'HA', 'CA', 'CB', 'C'])
    assert config.atom_sd == {
        'H': 0.1726, 'N': 1.1169, 'HA': 0.1231,
        'CA': 0.4412, 'CB': 0.5163, 'C': 0.5256
    }
    assert config.seq_link_threshold == 0.45
    assert config.iterate_until_consistent == True
    assert config.use_ss_class_info == False
    assert config.delta_correlation == True
    assert config.strategy == 'direct'

def test_complete(complete_config):
    config = build_config(config_file=complete_config)

    assert config.atom_set == sorted(['H', 'N', 'CA', 'CB'])
    assert config.atom_sd == {'H': 0.20, 'N': 1.25, 'HA': 0.15, 'CA': 0.50, 'CB': 0.60, 'C': 0.65}
    assert config.seq_link_threshold == 0.60
    assert config.iterate_until_consistent == False
    assert config.use_ss_class_info == True
    assert config.delta_correlation == False
    assert config.strategy == 'bayesian'

def test_partial(partial_config):
    config = build_config(config_file=partial_config)

    assert config.atom_set == sorted(['H', 'N', 'HA', 'CA', 'CB', 'C'])
    assert config.atom_sd == {
        'H': 0.1726, 'N': 1.1169, 'HA': 0.1231,
        'CA': 0.4412, 'CB': 0.5163, 'C': 0.5256
    }
    assert config.seq_link_threshold == 0.70
    assert config.iterate_until_consistent == True
    assert config.use_ss_class_info == False
    assert config.delta_correlation == True
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
            'C': 0.65
        },
        'seq_link_threshold': 0.60,
        'iterate_until_consistent': False,
        'use_ss_class_info': True,
        'delta_correlation': False,
        'strategy': 'bayesian'
    }


@pytest.fixture
def partial_cli_overrides():
    return {
        'seq_link_threshold': 0.70
    }


@pytest.fixture
def out_of_range_cli_overrides():
    return {
        'seq_link_threshold': 47
    }


def test_cli_default():
    config = build_config(cli_overrides={})

    assert config.atom_set == sorted(['H', 'N', 'HA', 'CA', 'CB', 'C'])
    assert config.atom_sd == {
        'H': 0.1726, 'N': 1.1169, 'HA': 0.1231,
        'CA': 0.4412, 'CB': 0.5163, 'C': 0.5256
    }
    assert config.seq_link_threshold == 0.45
    assert config.iterate_until_consistent == True
    assert config.use_ss_class_info == False
    assert config.delta_correlation == True
    assert config.strategy == 'direct'


def test_cli_complete(complete_cli_overrides):
    config = build_config(cli_overrides=complete_cli_overrides)

    assert config.atom_set == sorted(['H', 'N', 'CA', 'CB'])
    assert config.atom_sd == {
        'H': 0.20, 'N': 1.25, 'HA': 0.15,
        'CA': 0.50, 'CB': 0.60, 'C': 0.65
    }
    assert config.seq_link_threshold == 0.60
    assert config.iterate_until_consistent == False
    assert config.use_ss_class_info == True
    assert config.delta_correlation == False
    assert config.strategy == 'bayesian'


def test_cli_partial(partial_cli_overrides):
    config = build_config(cli_overrides=partial_cli_overrides)

    assert config.atom_set == sorted(['H', 'N', 'HA', 'CA', 'CB', 'C'])
    assert config.atom_sd == {
        'H': 0.1726, 'N': 1.1169, 'HA': 0.1231,
        'CA': 0.4412, 'CB': 0.5163, 'C': 0.5256
    }
    assert config.seq_link_threshold == 0.70
    assert config.iterate_until_consistent == True
    assert config.use_ss_class_info == False
    assert config.delta_correlation == True
    assert config.strategy == 'direct'


def test_cli_out_of_range(out_of_range_cli_overrides):
    with pytest.raises(ValueError):
        build_config(cli_overrides=out_of_range_cli_overrides)


def test_cli_overrides_file(complete_config):
    """CLI overrides take priority over YAML configuration."""
    overrides = {
        'seq_link_threshold': 0.80,
        'strategy': 'iterative',
        'iterate_until_consistent': True
    }

    config = build_config(
        config_file=complete_config,
        cli_overrides=overrides
    )

    # Values overridden by CLI
    assert config.seq_link_threshold == 0.80
    assert config.strategy == 'iterative'
    assert config.iterate_until_consistent == True

    # Values retained from YAML
    assert config.atom_set == sorted(['H', 'N', 'CA', 'CB'])
    assert config.atom_sd == {
        'H': 0.20, 'N': 1.25, 'HA': 0.15,
        'CA': 0.50, 'CB': 0.60, 'C': 0.65
    }
    assert config.use_ss_class_info == True
    assert config.delta_correlation == False


def test_cli_none_overrides():
    """None values should not override defaults."""
    overrides = {
        'seq_link_threshold': None,
        'strategy': None,
        'use_ss_class_info': True
    }

    config = build_config(cli_overrides=overrides)

    assert config.seq_link_threshold == 0.45
    assert config.strategy == 'direct'
    assert config.use_ss_class_info == True
