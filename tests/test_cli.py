from pathlib import Path

import pytest
from typer.testing import CliRunner

from refactor import SNAPS


runner = CliRunner()


def invoke_cli(monkeypatch, *arguments):
    captured = []
    monkeypatch.setattr(SNAPS, 'run_pipeline', captured.append)

    result = runner.invoke(SNAPS.app, list(arguments))

    return result, captured


def test_positional_and_common_options_build_expected_config(
    monkeypatch, tmp_path
):
    observed = tmp_path / 'observed.txt'
    predicted = tmp_path / 'predicted.cs'
    output = tmp_path / 'assignments.txt'

    result, captured = invoke_cli(
        monkeypatch,
        str(observed),
        str(predicted),
        str(output),
        '--obs-format',
        'sparky',
        '--pred-format',
        'shiftx2',
        '--obs-frame',
        'frame_1',
        '--pred-frame',
        'frame_2',
        '--obs-chain',
        'B',
        '--pred-chain',
        'C',
        '--output-type',
        'csv',
        '--strategy',
        'iterative',
    )

    assert result.exit_code == 0, result.stdout
    assert len(captured) == 1
    config = captured[0]
    assert config.obs_file == observed
    assert config.pred_file == predicted
    assert config.output_file == output
    assert config.obs_format == 'sparky'
    assert config.pred_format == 'shiftx2'
    assert config.obs_frame == 'frame_1'
    assert config.pred_frame == 'frame_2'
    assert config.obs_chain == 'B'
    assert config.pred_chain == 'C'
    assert config.output_type == 'csv'
    assert config.strategy == 'iterative'


def test_repeatable_and_boolean_options_build_expected_config(
    monkeypatch, tmp_path
):
    result, captured = invoke_cli(
        monkeypatch,
        str(tmp_path / 'observed.txt'),
        str(tmp_path / 'predicted.cs'),
        '--atom-set',
        'H',
        '--atom-set',
        'N',
        '--supported-atom-types',
        'H',
        '--supported-atom-types',
        'CA',
        '--atom-sd',
        'H=0.2',
        '--atom-sd',
        'N=1.25',
        '--no-maximise-assignments',
        '--no-iterate-until-consistent',
        '--use-ss-class-info',
        '--delta-correlation',
        '--pred-correction',
        '--pred-offset',
        '7',
        '--default-probability',
        '0.05',
        '--shift-scale-factor',
        '1.25',
        '--seq-link-threshold',
        '0.6',
    )

    assert result.exit_code == 0, result.stdout
    assert len(captured) == 1
    config = captured[0]
    assert config.atom_set == {'H', 'N'}
    assert config.supported_atom_types == {'H', 'CA'}
    assert config.atom_sd == {'H': 0.2, 'N': 1.25}
    assert config.maximise_assignments is False
    assert config.iterate_until_consistent is False
    assert config.use_ss_class_info is True
    assert config.delta_correlation is True
    assert config.pred_correction is True
    assert config.pred_offset == 7
    assert config.default_probability == 0.05
    assert config.shift_scale_factor == 1.25
    assert config.seq_link_threshold == 0.6


def test_cli_file_options_and_cli_values_override_config_file(
    monkeypatch, tmp_path
):
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(
        'strategy: bayesian\n'
        'seq_link_threshold: 0.2\n'
        'pred_offset: 3\n'
        'iterate_until_consistent: false\n'
    )
    observed = tmp_path / 'observed.txt'
    predicted = tmp_path / 'predicted.cs'
    output = tmp_path / 'from-option.txt'

    result, captured = invoke_cli(
        monkeypatch,
        '--shift-file',
        str(observed),
        '--pred-file',
        str(predicted),
        '--output-file',
        str(output),
        '--config-file',
        str(config_file),
        '--strategy',
        'direct',
        '--seq-link-threshold',
        '0.8',
        '--iterate-until-consistent',
    )

    assert result.exit_code == 0, result.stdout
    assert len(captured) == 1
    config = captured[0]
    assert config.obs_file == observed
    assert config.pred_file == predicted
    assert config.output_file == output
    assert config.strategy == 'direct'
    assert config.seq_link_threshold == 0.8
    assert config.pred_offset == 3
    assert config.iterate_until_consistent is True


def test_nef_input_uses_same_file_for_predictions(monkeypatch, tmp_path):
    nef_file = tmp_path / 'shifts.nef'

    result, captured = invoke_cli(monkeypatch, str(nef_file))

    assert len(captured) == 1
    assert result.exit_code == 0, result.stdout
    assert captured[0].obs_file == nef_file
    assert captured[0].pred_file == nef_file
    assert captured[0].output_file == Path('-')


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        (('--pred-file', 'predicted.cs'), 'a shift_file must be provided'),
        (('observed.txt',), 'a pred_file must be provided'),
    ],
)
def test_missing_required_input_is_reported(
    monkeypatch, tmp_path, arguments, message
):
    result, captured = invoke_cli(monkeypatch, *arguments)

    assert result.exit_code != 0
    assert message in result.stdout
    assert captured == []