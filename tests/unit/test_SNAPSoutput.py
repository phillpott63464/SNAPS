import pandas as pd
import pytest

from refactor.SNAPSoutput import (
    FormatOutput,
    OutputRegistry,
    SNAPSOutput,
    create_output,
)
from refactor.dataclasses import AssignmentResult
from refactor.SNAPSconfig import build_config


def _assignment_result():
    assign_df = pd.DataFrame([
        {
            "Res_name": "1G",
            "Res_N": 1,
            "Res_type": "G",
            "SS_name": "SS1",
            "CA": 45.2,
            "CA_m1": None,
            "CA_pred": 45.5,
            "CA_m1_pred": None,
            "Log_prob": -0.5,
            "Confidence": "High",
        },
        {
            "Res_name": "2A",
            "Res_N": 2,
            "Res_type": "A",
            "SS_name": "SS2",
            "CA": 52.1,
            "CA_m1": 45.2,
            "CA_pred": 52.4,
            "CA_m1_pred": 45.5,
            "Log_prob": -0.2,
            "Confidence": "Medium",
        },
    ])

    return AssignmentResult(
        assignments=pd.DataFrame(),
        assign_df=assign_df,
    )


def test_snaps_output_preserves_all_columns():
    result = _assignment_result()

    rendered = SNAPSOutput()._output(result)

    for column in result.assign_df.columns:
        assert column in rendered.splitlines()[0]


def test_snaps_output_preserves_column_order():
    result = _assignment_result()

    rendered = SNAPSOutput()._output(result)

    headers = rendered.splitlines()[0].split()

    assert headers == result.assign_df.columns.tolist()


def test_snaps_output_includes_sequential_shifts():
    result = _assignment_result()

    rendered = SNAPSOutput()._output(result)

    headers = rendered.splitlines()[0].split()

    assert "CA_m1" in headers
    assert "CA_m1_pred" in headers


def test_snaps_output_writes_rendered_table_to_file(tmp_path):
    output_file = tmp_path / "assignments.txt"
    result = _assignment_result()

    SNAPSOutput().output(
        result,
        build_config(cli_overrides={"output_file": output_file}),
    )

    assert output_file.read_text().splitlines() == (
        SNAPSOutput()._output(result).splitlines()
    )


def test_snaps_output_writes_rendered_table_to_stdout(capsys):
    result = _assignment_result()

    SNAPSOutput().output(
        result,
        build_config(cli_overrides={"output_file": "-"}),
    )

    captured = capsys.readouterr()

    assert SNAPSOutput()._output(result) in captured.out


def test_output_registry_registers_snaps_and_factory_preserves_config():
    config = {"format": "plain"}

    assert OutputRegistry.get("snaps") is SNAPSOutput
    assert "snaps" in OutputRegistry.list_names()

    output = create_output("snaps", config)

    assert isinstance(output, SNAPSOutput)
    assert output.config is config


def test_output_registry_rejects_unknown_output():
    with pytest.raises(ValueError, match="Unknown strategy: missing"):
        OutputRegistry.get("missing")


def test_format_output_requires_subclasses_to_implement_rendering():
    result = _assignment_result()

    assert FormatOutput()._output(result) is None