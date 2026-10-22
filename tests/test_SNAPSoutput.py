import pandas as pd
import pytest

from refactor.SNAPSoutput import (
    FormatOutput,
    OutputRegistry,
    SNAPSOutput,
    create_output,
)
from refactor.dataclasses import AssignmentResult, OutputSpec


def _assignment_result():
    headings = SNAPSOutput.headings
    first = {heading: f"first_{heading}" for heading in headings}
    second = {heading: f"second_{heading}" for heading in headings}
    # Deliberately use a different column order: SNAPS output has a fixed format.
    assign_df = pd.DataFrame([first, second], columns=list(reversed(headings)))
    return AssignmentResult(
        assignments=pd.DataFrame(),
        assign_df=assign_df,
    )


def test_snaps_output_uses_heading_order_and_preserves_rows():
    result = _assignment_result()

    output = SNAPSOutput()._output(result)
    lines = output.splitlines()

    assert lines[0].split() == SNAPSOutput.headings
    assert lines[1].split() == [
        f"first_{heading}" for heading in SNAPSOutput.headings
    ]
    assert lines[2].split() == [
        f"second_{heading}" for heading in SNAPSOutput.headings
    ]


def test_snaps_output_writes_rendered_table_to_file(tmp_path):
    output_file = tmp_path / "assignments.txt"

    SNAPSOutput().output(
        _assignment_result(),
        OutputSpec(output_file=output_file),
    )

    assert output_file.read_text().splitlines() == (
        SNAPSOutput()._output(_assignment_result()).splitlines()
    )


def test_snaps_output_writes_rendered_table_to_stdout(capsys):
    SNAPSOutput().output(
        _assignment_result(),
        OutputSpec(output_file="-"),
    )

    captured = capsys.readouterr()
    assert SNAPSOutput()._output(_assignment_result()) in captured.out


def test_output_registry_registers_snaps_and_factory_preserves_config():
    config = {"format": "plain"}

    assert OutputRegistry.get("SNAPS") is SNAPSOutput
    assert "SNAPS" in OutputRegistry.list_names()

    output = create_output("SNAPS", config)
    assert isinstance(output, SNAPSOutput)
    assert output.config is config


def test_output_registry_rejects_unknown_output():
    with pytest.raises(ValueError, match="Unknown strategy: missing"):
        OutputRegistry.get("missing")


def test_format_output_requires_subclasses_to_implement_rendering():
    result = _assignment_result()

    assert FormatOutput()._output(result) is None
