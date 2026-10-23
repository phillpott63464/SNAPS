import pytest
import pandas as pd

from refactor.SNAPSconfig import build_config
from refactor.SNAPSoutput import SNAPSOutput
from refactor.dataclasses import AssignmentResult, ShiftData

"""
Not certain about these tests, testing whether the outputs exist isn't a perfectly
reasonable test. But idk how I'd test the internals of the plot and I don't know if 
bokeh is deterministic so I guess this is the best option?
"""

def _models(value):
    if isinstance(value, dict):
        if 'name' in value:
            yield value
        for child in value.values():
            yield from _models(child)
    elif isinstance(value, list):
        for child in value:
            yield from _models(child)


def _result(with_sequential=True):
    row = {
        'Res_name': '  1A',
        'Res_N': 1,
        'Res_type': 'A',
        'SS_name': 'SS1',
        'Dummy_res': False,
        'Dummy_SS': False,
        'Confidence': 'High',
        'H': 8.1,
        'N': 120.0,
        'Max_mismatch_m1': 0.1,
        'Max_mismatch_p1': 0.2,
    }
    if with_sequential:
        row.update({'C': 175.0, 'C_m1': 174.0})
    return AssignmentResult(
        assignments=pd.DataFrame(),
        assign_df=pd.DataFrame([row]),
    )


def _data():
    return ShiftData(
        obs_shifts=pd.DataFrame(),
        pred_shifts=pd.DataFrame(
            [{'Res_name': '  1A', 'Res_N': 1, 'Res_type': 'A'}]
        ),
    )


def test_hsqc_returns_plot_json_with_expected_title_and_labels():
    output = SNAPSOutput().plot_hsqc(_result())

    assert output['target_id'] is None
    assert any(
        item.get('attributes', {}).get('title', {}).get('attributes', {}).get('text')
        == 'HSQC'
        for item in _models(output['doc']['roots'])
        if item.get('name') == 'Figure'
    )


def test_hsqc_writes_html_file(tmp_path):
    plot_file = tmp_path / 'nested' / 'hsqc.html'

    SNAPSOutput().plot_hsqc(_result(), outfile=plot_file)

    assert plot_file.exists()
    assert 'HSQC' in plot_file.read_text()


def test_strip_plot_returns_json_and_uses_configured_threshold():
    result = SNAPSOutput().plot_strips(
        _result(),
        _data(),
        config=build_config(cli_overrides={'seq_link_threshold': 0.6}),
    )

    assert result['target_id'] == 'strip_plot'
    span = next(
        item
        for item in _models(result['doc']['roots'])
        if item.get('name') == 'Span'
    )
    assert span['attributes']['location'] == 0.6


def test_strip_plot_is_not_created_without_sequential_shifts():
    assert SNAPSOutput().plot_strips(_result(False), _data()) is None
