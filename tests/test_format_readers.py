
import logging

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pathlib import Path

from modules.format_readers import ShiftX2Reader  # Update import as needed


@pytest.fixture
def reader():
    return ShiftX2Reader()


@pytest.fixture
def shiftx2_file(tmp_path):
    """Create a sample ShiftX2 prediction file."""
    file_path = tmp_path / "predictions.cs"

    data = pd.DataFrame({
        "NUM": [1, 1, 2, 2],
        "RES": ["ALA", "ALA", "GLY", "GLY"],
        "ATOMNAME": ["CA", "CB", "CA", "N"],
        "SHIFT": [52.1, 19.3, 45.2, 109.5],
    })

    data.to_csv(file_path, index=False)
    return file_path


def test_format_name(reader):
    assert reader.format_name == "shiftx2"


def test_file_extensions(reader):
    assert reader.file_extensions == [".cs", ".txt"]


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("test.cs", True),
        ("test.txt", True),
        ("test.CS", True),
        ("test.TXT", True),
        ("test.csv", False),
        ("test.json", False),
        ("test", False),
        ("test.nef", False)
    ],
)
def test_can_read(reader, filename, expected):
    assert reader.can_read(Path(filename)) is expected


def test_read_observed_shifts(reader, shiftx2_file):
    """ShiftX2 does not contain observed shifts."""
    assert reader.read_observed_shifts(shiftx2_file) is None


def test_read_predicted_shifts(reader, shiftx2_file):
    """Test reading and renaming prediction columns."""
    result = reader.read_predicted_shifts(shiftx2_file)

    expected = pd.DataFrame({
        "Res_N": [1, 1, 2, 2],
        "Res_type": ["ALA", "ALA", "GLY", "GLY"],
        "Atom_type": ["CA", "CB", "CA", "N"],
        "Shift": [52.1, 19.3, 45.2, 109.5],
    })

    assert_frame_equal(result, expected)


def test_read_predicted_shifts_single_chain(reader, tmp_path):
    """Test that a single CHAIN column is removed."""
    file_path = tmp_path / "single_chain.cs"

    pd.DataFrame({
        "CHAIN": ["A", "A"],
        "NUM": [1, 2],
        "RES": ["ALA", "GLY"],
        "ATOMNAME": ["CA", "CA"],
        "SHIFT": [52.1, 45.2],
    }).to_csv(file_path, index=False)

    result = reader.read_predicted_shifts(file_path)

    assert "CHAIN" not in result.columns
    assert list(result.columns) == [
        "Res_N", "Res_type", "Atom_type", "Shift"
    ]
    assert len(result) == 2


def test_read_predicted_shifts_multiple_chains(
    reader, tmp_path, capsys
):
    """Test multiple chains are merged and a warning is logged."""
    file_path = tmp_path / "multiple_chains.cs"

    pd.DataFrame({
        "CHAIN": ["A", "B"],
        "NUM": [1, 1],
        "RES": ["ALA", "GLY"],
        "ATOMNAME": ["CA", "CA"],
        "SHIFT": [52.1, 45.2],
    }).to_csv(file_path, index=False)

    result = reader.read_predicted_shifts(file_path)

    captured = capsys.readouterr()

    assert "CHAIN" not in result.columns
    assert len(result) == 2
    assert (
        "Chain identifier dropped" in captured.out
        or "Chain identifier dropped" in captured.err
    )


def test_read_predicted_shifts_extra_columns(reader, tmp_path):
    """Test that unexpected columns are removed."""
    file_path = tmp_path / "extra_columns.cs"

    pd.DataFrame({
        "NUM": [1],
        "RES": ["ALA"],
        "ATOMNAME": ["CA"],
        "SHIFT": [52.1],
        "EXTRA": ["unused"],
    }).to_csv(file_path, index=False)

    result = reader.read_predicted_shifts(file_path)

    assert list(result.columns) == [
        "Res_N", "Res_type", "Atom_type", "Shift"
    ]
    assert "EXTRA" not in result.columns


def test_read_predicted_shifts_missing_column(reader, tmp_path):
    """Test missing columns are filled with NaN."""
    file_path = tmp_path / "missing_column.cs"

    pd.DataFrame({
        "NUM": [1],
        "RES": ["ALA"],
        "ATOMNAME": ["CA"],
    }).to_csv(file_path, index=False)

    result = reader.read_predicted_shifts(file_path)

    assert pd.isna(result.loc[0, "Shift"])
    assert list(result.columns) == [
        "Res_N", "Res_type", "Atom_type", "Shift"
    ]


def test_read_predicted_shifts_nonexistent_file(reader, tmp_path):
    """Test that a missing file raises FileNotFoundError."""
    file_path = tmp_path / "nonexistent.cs"

    with pytest.raises(FileNotFoundError):
        reader.read_predicted_shifts(file_path)


def test_read_predicted_shifts_empty_file(reader, tmp_path):
    """Test that an empty file raises EmptyDataError."""
    file_path = tmp_path / "empty.cs"
    file_path.touch()

    with pytest.raises(pd.errors.EmptyDataError):
        reader.read_predicted_shifts(file_path)
