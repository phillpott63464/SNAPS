
import logging

import pandas as pd
import pytest
from bokeh.client.states import ErrorReason
from pandas.testing import assert_frame_equal
from pathlib import Path

from modules.format_readers import ShiftX2Reader, SpartaReader


@pytest.fixture
def shiftx2reader():
    return ShiftX2Reader()

@pytest.fixture
def spartareader():
    return SpartaReader()

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

@pytest.fixture
def sparta_file(tmp_path):
    file_path = tmp_path / "sparta.txt"

    data = """
REMARK SPARTA+ Protein Chemical Shift Prediction Table
REMARK  
REMARK  All chemical shifts are reported in ppm:
REMARK  
REMARK  SHIFT    is the predicted chemical shift.
REMARK  SS_SHIFT is the predicted secondary chemical shift.
REMARK  RC_SHIFT is the random coil chemical shift.
REMARK  HM_SHIFT is the ring current shift; a correction
REMARK           of 0.6 times this value is applied.
REMARK  EF_SHIFT is the adjustment for electric field effect,
REMARK           applied to proton shifts only.
REMARK  SIGMA    is the estimated prediction error.
REMARK  
REMARK Reference:
REMARK  Y. Shen and A. Bax: SPARTA+: a modest improvement in empirical NMR
REMARK  chemical shift prediction by means of an artificial neural network.
REMARK  J. Biomol. NMR 48, 13-22 (2010).

DATA SOURCE P3a_L273R_clement_model.pdb

DATA FIRST_RESID 28

DATA SEQUENCE IPNTDYIQLL SEIAKEQGFN ITYLDIDELS ANGQYQCRAE LSTSPITVCH 
DATA SEQUENCE GSGISCGNAQ SDAAHNALQY LKIIXERK

VARS   RESID RESNAME ATOMNAME SS_SHIFT SHIFT RC_SHIFT HM_SHIFT EF_SHIFT SIGMA 
FORMAT %4d %4s %4s %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f

  28    I   HA     0.000     4.204     4.170     0.056     0.000     0.288
  28    I    C     0.000   176.144   176.117     0.045     0.000     1.063
  28    I   CA     0.000    61.340    61.300     0.068     0.000     1.566
  28    I   CB     0.000    38.028    38.000     0.046     0.000     1.214
    """

    with open(file_path, 'w') as f:
        f.write(data)

    return file_path

def test_format_name(shiftx2reader, spartareader):
    assert shiftx2reader.format_name == "shiftx2"
    assert spartareader.format_name == "sparta+"


def test_file_extensions(shiftx2reader, spartareader):
    assert shiftx2reader.file_extensions == [".cs", ".txt"]
    assert spartareader.file_extensions == [".txt"]


"""
This needs to be updated because can_read needs to be updated for everything - it should
go into the guts of the file to identify if it can actually read the file rather than
guessing from extensions. 

Subsequently, these tests then need to be ran with real files. 
"""
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
def test_can_read_shiftx2(shiftx2reader, filename, expected):
    assert shiftx2reader.can_read(Path(filename)) is expected


def test_read_observed_shifts_shiftx2(shiftx2reader, shiftx2_file):
    """ShiftX2 does not contain observed shifts."""
    assert shiftx2reader.read_observed_shifts(shiftx2_file) is None

def test_read_observed_shifts_sparta(spartareader, sparta_file):
    """sparta+ does not contain observed shifts."""
    assert spartareader.read_observed_shifts(sparta_file) is None

def test_read_predicted_shifts_shiftx2(shiftx2reader, shiftx2_file):
    """Test reading and renaming prediction columns."""
    result = shiftx2reader.read_predicted_shifts(shiftx2_file)

    expected = pd.DataFrame({
        "Res_N": [1, 1, 2, 2],
        "Res_type": ["ALA", "ALA", "GLY", "GLY"],
        "Atom_type": ["CA", "CB", "CA", "N"],
        "Shift": [52.1, 19.3, 45.2, 109.5],
    })

    assert_frame_equal(result, expected)

def test_read_predicted_shifts_sparta(spartareader, sparta_file):
    """Test reading and renaming prediction columns."""
    result = spartareader.read_predicted_shifts(sparta_file)

    expected = pd.DataFrame({
        "Res_N": [28, 28, 28, 28],
        "Res_type": ['I', 'I', 'I', 'I'],
        "Atom_type": ['HA', 'C', 'CA', 'CB'],
        "Shift": [4.204, 176.144, 61.340, 38.028],
    })

    assert_frame_equal(result, expected)

def test_read_predicted_shifts_single_chain(shiftx2reader, tmp_path):
    """Test that a single CHAIN column is removed. Shiftx2 only concern."""
    file_path = tmp_path / "single_chain.cs"

    pd.DataFrame({
        "CHAIN": ["A", "A"],
        "NUM": [1, 2],
        "RES": ["ALA", "GLY"],
        "ATOMNAME": ["CA", "CA"],
        "SHIFT": [52.1, 45.2],
    }).to_csv(file_path, index=False)

    result = shiftx2reader.read_predicted_shifts(file_path)

    assert "CHAIN" not in result.columns
    assert list(result.columns) == [
        "Res_N", "Res_type", "Atom_type", "Shift"
    ]
    assert len(result) == 2


def test_read_predicted_shifts_multiple_chains(
    shiftx2reader, tmp_path, capsys
):
    """
    Test multiple chains are merged and a warning is logged.
    Logging removed temporarily in favour for print.
    Shiftx2 only concern.
    """
    file_path = tmp_path / "multiple_chains.cs"

    pd.DataFrame({
        "CHAIN": ["A", "B"],
        "NUM": [1, 1],
        "RES": ["ALA", "GLY"],
        "ATOMNAME": ["CA", "CA"],
        "SHIFT": [52.1, 45.2],
    }).to_csv(file_path, index=False)

    result = shiftx2reader.read_predicted_shifts(file_path)

    captured = capsys.readouterr()

    assert "CHAIN" not in result.columns
    assert len(result) == 2
    assert (
        "Chain identifier dropped" in captured.out
        or "Chain identifier dropped" in captured.err
    )


def test_read_predicted_shifts_extra_columns(shiftx2reader, tmp_path):
    """Test that unexpected columns are removed."""
    file_path = tmp_path / "extra_columns.cs"

    pd.DataFrame({
        "NUM": [1],
        "RES": ["ALA"],
        "ATOMNAME": ["CA"],
        "SHIFT": [52.1],
        "EXTRA": ["unused"],
    }).to_csv(file_path, index=False)

    result = shiftx2reader.read_predicted_shifts(file_path)

    assert list(result.columns) == [
        "Res_N", "Res_type", "Atom_type", "Shift"
    ]
    assert "EXTRA" not in result.columns


def test_read_predicted_shifts_missing_column(shiftx2reader, tmp_path):
    """Test missing columns are filled with NaN."""
    file_path = tmp_path / "missing_column.cs"

    pd.DataFrame({
        "NUM": [1],
        "RES": ["ALA"],
        "ATOMNAME": ["CA"],
    }).to_csv(file_path, index=False)

    result = shiftx2reader.read_predicted_shifts(file_path)

    assert pd.isna(result.loc[0, "Shift"])
    assert list(result.columns) == [
        "Res_N", "Res_type", "Atom_type", "Shift"
    ]


def test_read_predicted_shifts_nonexistent_file(shiftx2reader, tmp_path):
    """Test that a missing file raises FileNotFoundError."""
    file_path = tmp_path / "nonexistent.cs"

    with pytest.raises(FileNotFoundError):
        shiftx2reader.read_predicted_shifts(file_path)


def test_read_predicted_shifts_empty_file(shiftx2reader, spartareader, tmp_path):
    """Test that an empty file raises EmptyDataError."""
    file_path = tmp_path / "empty.cs"
    file_path.touch()

    with pytest.raises(pd.errors.EmptyDataError):
        shiftx2reader.read_predicted_shifts(file_path)
    with pytest.raises(Exception):
        spartareader.read_predicted_shifts(file_path)