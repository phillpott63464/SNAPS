# Tests for functions I have wrapped outside of SNAPS_assigner.py
# Basically making everything stateless for easier testing and to make the dataflow clearer rather than relying
# On previous operations being done in order on self.preds constantly
# This'll then make it easier to add custom dataclasses for data transfer in a legible way
import logging


import logging
import pytest
import pandas as pd
from modules.assigner_functions import import_predicted_shifts
from pathlib import Path

TEST_DATA = Path(__file__).parent / 'test_data'

@pytest.fixture
def logger():
    return logging.getLogger("test_predicted_shifts")

@pytest.mark.parametrize(
    "filetype",
    ["csv", "json", "shiftx", "SPARTA+", "", "banana", "tinkywinky"]
)
def test_invalid_filetypes(filetype, logger, caplog):
    """Only 'shiftx2' and 'sparta+' are supported."""
    with caplog.at_level(logging.ERROR):
        result = import_predicted_shifts(
            "dummy.csv", filetype, None, logger
        )

    assert result is None
    assert f"Invalid predicted shift type: '{filetype}'" in caplog.text
    assert "Allowed" in caplog.text
    assert "'shiftx2' or 'sparta+'" in caplog.text


@pytest.mark.parametrize("filetype", ["shiftx2", "sparta+"])
def test_missing_file(filetype, logger):
    """Missing input files should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError): # this is implicitly an assertion
        import_predicted_shifts(
            "this_file_does_not_exist.txt",
            filetype,
            None,
            logger,
        )


@pytest.mark.parametrize(
    "filetype, filename, snapshot",
    [
        ("shiftx2", "shiftx2.cs", "shiftx2_snaps_internal"),
        ("sparta+", "P3a_L273R_sparta.tab", "sparta_snaps_internal"),
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



