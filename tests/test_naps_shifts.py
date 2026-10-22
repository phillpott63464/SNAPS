"""
Regression tests for the NAPS/SNAPS internal shift-list format.
This is because these are the only test data I can find with sequential links.
However, outputting is broken with sequential links so this is as far as I can test
"""

from pathlib import Path

from pandas.testing import assert_frame_equal

from modules.importer_functions import import_observed_shifts
from refactor.SNAPSimporter import SNAPSImporter


TEST_DATA = Path(__file__).parent / "test_data"
NAPS_SHIFTS = Path(__file__).parents[1] / "data/P3a_L273R/naps_shifts.txt"


def test_naps_shift_list_contains_sequential_backbone_shifts():
    observed = import_observed_shifts(NAPS_SHIFTS, "snaps")

    assert observed.shape == (103, 8)
    assert set(["C_m1", "CA_m1", "CB_m1"]).issubset(observed.columns)
    assert observed[["C_m1", "CA_m1", "CB_m1"]].notna().sum().to_dict() == {
        "C_m1": 96,
        "CA_m1": 83,
        "CB_m1": 78,
    }
    assert observed.loc[" 237AlaH", "CA_m1"] == 62.837
    assert observed.loc[" 237AlaH", "C_m1"] == 176.443


def test_refactored_importer_preserves_naps_sequential_shifts():
    expected = import_observed_shifts(NAPS_SHIFTS, "snaps")
    imported = SNAPSImporter().import_data(
        obs_file=NAPS_SHIFTS,
        pred_file=TEST_DATA / "shiftx2.cs",
    )

    assert_frame_equal(imported.obs_shifts, expected)
