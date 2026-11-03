
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from SNAPS import run_snaps
from refactor.SNAPS import run_pipeline
from refactor.SNAPSconfig import build_config


CONFIG_FILE = Path("config/config_yaml_2.txt")
DATA_DIR = Path("data/P3a_L273R")


@pytest.mark.parametrize(
    "measured_shifts, shift_type",
    [
        ("ccpn_resonances.txt", "ccpn"),
        ("nmrpipe_shifts.tab", "nmrpipe"),
        ("P3a_L273R.nef", "nef"),
        ("sparky_shifts.txt", "sparky"),
    ],
    ids=["ccpn", "nmrpipe", "nef", "sparky"],
)
@pytest.mark.parametrize(
    "expected_shifts, prediction_type",
    [
        ("shiftx2.cs", "shiftx2"),
        ("P3a_L273R_sparta.tab", "sparta+"),
    ],
    ids=["shiftx2", "sparta"],
)
def test_run_snaps(
    tmp_path,
    measured_shifts,
    shift_type,
    expected_shifts,
    prediction_type,
):
    measured_shifts = DATA_DIR / measured_shifts
    expected_shifts = DATA_DIR / expected_shifts

    out_file_legacy = tmp_path / "out_legacy.txt"
    out_file_refactor = tmp_path / "out_refactor.txt"
    log_file = tmp_path / "log.txt"

    run_snaps(
        [
            str(measured_shifts),
            str(expected_shifts),
            str(out_file_legacy),
            "-c", str(CONFIG_FILE),
            "-l", str(log_file),
            "--shift_type", shift_type,
            "--pred_type", prediction_type,
            "--out_type", "snaps",
        ]
    )

    config = build_config(
        config_file=CONFIG_FILE,
        cli_overrides={
            "obs_file": measured_shifts,
            "pred_file": expected_shifts,
            "output_file": out_file_refactor,
        },
    )

    run_pipeline(config)

    """
    UPDATED
    Rather than checking that the output of the refactor is exactly the same as the output of the original,
    we now check that the output of the refactor does not differ from the original except in additional fields. 
    So:
    - Assignment changes, fail
    - Residue name changes, fail
    - Extra column in the refactor that wasn't present in the original, pass 
    """
    legacy = pd.read_csv(out_file_legacy, sep=r"\s+")
    refactor = pd.read_csv(out_file_refactor, sep=r"\s+")

    assert set(legacy.columns).issubset(refactor.columns)

    legacy = legacy.set_index("Res_name")
    refactor = refactor.set_index("Res_name")

    assert legacy.index.is_unique
    assert refactor.index.is_unique
    assert legacy.index.isin(refactor.index).all()

    refactor_subset = refactor.loc[legacy.index, legacy.columns]

    assert_frame_equal(
        refactor_subset,
        legacy,
        check_dtype=False,
        check_exact=False,
        rtol=1e-6,
        atol=1e-6,
    )
