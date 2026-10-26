import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

# Keeps original project path
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "alex-dev" / "python"))

from SNAPS_assigner import SNAPS_assigner
from SNAPS_importer import SNAPS_importer

TEST_DATA = PROJECT_ROOT / "test_data"
REAL_DATA = PROJECT_ROOT / "data" / "P3a_L273R"
EXPECTED_ASSIGNMENT_HEAD = [
    ("28I", "283Val"),
    ("29P", "237Ala"),
    ("30N", "29P"),
    ("31T", "239Thr"),
    ("32D", "240Asp"),
]
EXPECTED_ASSIGNMENT_TAIL = [
    ("103E", "311Glu"),
    ("104R", "312Arg"),
    ("105K", "313Lys"),
    ("DR_1", "280Pro"),
    ("DR_2", "238Met"),
]


def run_alex_pipeline(output_file):
    """Run the legacy Alex pipeline and write its user-facing shift output."""
    assigner = SNAPS_assigner()
    assigner.read_config_file(PROJECT_ROOT / "config" / "config_yaml_2.txt")

    importer = SNAPS_importer()
    assigner.obs = importer.import_obs_shifts(
        REAL_DATA / "ccpn_resonances.txt", "ccpn", SS_num=True
    )
    assigner.preds = assigner.import_pred_shifts(
        REAL_DATA / "shiftx2.cs", "shiftx2"
    )
    assigner.aligned_preds = assigner.preds.copy()

    # These columns are part of the old assigner's prepared-data contract.
    assigner.aligned_preds["Pred_N"] = assigner.aligned_preds["Res_N"]
    assigner.aligned_preds["Pred_name"] = assigner.aligned_preds["Res_name"]
    for frame in (assigner.obs, assigner.aligned_preds):
        for column in ("C_m1", "CA_m1", "CB_m1"):
            if column not in frame:
                frame[column] = float("nan")

    assigner.prepare_obs_preds()
    # pandas now rejects merge keys that are both an index level and a column.
    assigner.obs.index.name = None
    assigner.aligned_preds.index.name = None
    assigner.prob_matrix = assigner.calc_prob_matrix(normalise_by="SS")
    assigner.log_prob_matrix = assigner.calc_log_prob_matrix()
    assigner.calc_mismatch_matrix()
    assigner.assign_from_preds(set_assign_df=True)
    assigner.add_consistency_info(
        threshold=assigner.pars["seq_link_threshold"]
    )
    assigner.output_shiftlist(output_file, "sparky")
    return assigner.assign_df


def test_alex(tmp_path):
    # This is a stupid test but just making sure the pipeline works
    result1 = run_alex_pipeline('tests/result1.txt')
    result2 = run_alex_pipeline('tests/result2.txt')

    assert_frame_equal(result1, result2)