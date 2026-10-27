import subprocess
import sys
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_DATA = PROJECT_ROOT / "data" / "P3a_L273R"


def run_alex_pipeline(output_file):
    """Run the legacy Alex pipeline in an isolated process."""
    sys.path.insert(0, str(PROJECT_ROOT / "python" / "alex-dev" / "python"))

    from SNAPS_assigner import SNAPS_assigner
    from SNAPS_importer import SNAPS_importer

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

    assigner.aligned_preds["Pred_N"] = assigner.aligned_preds["Res_N"]
    assigner.aligned_preds["Pred_name"] = assigner.aligned_preds["Res_name"]

    for frame in (assigner.obs, assigner.aligned_preds):
        for column in ("C_m1", "CA_m1", "CB_m1"):
            if column not in frame:
                frame[column] = float("nan")

    assigner.prepare_obs_preds()
    assigner.obs.index.name = None
    assigner.aligned_preds.index.name = None

    assigner.prob_matrix = assigner.calc_prob_matrix(normalise_by="SS")
    assigner.log_prob_matrix = assigner.calc_log_prob_matrix()
    assigner.calc_mismatch_matrix()
    assigner.assign_from_preds(set_assign_df=True)
    assigner.add_consistency_info(
        threshold=assigner.pars["seq_link_threshold"]
    )

    return assigner.assign_df


def test_alex(tmp_path):
    results = []

    for i in range(2):
        output_file = tmp_path / f"result{i}.txt"
        dataframe_file = tmp_path / f"result{i}.pkl"

        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--run-pipeline",
                str(output_file),
                str(dataframe_file),
            ],
            cwd=PROJECT_ROOT,
            check=True,
            timeout=120,
        )

        results.append(pd.read_pickle(dataframe_file))

    assert_frame_equal(results[0], results[1])


if __name__ == "__main__" and len(sys.argv) == 4:
    if sys.argv[1] == "--run-pipeline":
        output_file = Path(sys.argv[2])
        dataframe_file = Path(sys.argv[3])

        result = run_alex_pipeline(output_file)
        result.to_pickle(dataframe_file)