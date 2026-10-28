import subprocess
import sys
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_DATA = PROJECT_ROOT / "data" / "P3a_L273R"

import shutil

@pytest.fixture
def config_file(tmp_path):
    data = f"""
# SNAPS configuration file
# Follows the YAML format
# Use spaces for indentation, not tabs!
atom_set: # Which atom types to include. Don't change indentation.
    - H
    - N
    - HA
    - CA
    - CB
    - C
    - CA_m1
    - CB_m1
    - C_m1
atom_sd: # Atom standard deviations. Don't change indentation.
    H: 0.454
    N: 2.429
    HA: 0.227
    C: 1.030
    CA: 0.932
    CB: 1.025
    C_m1: 1.030
    CA_m1: 0.932
    CB_m1: 1.025
iterate_until_consistent:       False   # If True, iteratively enforce consistent links for High and Medium confidence assignments
seq_link_threshold:             0.2     # Maximum allowed carbon shift difference (in ppm) between adjacent residues to count as a good sequential link.
delta_correlation:              True    # Account for correlations in prediction errors
delta_correlation_mean_file:     {PROJECT_ROOT / "config" / "d_mean.csv"}       # File containing mean prediction errors
delta_correlation_cov_file:      {PROJECT_ROOT / "config" / "d_cov.csv"}        # File containing covariances between the prediction errors
pred_correction:                False   # Apply a linear correction to the predicted shifts
pred_correction_file:           lin_model_shiftx2.csv    # File containing parameters for linear correction to predicted shift
delta_correlation_mean_corrected_file:     dd_mean.csv      # File containing mean prediction errors, assuming the predictions have been corrected
delta_correlation_cov_corrected_file:      dd_cov.csv       # File containing covariances between the prediction errors, assuming the predictions have been corrected
use_generic_predictions:        True                        # If True, missing predictions are replaced with an average value for that residue type
generic_shift_file:             {PROJECT_ROOT / "config" / "residue_shift_mean.csv"}      # File containing expected average chemical shift for each atom/residue pair
generic_shift_stdev_file:       {PROJECT_ROOT / "config" / "residue_shift_stdev.csv"}     # File containing expected standard deviation in chemical shift for each atom/residue pair
triplet_prob_iterations:        5
#May want to add more parameters to control generation of alternative assignments
alt_assignments:                0       # Number of alternative assignments to generate
use_SS_class_info:              False   # Whether to use information on amino acid type, if available

    """

    with open(tmp_path / 'config.yaml', 'w') as f:
        f.write(data)

    return tmp_path / 'config.yaml'


def test_alex_determinism(tmp_path, config_file):
    results = []

    for run in range(2):
        output_dir = tmp_path / f"alex_run_{run}"

        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "python" / "alex-dev" / "python" / "SNAPS.py"),
                str(REAL_DATA / "naps_shifts.txt"), # Only data with i - 1
                str(REAL_DATA / "P3a_L273R_sparta.tab"), # SHIFTX2 DOES NOT HAVE NECESSARY SHIFTS
                str(output_dir),
                "--shift_type", "snaps",
                "--pred_type", "sparta+",
                "--seq_file", str(REAL_DATA / "P3a_L273R_clean.fasta"),
                "-c", str(config_file),
            ],
            cwd=PROJECT_ROOT,
            check=True,
            timeout=120,
        )

        results.append(output_dir)

    for iteration in range(1, 6):
        filename = f"triplet_assign_df_iteration_{iteration}.tsv"

        df1 = pd.read_csv(results[0] / filename, sep="\t")
        df2 = pd.read_csv(results[1] / filename, sep="\t")

        assert_frame_equal(
            df1,
            df2,
            check_exact=False,
            rtol=1e-10,
            atol=1e-12,
            obj=f"Triplet iteration {iteration}",
        )