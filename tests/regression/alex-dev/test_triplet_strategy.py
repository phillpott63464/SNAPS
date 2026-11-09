import importlib
import sys
import subprocess
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from refactor.SNAPSconfig import build_config
from refactor.SNAPSpipeline import run_pipeline
from refactor.SNAPSimporter import SNAPSImporter
from refactor.SNAPSstrategy import create_strategy


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "P3a_L273R"
LEGACY = ROOT / "python" / "alex-dev" / "python"


def _config_file(tmp_path):
    config = f"""
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
delta_correlation_mean_file:     {ROOT / "config" / "d_mean.csv"}       # File containing mean prediction errors
delta_correlation_cov_file:      {ROOT / "config" / "d_cov.csv"}        # File containing covariances between the prediction errors
pred_correction:                False   # Apply a linear correction to the predicted shifts
pred_correction_file:           lin_model_shiftx2.csv    # File containing parameters for linear correction to predicted shift
delta_correlation_mean_corrected_file:     dd_mean.csv      # File containing mean prediction errors, assuming the predictions have been corrected
delta_correlation_cov_corrected_file:      dd_cov.csv       # File containing covariances between the prediction errors, assuming the predictions have been corrected
use_generic_predictions:        True                        # If True, missing predictions are replaced with an average value for that residue type
generic_shift_file:             {ROOT / "config" / "residue_shift_mean.csv"}      # File containing expected average chemical shift for each atom/residue pair
generic_shift_stdev_file:       {ROOT / "config" / "residue_shift_stdev.csv"}     # File containing expected standard deviation in chemical shift for each atom/residue pair
triplet_prob_iterations:        5
#May want to add more parameters to control generation of alternative assignments
alt_assignments:                0       # Number of alternative assignments to generate
use_SS_class_info:              False   # Whether to use information on amino acid type, if available

    """
    path = tmp_path / "config.yaml"
    path.write_text(config)
    return path


def _assert_frame(left, right, stage):
    left = left.copy()
    right = right.copy()
    left.index.name = right.index.name = None
    left.columns.name = right.columns.name = None
    print(stage)
    assert_frame_equal(
        left,
        right,
        check_dtype=False,
        check_names=True,
        check_exact=False,
        rtol=1e-10,
        atol=1e-12,
        obj=stage,
    )


def _legacy_trace(config_file):
    """Run the legacy stages while retaining every intermediate dataframe."""
    sys.path.insert(0, str(LEGACY))
    try:
        spec = importlib.util.spec_from_file_location(
            "_alex_legacy_SNAPS_assigner", LEGACY / "SNAPS_assigner.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError("Could not load the Alex legacy SNAPS_assigner module")
        legacy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy_module)
        legacy_assigner = legacy_module.SNAPS_assigner
        from modules.importer_functions import import_observed_shifts

        assigner = legacy_assigner()
        assigner.read_config_file(str(config_file))
        assigner.import_sequence(str(DATA / "P3a_L273R.fasta"), 1)
        predictions = assigner.import_pred_shifts_2(
            str(DATA / "P3a_L273R_sparta.tab"), "sparta+"
        )
        assigner.map_preds_to_sequence(assigner.seq_df, predictions)

        assigner.obs = import_observed_shifts(
            str(DATA / "naps_shifts.txt"), "snaps", SS_num=False
        )
        trace = {"raw_obs": assigner.obs.copy(), "raw_preds": assigner.aligned_preds.copy()}

        assigner.prepare_obs_preds()
        trace["obs"], trace["preds"] = (
            assigner.obs.copy(),
            assigner.aligned_preds.copy(),
        )
        assigner.prob_matrix = assigner.calc_prob_matrix(normalise_by="SS")
        if assigner.pars["use_generic_predictions"]:
            assigner.create_generic_predictions()
            trace["generic_preds"] = assigner.generic_preds.copy()
            assigner.generic_prob_matrix = assigner.calc_generic_prob_matrix()
            trace["generic_prob_matrix"] = assigner.generic_prob_matrix.copy()
            assigner.prob_matrix *= assigner.generic_prob_matrix
            assigner.prob_matrix = assigner.prob_matrix.div(
                assigner.prob_matrix.sum(axis=1), axis="rows"
            )
        for _ in range(20):
            assigner.prob_matrix = assigner.prob_matrix.div(
                assigner.prob_matrix.sum(axis=0), axis="columns"
            )
            assigner.prob_matrix = assigner.prob_matrix.div(
                assigner.prob_matrix.sum(axis=1), axis="rows"
            )
        trace["prob_matrix"] = assigner.prob_matrix.copy()
        assigner.log_prob_matrix = assigner.calc_log_prob_matrix()
        trace["log_prob_matrix"] = assigner.log_prob_matrix.copy()
        assigner.calc_mismatch_matrix()
        trace["mismatch_matrix"] = assigner.mismatch_matrix.copy()
        trace["consistent_links_matrix"] = assigner.consistent_links_matrix.copy()
        trace["allowed_links_matrix"] = assigner.allowed_links_matrix.copy()

        assign_df = assigner.assign_from_preds(set_assign_df=False)
        assign_df = assigner.add_consistency_info(assign_df, 0.2)
        trace["base_assign_df"] = assign_df.copy()
        trace["iterations"] = []
        prob = assigner.prob_matrix.copy()
        for _ in range(assigner.pars["triplet_prob_iterations"]):
            prob = assigner.calc_triplet_prob_matrix(prob, assigner.allowed_links_matrix, "SS")
            for _ in range(20):
                prob = prob.div(prob.sum(axis=0), axis="columns")
                prob = prob.div(prob.sum(axis=1), axis="rows")
            log_prob = assigner.calc_log_prob_matrix(prob)
            current = assigner.assign_from_preds(log_prob_matrix=log_prob)
            current = assigner.add_consistency_info(current, 0.2)
            trace["iterations"].append(
                {
                    "prob_matrix": prob.copy(),
                    "log_prob_matrix": log_prob.copy(),
                    "assign_df": current.copy(),
                    "matching": current[["SS_name", "Res_name"]].copy(),
                }
            )
        return trace
    finally:
        sys.path.remove(str(LEGACY))


def _refactor_trace(config):
    importer = SNAPSImporter(config=config)
    data = importer.import_data(
        obs_file=config.obs_file,
        pred_file=config.pred_file,
        obs_format="snaps",
        pred_format="sparta+",
    )
    strategy = create_strategy("triplet", config)
    obs, preds = strategy.preprocess_obs_preds(data.obs_shifts, data.pred_shifts)
    generic_preds = strategy.create_generic_predictions(preds, data.sequence)
    generic_prob = strategy.calc_generic_prob_matrix(
        obs, preds, generic_preds, data.sequence
    )
    prob = strategy.calc_prob_matrix(obs, preds, data.sequence)
    log_prob = strategy.calc_log_prob_matrix(prob)
    mismatch_matrix, consistent_links, allowed_links = (
        strategy.calc_mismatch_matrix(obs)
    )
    base_assign, base_matching = strategy.assign_from_preds(obs, preds, log_prob)
    base_assign, allowed = strategy.add_consistency_info(
        base_assign, obs, preds, config.seq_link_threshold
    )
    trace = {
        "raw_obs": data.obs_shifts,
        "raw_preds": data.pred_shifts,
        "obs": obs,
        "preds": preds,
        "generic_preds": generic_preds,
        "generic_prob_matrix": generic_prob,
        "prob_matrix": prob,
        "log_prob_matrix": log_prob,
        "mismatch_matrix": mismatch_matrix,
        "consistent_links_matrix": consistent_links,
        "allowed_links_matrix": allowed_links,
        "base_assign_df": base_assign,
        "allowed_links_matrix": allowed,
        "iterations": [],
    }
    for _ in range(config.triplet_prob_iterations):
        prob = strategy.calc_triplet_prob_matrix(prob, allowed, "SS")
        for _ in range(20):
            prob = prob.div(prob.sum(axis=0), axis="columns")
            prob = prob.div(prob.sum(axis=1), axis="rows")
        log_prob = strategy.calc_log_prob_matrix(prob)
        current, matching = strategy.assign_from_preds(obs, preds, log_prob)
        current, allowed = strategy.add_consistency_info(
            current, obs, preds, config.seq_link_threshold
        )
        trace["iterations"].append(
            {
                "prob_matrix": prob.copy(),
                "log_prob_matrix": log_prob.copy(),
                "assign_df": current.copy(),
                "matching": matching.copy(),
            }
        )
    return trace


def test_legacy_and_refactored_triplet_internal_states(tmp_path):
    config_file = _config_file(tmp_path)
    legacy_output = tmp_path / "legacy"
    subprocess.run(
        [
            sys.executable,
            str(LEGACY / "SNAPS.py"),
            str(DATA / "naps_shifts.txt"),
            str(DATA / "P3a_L273R_sparta.tab"),
            str(legacy_output),
            "--shift_type",
            "snaps",
            "--pred_type",
            "sparta+",
            "--seq_file",
            str(DATA / "P3a_L273R.fasta"),
            "-c",
            str(config_file),
        ],
        cwd=ROOT,
        check=True,
        timeout=120,
    )

    config = build_config(
        config_file,
        cli_overrides={
            "strategy": "triplet",
            "obs_file": DATA / "naps_shifts.txt",
            "pred_file": DATA / "P3a_L273R_sparta.tab",
            "sequence_file": DATA / "P3a_L273R.fasta",
            "output_file": tmp_path / "refactored.txt",
            "obs_format": "snaps",
            "pred_format": "sparta+",
        },
    )
    refactored_result = run_pipeline(config)
    legacy = _legacy_trace(config_file)
    refactored = _refactor_trace(config)

    for stage in (
        "raw_obs",
        "raw_preds",
        "obs",
        "preds",
        "generic_preds",
        "generic_prob_matrix",
        "prob_matrix",
        "log_prob_matrix",
    ):
        _assert_frame(refactored[stage], legacy[stage], stage)

    for stage in ("mismatch_matrix", "consistent_links_matrix", "allowed_links_matrix"):
        if stage in legacy and stage in refactored:
            _assert_frame(refactored[stage], legacy[stage], stage)
    _assert_frame(refactored["base_assign_df"], legacy["base_assign_df"], "base_assign_df")
    assert len(refactored_result.assign_dfs) == len(legacy["iterations"]) + 1
    for index, (actual, expected) in enumerate(
        zip(refactored["iterations"], legacy["iterations"]), 1
    ):
        for state in ("prob_matrix", "log_prob_matrix", "assign_df", "matching"):
            _assert_frame(actual[state], expected[state], f"iteration {index} {state}")
        written = pd.read_csv(
            legacy_output / f"triplet_assign_df_iteration_{index}.tsv", sep="\t"
        )
        _assert_frame(
            written,
            expected["assign_df"].round(3), # Parity rounding
            f"legacy output {index}",
        )
