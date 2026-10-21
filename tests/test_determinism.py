import os
import sys
import subprocess
import pickle
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules.assigner_functions import (
    prepare_obs_preds,
    calc_log_prob_matrix,
    find_best_assignment,
    make_assign_df,
    import_predicted_shifts,
)
from modules.importer_functions import import_observed_shifts


TEST_DATA = Path(__file__).parent / "test_data"

CASES = [
    ("shiftx2.cs", "P3a_L273R.nef", "shiftx2", "nef"),
    ("P3a_L273R_sparta.tab", "ccpn_resonances.txt", "sparta+", "ccpn"),
    ("P3a_L273R_sparta.tab", "nmrpipe_shifts.tab", "sparta+", "nmrpipe"),
    ("shiftx2.cs", "sparky_shifts.txt", "shiftx2", "sparky"),
]


def run_assignment(pred_file, obs_file, pred_format, obs_format):
    logger = logging.getLogger("test_SNAPSimporter")

    obs_init = import_observed_shifts(
        filename=TEST_DATA / obs_file,
        filetype=obs_format,
    )

    preds_init, _ = import_predicted_shifts(
        filename=TEST_DATA / pred_file,
        filetype=pred_format,
        logger=logger,
    )

    obs_prep, preds_prep = prepare_obs_preds(obs_init, preds_init, logger)

    log_prob_matrix = calc_log_prob_matrix(
        obs=obs_prep,
        preds=preds_prep,
        pred_correction=False,
        delta_correlation=True,
        delta_correlation_mean_corrected_file="config/dd_mean.csv",
        delta_correlation_cov_corrected_file="config/dd_cov.csv",
        delta_correlation_mean_file="config/d_mean.csv",
        delta_correlation_cov_file="config/d_cov.csv",
        logger=logger,
    )

    matching = find_best_assignment(
        log_prob_matrix,
        maximise=True,
        logger=logger,
    )

    assign_df = make_assign_df(
        obs=obs_prep,
        preds=preds_prep,
        log_prob_matrix=log_prob_matrix,
        matching=matching,
        logger=logger,
    )

    return (os.getpid(),
            hash("test"), # This hash will change depending on the hash seed
            obs_init, preds_init, obs_prep, preds_prep, log_prob_matrix, assign_df)

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal


CASES = [
    ("shiftx2.cs", "P3a_L273R.nef", "shiftx2", "nef"),
    ("P3a_L273R_sparta.tab", "ccpn_resonances.txt", "sparta+", "ccpn"),
    ("P3a_L273R_sparta.tab", "nmrpipe_shifts.tab", "sparta+", "nmrpipe"),
    ("shiftx2.cs", "sparky_shifts.txt", "shiftx2", "sparky"),
]


@pytest.mark.parametrize("case", CASES)
def test_assignment_determinism(case):
    with mp.get_context("spawn").Pool(
        processes=8,
        maxtasksperchild=1,
    ) as pool:
        results = pool.map(
            run_assignment_worker,
            [case] * 100,
            chunksize=1,
        )

    pids = [pid for pid, _, _, _, _, _, _, _ in results]
    hashes = [h for _, h, _, _, _, _, _, _ in results]

    assert len(set(pids)) == 100
    assert len(set(hashes)) == 100

    _, _, reference_obs_init, reference_preds_init, reference_obs_prep, reference_preds_prep, reference_matrix, reference_assignment = results[0]

    for i, (_, _, obs_init, preds_init, obs_prep, preds_prep, matrix, assignment) in enumerate(results[1:], start=1):

        assert_frame_equal(
            reference_obs_init,
            obs_init,
            check_exact=True,
        )

        assert_frame_equal(
            reference_preds_init,
            preds_init,
            check_exact=True,
        )

        assert_frame_equal(
            reference_obs_prep,
            obs_prep,
            check_exact=True,
        )

        assert_frame_equal(
            reference_preds_prep,
            preds_prep,
            check_exact=True,
        )

        assert_frame_equal(
            matrix,
            reference_matrix,
            check_exact=True,
        )

        assert_frame_equal(
            assignment,
            reference_assignment,
            check_exact=True,
        )


def run_assignment_worker(case):
    return run_assignment(*case)