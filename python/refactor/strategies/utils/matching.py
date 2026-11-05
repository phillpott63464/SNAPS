import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def find_best_assignment(
    config,
    score_matrix,
    maximise=True,
    inc=None,
    exc=None,
    dummy_rows=None,
    dummy_cols=None,
    return_none_all_dummy=False,
    exc_mask=None,
):
    """Use the Hungarian algorithm to find the best assignment with constraints.

    Args:
        config: Configuration object containing the logger and strategy.
        score_matrix: DataFrame of scores, with row and column labels.
        maximise: If True, maximize the total score; otherwise minimize it.
        inc: DataFrame of (row, column) pairs that must be assigned.
        exc: DataFrame of (row, column) pairs that must not be assigned.
        dummy_rows: Row labels that represent dummy rows.
        dummy_cols: Column labels that represent dummy columns.
        return_none_all_dummy: Return None if the reduced matrix contains only
            dummy rows or only dummy columns.
        exc_mask: Boolean DataFrame aligned with score_matrix. True entries
            are excluded from the assignment.
    """
    dummy_rows = [] if dummy_rows is None else dummy_rows
    dummy_cols = [] if dummy_cols is None else dummy_cols

    row_name = score_matrix.index.name
    col_name = score_matrix.columns.name

    config.logger.info("Started linear assignment")

    score_matrix = score_matrix.copy()

    if maximise:
        penalty = -2 * score_matrix.abs().max().max()
    else:
        penalty = 2 * score_matrix.abs().max().max()

    if exc_mask is not None:
        score_matrix = score_matrix.mask(exc_mask, other=penalty)

    if inc is not None:
        conflicts = (
            inc[row_name].duplicated(keep=False)
            | inc[col_name].duplicated(keep=False)
        )
        if conflicts.any():
            config.logger.warning(
                "Warning: entries in inc conflict with one another - "
                "dropping conflicts"
            )
            inc = inc.loc[~conflicts]

        if exc is not None:
            exc_in_inc = (
                exc[row_name].isin(inc[row_name])
                | exc[col_name].isin(inc[col_name])
            )
            if exc_in_inc.any():
                config.logger.warning(
                    "Some values in exc are also found in inc, so are redundant."
                )
                exc = exc.loc[~exc_in_inc, :]

        score_matrix_reduced = (
            score_matrix.drop(index=inc[row_name])
            .drop(columns=inc[col_name])
        )
        config.logger.info(
            "%d assignments were fixed, %d remain to be assigned"
            % (len(inc), len(score_matrix_reduced.index))
        )
    else:
        score_matrix_reduced = score_matrix.copy()

    if return_none_all_dummy:
        rows_are_dummy = set(score_matrix_reduced.index).issubset(dummy_rows)
        cols_are_dummy = set(score_matrix_reduced.columns).issubset(dummy_cols)
        if rows_are_dummy or cols_are_dummy:
            config.logger.debug("Score matrix includes only dummy rows/columns")
            return None

    if exc is not None:
        if maximise:
            penalty = -2 * score_matrix.abs().max().max()
        else:
            penalty = 2 * score_matrix.abs().max().max()

        for _, excluded in exc.iterrows():
            row = excluded[row_name]
            col = excluded[col_name]

            if col in dummy_cols:
                score_matrix_reduced.loc[row, dummy_cols] = penalty
            elif row in dummy_rows:
                score_matrix_reduced.loc[dummy_rows, col] = penalty
            else:
                score_matrix_reduced.loc[row, col] = penalty

        config.logger.info(
            "Penalised %d excluded row,column pairs" % len(exc.index)
        )

    if maximise:
        row_ind, col_ind = linear_sum_assignment(-1 * score_matrix_reduced)
    else:
        row_ind, col_ind = linear_sum_assignment(score_matrix_reduced)

    # Preserve the refactor's deterministic tie-break for non-triplet strategies.
    if config.strategy != "triplet":
        dummy_column_indices = {
            index
            for index, column in enumerate(score_matrix_reduced.columns)
            if str(column).strip().startswith("DR_")
        }
        assignments = dict(zip(row_ind, col_ind))

        for dummy_row, dummy_column in list(assignments.items()):
            if dummy_column not in dummy_column_indices:
                continue

            for real_row, real_column in assignments.items():
                if real_row <= dummy_row or real_column in dummy_column_indices:
                    continue

                current = (
                    score_matrix_reduced.iloc[dummy_row, dummy_column]
                    + score_matrix_reduced.iloc[real_row, real_column]
                )
                swapped = (
                    score_matrix_reduced.iloc[dummy_row, real_column]
                    + score_matrix_reduced.iloc[real_row, dummy_column]
                )

                if np.isclose(current, swapped, rtol=0, atol=1e-8):
                    assignments[dummy_row], assignments[real_row] = (
                        real_column,
                        dummy_column,
                    )
                    break

        row_ind = np.array(sorted(assignments))
        col_ind = np.array([assignments[row] for row in row_ind])

    matching_reduced = pd.DataFrame(
        {
            row_name: score_matrix_reduced.index[row_ind],
            col_name: score_matrix_reduced.columns[col_ind],
        }
    )

    if inc is not None:
        return pd.concat([inc, matching_reduced]).reset_index(drop=True)

    return matching_reduced.reset_index(drop=True)