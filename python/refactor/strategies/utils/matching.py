import numpy as np
import pandas as pd
from refactor.SNAPSconfig import SNAPSConfig
from scipy.optimize import linear_sum_assignment


def find_best_assignment(
    config: SNAPSConfig,
    score_matrix,
    maximise=True,
    inc=None,
    exc=None,
    dummy_rows=None,
    dummy_cols=None,
    return_none_all_dummy=False,
):
    """Use the Hungarian algorithm to find the highest scoring assignment,
    with constraints. Generalised so it can be used for assigning either
    sequentially or based on predictions.

    Returns a data frame with the index and column names of the matching.

    Parameters
    score_matrix: a pandas dataframe containing the scores, and with row and
        column labels
    inc: a DataFrame of (row, col) pairs which must be part of the assignment.
        First column has the index names, second has the column names.
    exc: a DataFrame of (row, col) pairs which may not be part of the assignment.
    """
    score_matrix = score_matrix.copy()

    row_name = score_matrix.index.name
    col_name = score_matrix.columns.name

    config.logger.info("Started linear assignment")

    if inc is not None:
        # Check for conflicting entries in inc
        conflicts = inc[row_name].duplicated(keep=False) | inc[col_name].duplicated(
            keep=False
        )
        if any(conflicts):
            config.logger.warning(
                "Warning: entries in inc conflict with one another - dropping conflicts"
            )
            # print(inc[conflicts])
            inc = inc[~conflicts]

        if exc is not None:
            # Check constraints are consistent
            # Get rid of any entries in exc which share a Res or SS with inc
            exc_in_inc = exc[row_name].isin(inc[row_name]) | exc[col_name].isin(
                inc[col_name]
            )
            if any(exc_in_inc):
                config.logger.warning(
                    "Some values in exc are also found in inc, so are redundant."
                )
                # print(exc[exc_in_inc])
                exc = exc.loc[~exc_in_inc, :]

        # Removed fixed assignments from probability matrix and obs, preds
        # dataframes. Latter is needed if inc includes any dummy SS/res,
        # and to detect if the reduced data is entirely dummies
        score_matrix_reduced = score_matrix.drop(index=inc[row_name]).drop(
            columns=inc[col_name]
        )
        config.logger.info(
            "%d assignments were fixed, %d remain to be assigned"
            % (len(inc), len(score_matrix_reduced.index))
        )
    else:
        score_matrix_reduced = score_matrix

    # If the matrix consists entirely of dummy rows or columns, the
    # assignment will not be meaningful. In this case, may want to return None
    if return_none_all_dummy:
        if set(score_matrix_reduced.index).issubset(dummy_rows) or set(
            score_matrix_reduced.columns
        ).issubset(dummy_cols):
            config.logger.debug("Score matrix includes only dummy rows/columns")
            return None

    if exc is not None:
        # Penalise excluded (row, col) pairs
        if maximise:
            penalty = -2 * score_matrix.abs().max().max()
        else:
            penalty = 2 * score_matrix.abs().max().max()

        for (
            i,
            r,
        ) in exc.iterrows():  # iterates over (index, row as pd.Series) tuples
            # If one side of an exclude pair is a dummy row or column,
            # exclude *all* dummy rows and columns
            if r[row_name] in dummy_rows:
                score_matrix_reduced.loc[r[row_name], dummy_cols] = penalty
            elif r[col_name] in dummy_cols:
                score_matrix_reduced.loc[dummy_rows, r[col_name]] = penalty
            else:
                score_matrix_reduced.loc[r["SS_name"], r["Res_name"]] = penalty

        config.logger.info("Penalised %d excluded row,column pairs" % len(exc.index))

    if maximise:
        row_ind, col_ind = linear_sum_assignment(-1 * score_matrix_reduced)
        # -1 because linear_sum_assignment() minimises sum, but we want to
        # maximise it.
    else:
        row_ind, col_ind = linear_sum_assignment(score_matrix_reduced)

    dummy_columns = {
        index
        for index, column in enumerate(score_matrix_reduced.columns)
        if str(column).strip().startswith("DR_")
    }
    assignments = dict(zip(row_ind, col_ind))
    for dummy_row, dummy_column in list(assignments.items()):
        if dummy_column not in dummy_columns:
            continue
        for real_row, real_column in assignments.items():
            if real_row <= dummy_row or real_column in dummy_columns:
                continue
            current = score_matrix_reduced.iloc[dummy_row, dummy_column]
            current += score_matrix_reduced.iloc[real_row, real_column]
            swapped = score_matrix_reduced.iloc[dummy_row, real_column]
            swapped += score_matrix_reduced.iloc[real_row, dummy_column]
            if np.isclose(current, swapped, rtol=0, atol=1e-8):
                assignments[dummy_row], assignments[real_row] = (
                    real_column,
                    dummy_column,
                )
                break
    row_ind = np.array(sorted(assignments))
    col_ind = np.array([assignments[row] for row in row_ind])

    # Construct results dataframe
    matching_reduced = pd.DataFrame(
        {
            row_name: score_matrix_reduced.index[row_ind],
            col_name: score_matrix_reduced.columns[col_ind],
        }
    )

    if inc is not None:
        matching = pd.concat([inc, matching_reduced])
        return matching
    else:
        return matching_reduced
