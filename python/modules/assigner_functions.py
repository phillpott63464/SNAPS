from typing import Optional

import pandas as pd
import logging
import numpy as np
from scipy.stats import norm, multivariate_normal
from math import log10
from scipy.optimize import linear_sum_assignment

def import_predicted_shifts(
    filename: str,
    filetype: str,
    seq_df: Optional[pd.DataFrame] = None,
    logger: logging.Logger = None,
    offset=0,
):
    """Import predicted chemical shifts from a ShiftX2 results file.

    Returns
    A DataFrame containing the predicted shifts, or None if the import failed.

    Parameters
    filename: path to file containing predicted shifts
    filetype: either "shiftx2" or "sparta+"
    offset: an optional integer to add to the residue number.
    """

    #### Import the raw data
    if filetype == 'shiftx2':
        preds_long = pd.read_csv(filename)
        if any(preds_long.columns == 'CHAIN'):
            if len(preds_long['CHAIN'].unique()) > 1:
                logger.warning(
                    """Chain identifier dropped - if multiple chains are
                        present in the predictions, they will be merged."""
                )
            preds_long = preds_long.drop('CHAIN', axis=1)
        preds_long = preds_long.reindex(
            columns=['NUM', 'RES', 'ATOMNAME', 'SHIFT']
        )
        preds_long.columns = ['Res_N', 'Res_type', 'Atom_type', 'Shift']
    elif filetype == 'sparta+':
        # Work out where the column names and data are
        with open(filename, 'r') as f:
            for num, line in enumerate(f, 1):
                if line.find('VARS') > -1:
                    colnames_line = num
                    colnames = line.split()[1:]
                    break

        preds_long = pd.read_table(
            filename, sep='\s+', names=colnames, skiprows=colnames_line + 1
        )
        preds_long = preds_long.reindex(
            columns=['RESID', 'RESNAME', 'ATOMNAME', 'SHIFT']
        )
        preds_long.columns = ['Res_N', 'Res_type', 'Atom_type', 'Shift']

        # Sparta+ uses HN for backbone amide proton - convert to H
        preds_long.loc[preds_long['Atom_type'] == 'HN', 'Atom_type'] = 'H'
    else:
        logger.error(
            """Invalid predicted shift type: '%s'. Allowed
                          options are 'shiftx2' or 'sparta+'"""
            % (filetype)
        )
        return None

    logger.info(
        'Imported %d predicted chemical shifts from %s'
        % (len(preds_long.index), filename)
    )

    #### Initial processing and conversion from long to wide
    # Add sequence number offset and create residue names
    preds_long['Res_N'] = preds_long['Res_N'] + offset
    preds_long.insert(
        1,
        'Res_name',
        (preds_long['Res_N'].astype(str) + preds_long['Res_type']),
    )
    # Left pad with spaces to a constant length (helps with sorting)
    preds_long['Res_name'] = preds_long['Res_name'].str.rjust(5)

    # Convert from long to wide format
    preds = preds_long.pivot(
        index='Res_N', columns='Atom_type', values='Shift'
    )
    preds.index.name = None

    # Add the other residue name and type back in
    tmp = preds_long[['Res_N', 'Res_type', 'Res_name']]
    tmp = tmp.drop_duplicates(subset='Res_name')
    tmp.index = tmp['Res_N']
    tmp.index.name = None
    preds = pd.concat([tmp, preds], axis=1)

    #### Make consistent with seq_df (and create if it doesn't already exist)
    # TODO: Maybe this should be split off into a separate function?
    # If seq_df is missing, create it based on preds
    if seq_df is None:
        seq_df = preds.copy()[['Res_N', 'Res_type', 'Res_name']]

        # If there are any missing residue numbers, create them
        min_N = seq_df['Res_N'].min()
        max_N = seq_df['Res_N'].max()
        missing_residue_numbers = set(range(min_N, max_N + 1)).difference(
            seq_df['Res_N']
        )
        if len(missing_residue_numbers) > 0:
            tmp = pd.DataFrame(
                {
                    'Res_N': list(missing_residue_numbers),
                    'Res_type': 'X',
                    'Res_name': np.NaN,
                }
            )
            tmp['Res_name'] = tmp['Res_N'].astype(str) + tmp['Res_type']
            tmp['Res_name'] = tmp['Res_name'].str.rjust(5)
            tmp.index = tmp['Res_N']
            tmp.index.name = None
            # seq_df = seq_df.append(tmp).sort_index()
            seq_df = pd.concat([seq_df, tmp]).sort_index()

    # Add/delete residues from preds so it matches seq_df
    # Only keep predictions that are in seq_df
    tmp = len(preds.index)
    preds = preds[preds['Res_N'].isin(seq_df['Res_N'])]
    tmp2 = tmp - len(preds.index)
    if tmp2 > 0:
        logger.info(
            (
                'Predictions for %d residues were discarded '
                'because they were not present in the imported '
                'sequence file'
            )
            % tmp2
        )

    # Add predictions for any residue number that is in seq_df but not preds
    tmp = pd.DataFrame(
        data=seq_df[~seq_df['Res_N'].isin(preds['Res_N'])],
        columns=preds.columns,
    )
    # preds = preds.append(tmp)
    preds = pd.concat([preds, tmp])
    if len(tmp.index) > 0:
        logger.info(
            (
                '%d residues from the sequence were missing '
                'from the predictions'
            )
            % len(tmp.index)
        )

    # If Res_name is inconsistent between seq_df and preds, make a compromise
    preds.index = preds['Res_N']
    preds = preds.sort_index()
    seq_df.index = seq_df['Res_N']
    seq_df = seq_df.sort_index()
    mask = preds['Res_name'] != seq_df['Res_name']

    if any(mask):
        ambiguous_res_names = (
            seq_df.loc[mask, 'Res_name']
            + '('
            + preds.loc[mask, 'Res_type']
            + '?)'
        )
        preds.loc[mask, 'Res_name'] = ambiguous_res_names
        seq_df.loc[mask, 'Res_name'] = ambiguous_res_names
        # Note we don't update the Res_type in preds, because this is potentially
        # used for correcting the chemical shift
        logger.warning(
            (
                'There were inconsistencies between the provided sequence '
                'file and the predicted shifts. The following residues had '
                'inconsistent amino acid types: %s'
            )
            % ', '.join(ambiguous_res_names)
        )

    #        preds.index = preds["Res_name"]
    #        preds.index.name = None
    seq_df.index = seq_df['Res_name']
    seq_df.index.name = None

    #### Add the chemical shift info back in

    # Make columns for the i-1 predicted shifts of C, CA and CB
    preds_m1 = preds[
        list(
            {'C', 'CA', 'CB', 'Res_type', 'Res_name'}.intersection(
                preds.columns
            )
        )
    ].copy()
    preds_m1.index = preds_m1.index + 1
    preds_m1.columns = preds_m1.columns + '_m1'
    preds = pd.merge(
        preds, preds_m1, how='left', left_index=True, right_index=True
    )

    # Make column for the i+1 Res_name
    preds_p1 = preds[['Res_name']].copy()
    preds_p1.index = preds_p1.index - 1
    preds_p1.columns = ['Res_name_p1']
    preds = pd.merge(
        preds, preds_p1, how='left', left_index=True, right_index=True
    )

    # Set index to Res_name
    preds.index = preds['Res_name']
    preds.index.name = None

    # Restrict to only certain atom types
    atom_set = {'H', 'N', 'C', 'CA', 'CB', 'C_m1', 'CA_m1', 'CB_m1', 'HA'}
    preds = preds[
        [
            'Res_name',
            'Res_N',
            'Res_type',
            'Res_name_m1',
            'Res_name_p1',
            'Res_type_m1',
        ]
        + list(atom_set.intersection(preds.columns))
    ]

    logger.info(
        'Finished reading in %d predicted residues from %s'
        % (len(preds.index), filename)
    )

    return preds, seq_df


def prepare_obs_preds(
    obs,
    preds,
    logger,
    atom_set={'H', 'N', 'HA', 'C', 'CA', 'CB', 'C_m1', 'CA_m1', 'CB_m1'},
):
    """Perform preprocessing on the observed and predicted shifts to prepare
    them for calculation of the log probability matrix.

    1) Remove proline residues from predictions
    2) Discard atom types that are not used or aren't present in both
    3) Add dummy rows to obs or preds to bring them to the same length

    Returns the modified obs and preds DataFrames.
    """

    #### Delete any prolines in preds
    logger.info(
        'Removing %d prolines from predictions' % sum(preds['Res_type'] == 'P')
    )
    preds = preds.drop(preds.index[preds['Res_type'] == 'P'])

    # Remove references to deleted residues from Res_name_m1/p1
    preds.loc[
        ~preds['Res_name_m1'].isin(preds['Res_name']), 'Res_name_m1'
    ] = np.NaN
    preds.loc[
        ~preds['Res_name_p1'].isin(preds['Res_name']), 'Res_name_p1'
    ] = np.NaN

    #### Restrict atom types
    # self.pars["atom_set"] is the set of atoms to be used in the analysis

    # Get all non-atom columns
    all_atoms = {'H', 'N', 'C', 'CA', 'CB', 'C_m1', 'CA_m1', 'CB_m1', 'HA'}
    obs_metadata = list(set(obs.columns).difference(all_atoms))
    preds_metadata = list(set(preds.columns).difference(all_atoms))
    # Get atoms that are in atom_set *and* present in both obs and preds
    shared_atoms = list(
        atom_set.intersection(obs.columns).intersection(preds.columns)
    )
    obs = obs.loc[:, obs_metadata + shared_atoms]
    preds = preds.loc[:, preds_metadata + shared_atoms]

    logger.info('Restricted atom types to %s' % ','.join(shared_atoms))

    #### Create dummy rows
    # Create columns to keep track of dummy status
    preds['Dummy_res'] = False
    obs['Dummy_SS'] = False

    N = len(obs.index)
    M = len(preds.index)

    if N > M:     # If there are more spin systems than predictions
        dummies = pd.DataFrame(
            np.NaN,
            columns=preds.columns,
            index=['DR_' + str(i) for i in 1 + np.arange(N - M)],
        )
        dummies['Res_name'] = dummies.index
        dummies['Dummy_res'] = True
        # preds = preds.append(dummies)
        preds = pd.concat([preds, dummies])
        logger.info('Added %d dummy predicted residues' % len(dummies.index))
    elif M > N:
        dummies = pd.DataFrame(
            np.NaN,
            columns=obs.columns,
            index=['DSS_' + str(i) for i in 1 + np.arange(M - N)],
        )
        dummies['SS_name'] = dummies.index
        dummies['Dummy_SS'] = True
        # obs = obs.append(dummies)
        obs = pd.concat([obs, dummies])
        logger.info('Added %d dummy observed residues' % len(dummies.index))

    return (obs, preds)


def calc_log_prob_matrix(
    obs: pd.DataFrame,
    preds: pd.DataFrame,
    pred_correction: bool = False,
    delta_correlation: bool = False,
    pred_correction_file=None,
    delta_correlation_mean_corrected_file=None,
    delta_correlation_cov_corrected_file=None,
    delta_correlation_mean_file=None,
    delta_correlation_cov_file=None,
    atom_sd=None,
    sf=1,
    default_prob=0.01,
    logger=None,
    atom_set={'H', 'N', 'HA', 'C', 'CA', 'CB', 'C_m1', 'CA_m1', 'CB_m1'},
):
    """Calculate a matrix of -log10(match probabilities).

    By probability, we mean the value of the probability density function,
    assuming the prediction errors follow a Gaussian distribution.
    If self.pars["delta_correlation"]==True, correlations in errors between
    different atom types will be accounted for.
    If self.pars["pred_correction"]==True, a linear correction will be
    applied to the predicted shifts to compensate for prediction bias
    towards random coil values.

    Returns
    A DataFrame containing the log probabilities

    Parameters
    atom_sd: A dictionary containing the expected standard error in the
        predictions for each atom type
    sf: A scale factor that multiplies the atom_sd
    default_prob: penalty for missing data
    """

    atoms = list(atom_set.intersection(obs.columns))

    if pred_correction:
        # Import parameters for correcting the shifts
        lm_pars = pd.read_csv(pred_correction_file, index_col=0)
        logger.info(
            'Imported pred_correction info from %s' % pred_correction_file
        )

    if delta_correlation:
        # Import parameters describing the delta correlations
        # Note: this also sorts the atom types in d_mean and c_cov into the
        # same order as the 'atoms' list defined above.
        if pred_correction:
            d_mean = pd.read_csv(
                delta_correlation_mean_corrected_file,
                header=None,
                index_col=0,
            ).loc[atoms, 1]
            d_cov = pd.read_csv(
                delta_correlation_cov_corrected_file, index_col=0
            ).loc[atoms, atoms]
            logger.info(
                'Imported delta_correlation info from %s and %s'
                % (
                    delta_correlation_mean_corrected_file,
                    delta_correlation_cov_corrected_file,
                )
            )
        else:
            d_mean = pd.read_csv(
                delta_correlation_mean_file,
                header=None,
                index_col=0,
            ).loc[atoms, 1]
            d_cov = pd.read_csv(delta_correlation_cov_file, index_col=0).loc[
                atoms, atoms
            ]
            logger.info(
                'Imported delta_correlation info from %s and %s'
                % (
                    delta_correlation_mean_file,
                    delta_correlation_cov_file,
                )
            )

        # For the delta_correlation method, we need to store the prediction
        # errors for each atom type for analysis at the end.
        delta_list = []

    log_prob_matrix = pd.DataFrame(0, index=obs.index, columns=preds.index)
    log_prob_matrix.index.name = 'SS_name'
    log_prob_matrix.columns.name = 'Res_name'

    for atom in atoms:
        # The most efficient way I've found to do the calculation is to
        # take the obs and preds shift columns for an atom, repeat each
        # into a matrix, then subtract these matrixes from each other.
        # That way, all calculations take advantage of vectorisation.
        # Much faster than using loops.
        obs_atom = pd.DataFrame(
            obs[atom]
            .repeat(len(obs.index))
            .values.reshape([len(obs.index), -1]),
            index=obs.index,
            columns=preds.index,
        )
        preds_atom = pd.DataFrame(
            preds[atom]
            .repeat(len(preds.index))
            .values.reshape([len(preds.index), -1])
            .transpose(),
            index=obs.index,
            columns=preds.index,
        )

        # If correcting the predictions, subtract a linear function of the
        # observed shift from each predicted shift
        if pred_correction:
            logger.info('Calculating corrections to predicted shifts')
            preds_corr_atom = preds_atom
            for res in preds['Res_type'].dropna().unique():
                if (atom + '_' + res) in lm_pars.index:
                    # Find shifts of the current atom/residue combination
                    mask = (lm_pars['Atom_type'] == atom) & (
                        lm_pars['Res_type'] == res
                    )

                    # Look up model parameters
                    grad = lm_pars.loc[mask, 'Grad'].tolist()[0]
                    offset = lm_pars.loc[mask, 'Offset'].tolist()[0]

                    # Make the correction
                    if atom in ('C_m1', 'CA_m1', 'CB_m1'):
                        mask = preds['Res_type_m1'] == res
                        preds_corr_atom.loc[:, mask] = (
                            preds_atom.loc[:, mask]
                            - grad * obs_atom.loc[:, mask]
                            - offset
                        )
                    else:
                        mask = preds['Res_type'] == res
                        preds_corr_atom.loc[:, mask] = (
                            preds_atom.loc[:, mask]
                            - grad * obs_atom.loc[:, mask]
                            - offset
                        )

            delta_atom = preds_corr_atom - obs_atom
        else:
            delta_atom = preds_atom - obs_atom

        if delta_correlation:
            # Store delta matrix for this atom type for later analysis
            delta_list = delta_list + [delta_atom.values]
        else:
            # Make a note of NA positions in delta, and set them to zero
            # (this avoids warnings when using norm.logpdf)
            na_mask = np.isnan(delta_atom)
            delta_atom[na_mask] = 0

            # Calculate the log probability density
            prob_atom = pd.DataFrame(
                norm.logpdf(delta_atom, scale=atom_sd[atom]),
                index=obs.index,
                columns=preds.index,
            )

            # Replace former NA values with a default value
            prob_atom[na_mask] = log10(default_prob)

            # Add to the log prob matrix
            log_prob_matrix = log_prob_matrix + prob_atom

    if delta_correlation:
        logger.info('Accounting for correlated prediction errors')

        # Combine the delta matrixes from each atom type into a single 3D matrix
        delta_mat = np.array(delta_list)
        # Move the axis containing the atom category to the end
        # (necessary for multivariate_normal function)
        delta_mat = np.moveaxis(delta_mat, 0, -1)

        # Make a note of NA positions in delta, and set them to zero
        na_mask = np.isnan(delta_mat)
        delta_mat[na_mask] = 0

        # Initialise multivariate model and calculate log_pdf
        mvn = multivariate_normal(d_mean, d_cov)
        log_prob_matrix = pd.DataFrame(
            mvn.logpdf(delta_mat), index=obs.index, columns=preds.index
        )

        # Apply a penalty for missing data
        na_matrix = na_mask.sum(axis=-1)    # Count how many NA values for
        # each Res/SS pair
        log_prob_matrix = log_prob_matrix + log10(default_prob) * na_matrix

    # HP - this is the section of code that uses amino acid restraints. As you can see, it is commented out. Thus I won't bother implementing AA restraints yet into the refactor. Because I can't test what it does.
    #        if self.pars["use_ss_class_info"]:
    #            # For each type of residue type information that's available, make a
    #            # matrix showing the probability modifications due to type mismatch,
    #            # then add it to log_prob_matrix
    #            # Maybe make SS_class mismatch a parameter in config file?
    #            for ss_class in {"SS_class","SS_class_m1"}.intersection(obs.columns):
    #                #print(ss_class)
    #                SS_class_matrix = pd.DataFrame(0, index=log_prob_matrix.index,
    #                                           columns=log_prob_matrix.columns)
    #
    #                # For each amino acid type in turn:
    #                for res in preds["Res_type"].dropna().unique():
    #                    # Work out which observations could be that aa type
    #                    allowed = obs[ss_class].str.contains(res).fillna(True)
    #                    # Select the predictions which are that aa type
    #                    pred_list = preds.loc[preds["Res_type_m1"]==res,"Res_name"]
    #                    # For the selected predictions, penalise any observations
    #                    # where the current aa type is not allowed
    #                    for p in pred_list:
    #                        SS_class_matrix.loc[:,p] = (~allowed)*-100 #log10(0.01)
    #
    #                log_prob_matrix = log_prob_matrix + SS_class_matrix

    # Sort out NAs and dummy residues/spin systems
    log_prob_matrix[log_prob_matrix.isna()] = 2 * np.nanmin(
        log_prob_matrix.values
    )
    log_prob_matrix.loc[obs['Dummy_SS'], :] = 0
    log_prob_matrix.loc[:, preds['Dummy_res']] = 0

    log_prob_matrix.index.name = 'SS_name'
    log_prob_matrix.columns.name = 'Res_name'

    logger.info(
        'Calculated log probability matrix (%dx%d)',
        log_prob_matrix.shape[0],
        log_prob_matrix.shape[1],
    )

    return log_prob_matrix


def find_best_assignment(
    score_matrix,
    maximise=True,
    inc=None,
    exc=None,
    dummy_rows=None,
    dummy_cols=None,
    return_none_all_dummy=False,
    logger = None,
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

    logger.info('Started linear assignment')

    if inc is not None:
        # Check for conflicting entries in inc
        conflicts = inc[row_name].duplicated(keep=False) | inc[
            col_name
        ].duplicated(keep=False)
        if any(conflicts):
            logger.warning(
                'Warning: entries in inc conflict with one another - dropping conflicts'
            )
            # print(inc[conflicts])
            inc = inc[~conflicts]

        if exc is not None:
            # Check constraints are consistent
            # Get rid of any entries in exc which share a Res or SS with inc
            exc_in_inc = exc[row_name].isin(inc[row_name]) | exc[
                col_name
            ].isin(inc[col_name])
            if any(exc_in_inc):
                logger.warning(
                    'Some values in exc are also found in inc, so are redundant.'
                )
                # print(exc[exc_in_inc])
                exc = exc.loc[~exc_in_inc, :]

        # Removed fixed assignments from probability matrix and obs, preds
        # dataframes. Latter is needed if inc includes any dummy SS/res,
        # and to detect if the reduced data is entirely dummies
        score_matrix_reduced = score_matrix.drop(index=inc[row_name]).drop(
            columns=inc[col_name]
        )
        logger.info(
            '%d assignments were fixed, %d remain to be assigned'
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
            logger.debug('Score matrix includes only dummy rows/columns')
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
        ) in (
            exc.iterrows()
        ):     # iterates over (index, row as pd.Series) tuples
            # If one side of an exclude pair is a dummy row or column,
            # exclude *all* dummy rows and columns
            if r[row_name] in dummy_rows:
                score_matrix_reduced.loc[r[row_name], dummy_cols] = penalty
            elif r[col_name] in dummy_cols:
                score_matrix_reduced.loc[dummy_rows, r[col_name]] = penalty
            else:
                score_matrix_reduced.loc[r['SS_name'], r['Res_name']] = penalty

        logger.info(
            'Penalised %d excluded row,column pairs' % len(exc.index)
        )

    if maximise:
        row_ind, col_ind = linear_sum_assignment(-1 * score_matrix_reduced)
        # -1 because linear_sum_assignment() minimises sum, but we want to
        # maximise it.
    else:
        row_ind, col_ind = linear_sum_assignment(score_matrix_reduced)

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


def make_assign_df(self, matching, set_assign_df=False):
    """Make a dataframe with full assignment information, given a dataframe
    of SS_name and Res_name.

    Matching may have additional columns, which will also be kept.
    """
    obs = self.obs.copy()
    obs.index.name = 'name'   # Needed to avoid error when merging dataframes.
    preds = self.preds.copy()
    log_prob_matrix = self.log_prob_matrix.copy()
    valid_atoms = list(self.pars['atom_set'])
    extra_cols = set(matching.columns).difference({'SS_name', 'Res_name'})

    if not {'SS_name', 'Res_name'}.issubset(matching.columns):
        self.logger.warning(
            'Cannot make assignment dataframe, as matching'
            + ' dataframe does not have the correct columns'
        )

    assign_df = pd.merge(
        matching,
        preds.loc[:, ['Res_N', 'Res_type', 'Res_name', 'Dummy_res']],
        on='Res_name',
        how='left',
    )
    assign_df = assign_df[
        ['Res_name', 'Res_N', 'Res_type', 'SS_name', 'Dummy_res']
        + list(extra_cols)
    ]
    assign_df = pd.merge(
        assign_df,
        obs.loc[:, obs.columns.isin(['SS_name', 'Dummy_SS'] + valid_atoms)],
        on='SS_name',
        how='left',
    )
    assign_df = pd.merge(
        assign_df,
        preds.loc[:, preds.columns.isin(valid_atoms + ['Res_name'])],
        on='Res_name',
        suffixes=('', '_pred'),
        how='left',
    )

    # assign_df["Log_prob"] = log_prob_matrix.lookup(
    #                                     assign_df["SS_name"],
    #                                     assign_df["Res_name"])
    assign_df['Log_prob'] = df_lookup(
        log_prob_matrix, assign_df['SS_name'], assign_df['Res_name']
    )
    # Careful above not to get rows/columns confused

    assign_df = assign_df.sort_values(by='Res_N')

    if set_assign_df:
        self.assign_df = assign_df

    return assign_df
