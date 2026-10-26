from abc import ABC, abstractmethod
from refactor.dataclasses import ShiftData, AssignmentResult
from refactor.SNAPSconfig import SNAPSConfig, build_config
import pandas as pd
import numpy as np
from typing import Dict, Type, List, Optional, Union
from scipy.stats import norm, multivariate_normal
from math import log10
from modules.assigner_functions import df_lookup

from scipy.optimize import linear_sum_assignment


class AssignmentStrategy(ABC):
    """Base class for assignment strategies"""

    name: str = ''  # Subclasses must set this
    config: Optional[SNAPSConfig] = None

    def __init__(self, config: Optional[Union[SNAPSConfig, dict]] = None):
        if config is None:
            self.config = build_config()
        elif isinstance(config, dict):
            self.config = build_config(cli_overrides=config)
        else:
            self.config = config

        # May want to change this to verbally complain about not having a config set
        # Instead of just silently falling back to defaults

    @abstractmethod
    def assign(self, data: ShiftData) -> AssignmentResult:
        """Perform assignment and return results"""
        pass

    # Seems to be shared between two strategies at least, so might as well put in assignment strategy instead of remaking
    def preprocess_obs_preds(self, obs: pd.DataFrame, preds: pd.DataFrame):
        """Perform preprocessing on the observed and predicted shifts to prepare
        them for calculation of the log probability matrix.

        1) Remove proline residues from predictions
        2) Discard atom types that are not used or aren't present in both
        3) Add dummy rows to obs or preds to bring them to the same length
        """

        #### Delete any prolines in preds
        print(
            'Removing %d prolines from predictions'
            % sum(preds['Res_type'] == 'P')
        )
        preds = preds.drop(preds.index[preds['Res_type'] == 'P'])

        # Remove references to deleted residues from Res_name_m1/p1
        preds.loc[
            ~preds['Res_name_m1'].isin(preds['Res_name']), 'Res_name_m1'
        ] = np.nan
        preds.loc[
            ~preds['Res_name_p1'].isin(preds['Res_name']), 'Res_name_p1'
        ] = np.nan

        #### Restrict atom types
        # self.pars["atom_set"] is the set of atoms to be used in the analysis

        # Get all non-atom columns
        all_atoms = self.config.supported_atom_types
        obs_metadata = sorted(set(obs.columns).difference(all_atoms))
        preds_metadata = sorted(set(preds.columns).difference(all_atoms))
        # Get atoms that are in atom_set *and* present in both obs and preds
        shared_atoms = sorted(
            list(
                self.config.atom_set.intersection(obs.columns).intersection(
                    preds.columns
                )
            )
        )   # It has to be a set before this point but has to be a list past this point, oddly
        obs = obs.loc[:, obs_metadata + shared_atoms]
        preds = preds.loc[:, preds_metadata + shared_atoms]

        print('Restricted atom types to %s' % ','.join(shared_atoms))

        #### Create dummy rows
        # Create columns to keep track of dummy status
        preds['Dummy_res'] = False
        obs['Dummy_SS'] = False

        N = len(obs.index)
        M = len(preds.index)

        if N > M:     # If there are more spin systems than predictions
            dummies = pd.DataFrame(
                np.nan,
                columns=preds.columns,
                index=[
                    self.config.dummy_predicted_prefix + str(i)
                    for i in 1 + np.arange(N - M)
                ],
            )
            dummies['Res_name'] = dummies.index
            dummies['Dummy_res'] = True
            # preds = preds.append(dummies)
            preds = pd.concat([preds, dummies])
            print('Added %d dummy predicted residues' % len(dummies.index))
        elif M > N:
            dummies = pd.DataFrame(
                np.nan,
                columns=obs.columns,
                index=[
                    self.config.dummy_observed_prefix + str(i)
                    for i in 1 + np.arange(M - N)
                ],
            )
            dummies['SS_name'] = dummies.index
            dummies['Dummy_SS'] = True
            # obs = obs.append(dummies)
            obs = pd.concat([obs, dummies])
            print('Added %d dummy observed residues' % len(dummies.index))

        return (obs, preds)

    # Same with this
    def calc_log_prob_matrix(
        self,
        obs: pd.DataFrame,
        preds: pd.DataFrame,
    ) -> pd.DataFrame:
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
        The configured shift scale factor is applied to atom standard
        deviations, and the configured default probability penalizes
        missing data.
        """
        obs = obs.copy()
        preds = preds.copy()
        atoms = sorted(self.config.atom_set.intersection(obs.columns))

        if self.config.pred_correction:
            # Import parameters for correcting the shifts
            lm_pars = pd.read_csv(
                self.config.pred_correction_file, index_col=0
            )
            self.config.logger.info(
                'Imported pred_correction info from %s'
                % self.config.pred_correction_file
            )

        if self.config.delta_correlation:
            # Import parameters describing the delta correlations
            # Note: this also sorts the atom types in d_mean and c_cov into the
            # same order as the 'atoms' list defined above.
            if self.config.pred_correction:
                d_mean = pd.read_csv(
                    self.config.delta_correlation_mean_corrected_file,
                    header=None,
                    index_col=0,
                ).loc[atoms, 1]
                d_cov = pd.read_csv(
                    self.config.delta_correlation_cov_corrected_file,
                    index_col=0,
                ).loc[atoms, atoms]
                self.config.logger.info(
                    'Imported delta_correlation info from %s and %s'
                    % (
                        self.config.delta_correlation_mean_corrected_file,
                        self.config.delta_correlation_cov_corrected_file,
                    )
                )
            else:
                d_mean = pd.read_csv(
                    self.config.delta_correlation_mean_file,
                    header=None,
                    index_col=0,
                ).loc[atoms, 1]
                d_cov = pd.read_csv(
                    self.config.delta_correlation_cov_file, index_col=0
                ).loc[atoms, atoms]
                self.config.logger.info(
                    'Imported delta_correlation info from %s and %s'
                    % (
                        self.config.delta_correlation_mean_file,
                        self.config.delta_correlation_cov_file,
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
            if self.config.pred_correction:
                self.config.logger.info(
                    'Calculating corrections to predicted shifts'
                )
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

            if self.config.delta_correlation:
                # Store delta matrix for this atom type for later analysis
                delta_list = delta_list + [delta_atom.values]
            else:
                # Make a note of NA positions in delta, and set them to zero
                # (this avoids warnings when using norm.logpdf)
                na_mask = np.isnan(delta_atom)
                delta_atom[na_mask] = 0

                # Calculate the log probability density
                prob_atom = pd.DataFrame(
                    norm.logpdf(
                        delta_atom,
                        scale=self.config.atom_sd[atom]
                        * self.config.shift_scale_factor,
                    ),
                    index=obs.index,
                    columns=preds.index,
                )

                # Replace former NA values with a default value
                prob_atom[na_mask] = log10(self.config.default_probability)

                # Add to the log prob matrix
                log_prob_matrix = log_prob_matrix + prob_atom

        if self.config.delta_correlation:
            self.config.logger.info(
                'Accounting for correlated prediction errors'
            )

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
            na_matrix = na_mask.sum(axis=-1)  # Count how many NA values for
            # each Res/SS pair
            log_prob_matrix = (
                log_prob_matrix
                + log10(self.config.default_probability) * na_matrix
            )

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

        self.config.logger.info(
            'Calculated log probability matrix (%dx%d)',
            log_prob_matrix.shape[0],
            log_prob_matrix.shape[1],
        )

        return log_prob_matrix

    def calc_mismatch_matrix(self, obs, threshold=0.2, logger=None):
        """Calculate pairwise sequential carbon mismatches for observed shifts."""
        carbons = pd.Series(['C', 'CA', 'CB'])
        seq_atoms = carbons[
            carbons.isin(obs.columns) & (carbons + '_m1').isin(obs.columns)
        ]

        if seq_atoms.empty:
            if logger:
                logger.warning(
                    'No sequential links in data - mismatch matrix not calculated.'
                )
            return None

        mismatch_matrix = pd.DataFrame(
            0, index=obs['SS_name'], columns=obs['SS_name']
        )
        mismatch_matrix.columns.name = 'i'
        mismatch_matrix.index.name = 'i_m1'
        consistent_links_matrix = mismatch_matrix.copy()

        for atom in seq_atoms:
            mismatch_atom = pd.DataFrame(
                obs[atom].to_numpy()[:, None]
                - obs[f'{atom}_m1'].to_numpy()[None, :],
                index=obs['SS_name'],
                columns=obs['SS_name'],
            )
            mismatch_atom.columns.name = 'i'
            mismatch_atom.index.name = 'i_m1'

            consistent_links_atom = abs(mismatch_atom) < threshold
            na_mask = mismatch_atom.isna()
            mismatch_atom = mismatch_atom.mask(na_mask, 0)
            consistent_links_atom = consistent_links_atom.mask(na_mask, 0)

            mismatch_matrix = mismatch_matrix.combine(
                abs(mismatch_atom), np.maximum
            )
            consistent_links_matrix = (
                consistent_links_matrix + consistent_links_atom
            )

        if logger:
            logger.info(
                'Calculated mismatch and consistent_links matrixes (%dx%d)',
                mismatch_matrix.shape[0],
                mismatch_matrix.shape[1],
            )
        return mismatch_matrix, consistent_links_matrix

    def add_consistency_info(
        self,
        assign_df,
        preds,
        mismatch_matrix=None,
        consistent_links_matrix=None,
        threshold=0.2,
        logger=None,
    ):
        """Add sequential mismatch metrics and confidence to assignments."""
        assign_df = assign_df.copy()
        sequential_atoms = {
            'C',
            'CA',
            'CB',
            'C_m1',
            'CA_m1',
            'CB_m1',
        }.issubset(assign_df.columns) or any(
            atom in assign_df.columns and f'{atom}_m1' in assign_df.columns
            for atom in ('C', 'CA', 'CB')
        )

        if not sequential_atoms:
            assign_df['Max_mismatch_m1'] = np.nan
            assign_df['Max_mismatch_p1'] = np.nan
            assign_df['Num_good_links_m1'] = 0
            assign_df['Num_good_links_p1'] = 0
            assign_df['Confidence'] = 'Undefined'
            if logger:
                logger.warning(
                    "Couldn't calculate assignment confidence: "
                    'need sequential atom information'
                )
            return assign_df

        if mismatch_matrix is None or consistent_links_matrix is None:
            raise ValueError(
                'mismatch_matrix and consistent_links_matrix are required '
                'when adding consistency information'
            )

        matching = assign_df.copy()
        matching.index = matching['Res_name']
        matching.index.name = None
        tmp = pd.concat(
            [matching, preds[['Res_name_m1', 'Res_name_p1']]],
            axis=1,
        )
        tmp = tmp.merge(
            matching[['SS_name']],
            how='left',
            left_on='Res_name_m1',
            right_index=True,
            suffixes=('', '_m1'),
        )
        tmp = tmp.merge(
            matching[['SS_name']],
            how='left',
            left_on='Res_name_p1',
            right_index=True,
            suffixes=('', '_p1'),
        )

        tmp_m1 = tmp.dropna(subset=['SS_name_m1'])
        tmp_p1 = tmp.dropna(subset=['SS_name_p1'])
        tmp['Max_mismatch_m1'] = pd.Series(
            list(
                df_lookup(
                    mismatch_matrix,
                    tmp_m1['SS_name_m1'],
                    tmp_m1['SS_name'],
                )
            ),
            index=tmp_m1.index,
        )
        tmp['Max_mismatch_p1'] = pd.Series(
            list(
                df_lookup(
                    mismatch_matrix,
                    tmp_p1['SS_name'],
                    tmp_p1['SS_name_p1'],
                )
            ),
            index=tmp_p1.index,
        )
        tmp['Max_mismatch'] = tmp[['Max_mismatch_m1', 'Max_mismatch_p1']].max(
            axis=1
        )
        tmp['Num_good_links_m1'] = pd.Series(
            list(
                df_lookup(
                    consistent_links_matrix,
                    tmp_m1['SS_name_m1'],
                    tmp_m1['SS_name'],
                )
            ),
            index=tmp_m1.index,
        )
        tmp['Num_good_links_p1'] = pd.Series(
            list(
                df_lookup(
                    consistent_links_matrix,
                    tmp_p1['SS_name'],
                    tmp_p1['SS_name_p1'],
                )
            ),
            index=tmp_p1.index,
        )
        tmp['Num_good_links_m1'] = tmp['Num_good_links_m1'].fillna(0)
        tmp['Num_good_links_p1'] = tmp['Num_good_links_p1'].fillna(0)
        tmp['Num_good_links'] = (
            tmp['Num_good_links_m1'] + tmp['Num_good_links_p1']
        )

        for direction in ('m1', 'p1'):
            tmp[f'Conf_{direction}'] = 'N'
            tmp.loc[
                tmp[f'Num_good_links_{direction}'] > 0, f'Conf_{direction}'
            ] = 'W'
            tmp.loc[
                tmp[f'Num_good_links_{direction}'] > 1, f'Conf_{direction}'
            ] = 'S'
            tmp.loc[
                tmp[f'Max_mismatch_{direction}'] > threshold,
                f'Conf_{direction}',
            ] = 'X'

        tmp['Conf2'] = tmp['Conf_m1'] + tmp['Conf_p1']
        tmp['Conf2'] = tmp['Conf2'].replace(
            ['WS', 'NS', 'XS', 'NW', 'XW', 'XN'],
            ['SW', 'SN', 'SX', 'WN', 'WX', 'NX'],
        )
        tmp['Confidence'] = tmp['Conf2'].replace(
            {
                'SS': 'High',
                'SW': 'High',
                'SN': 'High',
                'WW': 'High',
                'SX': 'Medium',
                'WN': 'Medium',
                'WX': 'Low',
                'NX': 'Unreliable',
                'XX': 'Unreliable',
                'NN': 'Undefined',
            }
        )
        if logger:
            summary = ', '.join(
                f'{count} {confidence}'
                for confidence, count in tmp['Confidence']
                .value_counts()
                .sort_values(ascending=False)
                .items()
            )
            logger.info('Calculated assignment confidence: ' + summary)

        consistency = tmp[
            [
                'SS_name',
                'Res_name',
                'Max_mismatch_m1',
                'Max_mismatch_p1',
                'Max_mismatch',
                'Num_good_links_m1',
                'Num_good_links_p1',
                'Num_good_links',
                'Confidence',
            ]
        ]
        return assign_df.merge(consistency, how='left')


# Strategy registry for self-registration
class StrategyRegistry:
    """Registry for self-registering assignment strategies"""

    _strategies: Dict[str, Type[AssignmentStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a strategy"""

        def decorator(strategy_class):
            strategy_class.name = name
            cls._strategies[name] = strategy_class
            return strategy_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[AssignmentStrategy]:
        """Get strategy class by name"""
        if name not in cls._strategies:
            available = ', '.join(cls._strategies.keys())
            raise ValueError(
                f'Unknown strategy: {name}. Available: {available}'
            )
        return cls._strategies[name]

    @classmethod
    def list_names(cls) -> List[str]:
        """List all registered strategy names"""
        return sorted(cls._strategies.keys())

    @classmethod
    def create(
        cls,
        name: str,
        config: Optional[Union[SNAPSConfig, dict]] = None,
    ) -> AssignmentStrategy:
        """Factory method to create a strategy instance"""
        strategy_class = cls.get(name)
        return strategy_class(config)


@StrategyRegistry.register('direct')
class DirectStrategy(AssignmentStrategy):
    def prepare(self, data: ShiftData):
        """Prepare input data and calculate the unmodified score matrix."""
        obs, preds = self.preprocess_obs_preds(
            data.obs_shifts, data.pred_shifts
        )
        return obs, preds, self.calc_log_prob_matrix(obs, preds)

    def assign(
        self, data: ShiftData, set_assign_df: bool = True
    ) -> AssignmentResult:
        """Perform direct assignment from raw shift data."""
        obs, preds, log_prob = self.prepare(data)
        return self.assign_from_matrix(obs, preds, log_prob)

    def assign_from_matrix(
        self,
        obs: pd.DataFrame,
        preds: pd.DataFrame,
        log_prob: pd.DataFrame,
    ) -> AssignmentResult:
        """Solve an assignment using an already calculated score matrix."""
        matching = self.find_best_assignment(
            score_matrix=log_prob,
            maximise=self.config.maximise_assignments,
        )
        assign_df = self.make_assign_df(
            obs=obs,
            preds=preds,
            log_prob_matrix=log_prob,
            matching=matching,
        )

        result = self.calc_mismatch_matrix(
            obs=obs,
            threshold=self.config.seq_link_threshold,
            logger=self.config.logger,
        )
        if result is not None:
            mismatch_matrix, consistent_links_matrix = result
        else:
            mismatch_matrix, consistent_links_matrix = (None, None)

        assign_df = self.add_consistency_info(
            assign_df=assign_df,
            preds=preds,
            mismatch_matrix=mismatch_matrix,
            consistent_links_matrix=consistent_links_matrix,
            threshold=self.config.seq_link_threshold,
            logger=self.config.logger,
        )

        return AssignmentResult(
            assignments=matching,
            log_prob_matrix=log_prob,
            assign_df=assign_df,
            mismatch_matrix=mismatch_matrix,
            confidence=assign_df[['SS_name', 'Res_name', 'Confidence']]
            if 'Confidence' in assign_df
            else None,
            metadata={
                'strategy': 'direct',
                'config': self.config,
            },
        )

    def find_best_assignment(
        self,
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

        self.config.logger.info('Started linear assignment')

        if inc is not None:
            # Check for conflicting entries in inc
            conflicts = inc[row_name].duplicated(keep=False) | inc[
                col_name
            ].duplicated(keep=False)
            if any(conflicts):
                self.config.logger.warning(
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
                    self.config.logger.warning(
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
            self.config.logger.info(
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
                self.config.logger.debug(
                    'Score matrix includes only dummy rows/columns'
                )
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
            ):  # iterates over (index, row as pd.Series) tuples
                # If one side of an exclude pair is a dummy row or column,
                # exclude *all* dummy rows and columns
                if r[row_name] in dummy_rows:
                    score_matrix_reduced.loc[r[row_name], dummy_cols] = penalty
                elif r[col_name] in dummy_cols:
                    score_matrix_reduced.loc[dummy_rows, r[col_name]] = penalty
                else:
                    score_matrix_reduced.loc[
                        r['SS_name'], r['Res_name']
                    ] = penalty

            self.config.logger.info(
                'Penalised %d excluded row,column pairs' % len(exc.index)
            )

        if maximise:
            row_ind, col_ind = linear_sum_assignment(-1 * score_matrix_reduced)
            # -1 because linear_sum_assignment() minimises sum, but we want to
            # maximise it.
        else:
            row_ind, col_ind = linear_sum_assignment(score_matrix_reduced)

        dummy_columns = {
            index for index, column in enumerate(score_matrix_reduced.columns)
            if str(column).strip().startswith('DR_')
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
                    assignments[dummy_row], assignments[real_row] = real_column, dummy_column
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

    def make_assign_df(
        self,
        obs: pd.DataFrame,
        preds: pd.DataFrame,
        log_prob_matrix: pd.DataFrame,
        matching: pd.DataFrame,
    ):
        """Make a dataframe with full assignment information, given a dataframe
        of SS_name and Res_name.

        Matching may have additional columns, which will also be kept.
        """
        obs = obs.copy()
        preds = preds.copy()
        obs.index.name = (
            'name'  # Needed to avoid error when merging dataframes.
        )
        valid_atoms = sorted(self.config.atom_set)
        extra_cols = set(matching.columns).difference({'SS_name', 'Res_name'})

        if not {'SS_name', 'Res_name'}.issubset(matching.columns):
            self.config.logger.warning(
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
            + sorted(extra_cols)
        ]
        assign_df = pd.merge(
            assign_df,
            obs.loc[
                :, obs.columns.isin(['SS_name', 'Dummy_SS'] + valid_atoms)
            ],
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

        return assign_df


@StrategyRegistry.register('iterative')
class IterativeStrategy(AssignmentStrategy):
    """Iterative refinement of direct assignment."""

    def assign(self, data: ShiftData) -> AssignmentResult:
        direct = DirectStrategy(config=self.config)
        obs, preds, original_log_prob = direct.prepare(data)
        initial_result = direct.assign_from_matrix(
            obs, preds, original_log_prob
        )
        return self._iterate_until_consistent(
            obs=obs,
            preds=preds,
            original_log_prob=original_log_prob,
            direct=direct,
            initial_result=initial_result,
        )

    def _iterate_until_consistent(
        self,
        obs: pd.DataFrame,
        preds: pd.DataFrame,
        original_log_prob: pd.DataFrame,
        direct: DirectStrategy,
        initial_result: AssignmentResult,
    ) -> AssignmentResult:
        """Iterate assignments until sequentially consistent"""
        threshold = self.config.seq_link_threshold
        best_result = initial_result
        best_count = (
            initial_result.assign_df['Confidence']
            .isin(['High', 'Medium'])
            .sum()
        )

        mismatch_result = direct.calc_mismatch_matrix(
            obs=obs, threshold=threshold, logger=self.config.logger
        )
        if mismatch_result is None:
            best_result.log_prob_matrix = original_log_prob
            best_result.metadata['strategy'] = 'iterative'
            return best_result
        mismatch_matrix, _ = mismatch_result

        while True:
            confident = best_result.assign_df.loc[
                best_result.assign_df['Confidence'].isin(['High', 'Medium']),
                ['Res_name', 'SS_name'],
            ]
            working_log_prob = original_log_prob.copy()
            assignment_by_residue = confident.set_index('Res_name')['SS_name']

            for res, ss in assignment_by_residue.items():
                res_m1 = preds.loc[res, 'Res_name_m1']
                res_p1 = preds.loc[res, 'Res_name_p1']
                penalty = working_log_prob.min().min()

                if not pd.isna(res_m1):
                    allowed_ss = mismatch_matrix.loc[:, ss] <= threshold
                    working_log_prob.loc[~allowed_ss, res_m1] += penalty
                if not pd.isna(res_p1):
                    allowed_ss = mismatch_matrix.loc[ss, :] <= threshold
                    working_log_prob.loc[~allowed_ss, res_p1] += penalty

            candidate = direct.assign_from_matrix(obs, preds, working_log_prob)
            candidate_count = (
                candidate.assign_df['Confidence']
                .isin(['High', 'Medium'])
                .sum()
            )
            self.config.logger.info(
                'At the current stage, there are %d high or medium '
                'confidence assignments',
                candidate_count,
            )

            if candidate_count <= best_count:
                break
            best_result = candidate
            best_count = candidate_count

        best_result.log_prob_matrix = original_log_prob
        best_result.metadata['strategy'] = 'iterative'
        return best_result


# Backwards-compatible name retained for callers of the initial refactor.
DirectStragety = DirectStrategy


# Factory function (simplified - just delegates to registry)
def create_strategy(
    strategy_name: str,
    config: Optional[Union[SNAPSConfig, dict]] = None,
) -> AssignmentStrategy:
    """
    Factory function to create assignment strategies.
    Strategies are auto-registered via @StrategyRegistry.register decorator.
    """
    return StrategyRegistry.create(strategy_name, config)
