

import numpy as np
import pandas as pd
from refactor.dataclasses import AssignmentResult, ShiftData
from refactor.strategies.base import AssignmentStrategy
from refactor.strategies.registry import StrategyRegistry
from refactor.strategies.utils.output import make_assign_df
from scipy.stats import multivariate_normal, norm


@StrategyRegistry.register("triplet")
class TripletStrategy(AssignmentStrategy):
    """Iterative triplet-informed assignment using sequential links."""

    def assign(self, data: ShiftData) -> AssignmentResult:
        # Prepare observations and predictions directly.
        obs, preds = self.preprocess_obs_preds(
            data.obs_shifts,
            data.pred_shifts,
        )

        original_matrix = self.calc_prob_matrix(
            obs=obs,
            preds=preds,
        )

        prob_matrix = self._normalise(original_matrix)

        # Identify compatible sequential spin-system pairs.
        mismatch_result = self.calc_mismatch_matrix(
            obs=obs,
            threshold=self.config.seq_link_threshold,
            logger=self.config.logger,
        )

        if mismatch_result is None:
            return self._assign_from_matrix(
                obs=obs,
                preds=preds,
                log_prob=original_matrix,
                mismatch_result=None,
                iterations=0,
            )

        mismatch_matrix, _ = mismatch_result

        allowed_links = (
            mismatch_matrix < self.config.seq_link_threshold
        ).astype(int)

        np.fill_diagonal(allowed_links.values, 0)

        iterations = self.config.triplet_prob_iterations
        log_prob_matrix = original_matrix

        for iteration in range(iterations):
            # Calculate neighbouring contributions.
            left = allowed_links.T.dot(prob_matrix)
            right = allowed_links.dot(prob_matrix)

            left = left.shift(1, axis=1).fillna(1.0)
            right = right.shift(-1, axis=1).fillna(1.0)

            # Triplet probability matrix.
            prob_matrix = left * prob_matrix * right
            prob_matrix.index.name = "SS_name"
            prob_matrix.columns.name = "Res_name"

            # Normalise as in alex-dev.
            prob_matrix = self._normalise(prob_matrix)

            # Convert to log10 scores for Hungarian matching.
            log_prob_matrix = np.log10(
                prob_matrix.clip(lower=1e-200)
            )

            self.config.logger.info(
                "Completed triplet iteration %d",
                iteration + 1,
            )

        return self._assign_from_matrix(
            obs=obs,
            preds=preds,
            log_prob=log_prob_matrix,
            mismatch_result=mismatch_result,
            iterations=iterations,
        )

    def _assign_from_matrix(
        self,
        obs: pd.DataFrame,
        preds: pd.DataFrame,
        log_prob: pd.DataFrame,
        mismatch_result: tuple | None,
        iterations: int,
    ) -> AssignmentResult:
        """Construct an assignment without using DirectStrategy."""

        # Hungarian assignment.
        matching = self.find_best_assignment(
            score_matrix=log_prob,
            maximise=self.config.maximise_assignments,
        )

        # Construct the assignment dataframe.
        assign_df = make_assign_df(
            config=self.config,
            obs=obs,
            preds=preds,
            log_prob_matrix=log_prob,
            matching=matching,
        )

        # Add sequential consistency information.
        if mismatch_result is not None:
            mismatch_matrix, consistent_links_matrix = mismatch_result
        else:
            mismatch_matrix = None
            consistent_links_matrix = None

        assign_df = self.add_consistency_info(
            assign_df=assign_df,
            preds=preds,
            mismatch_matrix=mismatch_matrix,
            consistent_links_matrix=consistent_links_matrix,
            threshold=self.config.seq_link_threshold,
            logger=self.config.logger,
        )

        confidence = (
            assign_df[["SS_name", "Res_name", "Confidence"]]
            if "Confidence" in assign_df.columns
            else None
        )

        return AssignmentResult(
            assignments=matching,
            assign_df=assign_df,
            log_prob_matrix=log_prob,
            mismatch_matrix=mismatch_matrix,
            confidence=confidence,
            metadata={
                "strategy": "triplet",
                "iterations": iterations,
                "config": self.config,
            },
        )

    def calc_prob_matrix(self, preds, obs):
        """
        Updated version from alex-dev

        Calculate a matrix of probabilities based on match between observed
        and predicted chemical shifts.

        By probability, we mean the value of the probability density function,
        assuming the prediction errors follow a Gaussian distribution.
        If self.pars["delta_correlation"]==True, correlations in errors between
        different atom types will be accounted for.

        Returns
        A DataFrame containing the probabilities

        Parameters
        preds: The predictions to use when calculating the probability. If None,
            the aligned_predictions are used.
        atom_sd: A dictionary containing the expected standard error in the
            predictions for each atom type
        glycine_CB_penalty: How much to penalise the probability that an
            observation corresponds to a glycine if it has a CB chemical shift
            normalise_by: If "SS" or "Res", the sum of probabilities is normalised
            to 1 along that axis. If None, no normalisation is done.
        """

        atoms = sorted(self.config.atom_set.intersection(obs.columns))
        prob_matrix = pd.DataFrame(1.0, index=obs.index, columns=preds.index)
        prob_matrix.index.name = "SS_name"
        prob_matrix.columns.name = "Res_name"

        glycine_CB_penalty = 1 # This was hardcoded, maybe make it a config value in the end

        if self.config.generic_shift_stdev_file != None:
            atom_res_sd = pd.read_csv(self.config.generic_shift_stdev_file, index_col=0)
        else:
            atom_res_sd = None

        if self.config.delta_correlation:
            # Import parameters describing the delta correlations
            # Note: this also sorts the atom types in d_mean and c_cov into the
            # same order as the 'atoms' list defined above.
            d_mean = pd.read_csv(self.config.delta_correlation_mean_file,
                                 header=None, index_col=0).loc[atoms, 1]
            d_cov = (pd.read_csv(self.config.delta_correlation_cov_file,
                                 index_col=0).loc[atoms, atoms])
            self.config.logger.info("Imported delta_correlation info from %s and %s"
                             % (self.config.delta_correlation_mean_file,
                                self.config.delta_correlation_cov_file))

            # For the delta_correlation method, we need to store the prediction
            # errors for each atom type for analysis at the end.
            delta_list = []

        # Calculate the differences between every prediction and every observation,
        # for each atom type
        for atom in atoms:
            # The most efficient way I've found to do the calculation is to
            # take the obs and preds shift columns for an atom, repeat each
            # into a matrix, then subtract these matrixes from each other.
            # That way, all calculations take advantage of vectorisation.
            # Much faster than using loops.
            obs_atom = pd.DataFrame(obs[atom].repeat(len(obs.index)).values.
                                    reshape([len(obs.index), -1]),
                                    index=obs.index, columns=preds.index)
            preds_atom = pd.DataFrame(preds[atom].repeat(len(preds.index)).values.
                                      reshape([len(preds.index), -1]).transpose(),
                                      index=obs.index, columns=preds.index)

            delta_atom = preds_atom - obs_atom

            # Make sure delta_atom is entirely numeric, with any NAs replaced with 0
            delta_atom = delta_atom.fillna(0.0).astype(float)

            if self.config.delta_correlation:
                # Store delta matrix for this atom type for later analysis
                delta_list = delta_list + [delta_atom.values]
            else:
                # Calculate probability density and apply to probability matrix
                if atom_res_sd is None:
                    prob_atom = pd.DataFrame(norm.pdf(delta_atom, scale=self.config.atom_sd[atom]),
                                             index=obs.index, columns=preds.index)
                else:
                    # If atom is from a neighbouring residue, choose the appropriate atom type.
                    atom2 = atom
                    if atom2 == "C_m1": atom2 = "C"
                    if atom2 == "CA_m1": atom2 = "CA"
                    if atom2 == "CB_m1": atom2 = "CB"
                    if atom2 in ["N_m1", "N_p1"]: atom2 = "N"

                    prob_atom = pd.DataFrame(1.0, index=obs.index, columns=preds.index)
                    for res in atom_res_sd.index:
                        mask = preds.Res_type == res
                        tmp = norm.pdf(delta_atom.loc[:, mask], scale=atom_res_sd.loc[res, atom2])
                        prob_atom.loc[:, mask] = pd.DataFrame(tmp,
                                                              index=obs.index, columns=preds.index[mask])

                prob_matrix = prob_matrix * prob_atom

        if self.config.delta_correlation:
            self.config.logger.info("Accounting for correlated prediction errors")

            # Combine the delta matrixes from each atom type into a single 3D matrix
            delta_mat = np.array(delta_list)
            # Move the axis containing the atom category to the end
            # (necessary for multivariate_normal function)
            delta_mat = np.moveaxis(delta_mat, 0, -1)

            # Initialise multivariate model and calculate log_pdf
            mvn = multivariate_normal(d_mean, d_cov)
            prob_matrix = pd.DataFrame(mvn.pdf(delta_mat),
                                       index=obs.index, columns=preds.index)

        # Penalise matches to glycine residues if SS has a CB
        if "CB" in obs.columns:
            glycines = preds[preds.Res_type == "G"].Res_name
            obs_CB = obs[~obs.CB.isna()].SS_name
            for g in glycines:
                prob_matrix.loc[obs_CB, g] = prob_matrix.loc[obs_CB, g] * glycine_CB_penalty

            glycines_m1 = preds[preds.Res_type == "G"].Res_name
            obs_CB_m1 = obs[~obs.CB_m1.isna()].SS_name
            for g in glycines_m1:
                prob_matrix.loc[obs_CB_m1, g] = prob_matrix.loc[obs_CB_m1, g] * glycine_CB_penalty

        # Pair up the proline dummy spin systems with the appropriate residue
        prolines = preds[preds.Res_type == "P"].Res_name
        prob_matrix.loc[:, prolines] = 1e-10
        prob_matrix.loc[prolines, :] = 1e-10
        for x in prolines:
            prob_matrix.loc[x, x] = 1.0

        # Replace any zero probabilities which a very small number
        # (Zero probabilities can happen due to rounding error, I think)
        prob_matrix[prob_matrix == 0] = 1e-100

        # Do the normalisation, if needed
        normalise_by = None #Hardcoded none in original implementation, but kept just in case it's needed
        if normalise_by == "Res":
            prob_matrix = prob_matrix.div(prob_matrix.sum(axis=0), axis="columns")
        elif normalise_by == "SS":
            prob_matrix = prob_matrix.div(prob_matrix.sum(axis=1), axis="rows")

        self.config.logger.info("Calculated probability matrix (%dx%d)",
                         prob_matrix.shape[0], prob_matrix.shape[1])

        return (prob_matrix)

    def preprocess_obs_preds(self, obs, preds):
        """Perform preprocessing on the observed and predicted shifts to prepare
        them for calculation of the log probability matrix.

        1) Discard atom types that are not used or aren't present in both
        2) Create dummy spin systems for prolines
        3) Add dummy rows to obs or preds to bring them to the same length

        Returns the modified obs and preds DataFrames.
        """

        #### Restrict atom types
        # self.pars["atom_set"] is the set of atoms to be used in the analysis
        # Get all non-atom columns
        all_atoms = {"H", "N", "C", "CA", "CB", "C_m1", "CA_m1", "CB_m1", "HA", "N_m1", "N_p1"}
        obs_metadata = sorted(set(obs.columns).difference(all_atoms))
        preds_metadata = sorted(set(preds.columns).
                                difference(all_atoms))
        # Get atoms that are in atom_set *and* present in both obs and preds
        shared_atoms = sorted(self.config.atom_set.intersection(obs.columns).
                              intersection(preds.columns))
        obs = obs.loc[:, obs_metadata + shared_atoms]
        preds = preds.loc[:, preds_metadata + shared_atoms]

        self.config.logger.info("Restricted atom types to %s" % ",".join(shared_atoms))

        #### Create dummy rows
        # Create columns to keep track of dummy status
        preds["Dummy_res"] = False
        obs["Dummy_SS"] = False

        # Create dummy observations for each proline residue
        # When probabilities are assigned, each proline observation will only match to its own proline
        prolines = preds[preds.Res_type == "P"].Res_name
        proline_obs = pd.DataFrame(index=prolines, columns=obs.columns)
        proline_obs["SS_name"] = prolines
        proline_obs["Dummy_SS"] = True
        obs = pd.concat([obs, proline_obs])

        # Add extra dummy residues or spin systems, so that there are equal numbers
        N = len(obs.index)
        M = len(preds.index)

        if N > M:  # If there are more spin systems than predictions
            dummies = pd.DataFrame(np.nan, columns=preds.columns,
                                   index=["DR_" + str(i) for i in 1 + np.arange(N - M)])
            dummies["Res_name"] = dummies.index
            dummies["Dummy_res"] = True
            # preds = preds.append(dummies)
            preds = pd.concat([preds, dummies])
            self.config.logger.info("Added %d dummy predicted residues" % len(dummies.index))
        elif M > N:
            dummies = pd.DataFrame(np.nan, columns=obs.columns,
                                   index=["DSS_" + str(i) for i in 1 + np.arange(M - N)])
            dummies["SS_name"] = dummies.index
            dummies["Dummy_SS"] = True
            # obs = obs.append(dummies)
            obs = pd.concat([obs, dummies])
            self.config.logger.info("Added %d dummy observed residues" % len(dummies.index))

        obs = obs.sort_values("SS_name")
        obs.index = obs.SS_name
        obs.index.name = None
        preds.index.name = None

        return obs, preds

    @staticmethod
    def _normalise(prob_matrix: pd.DataFrame) -> pd.DataFrame:
        """Alternating column/row normalisation."""
        matrix = prob_matrix.copy()

        for _ in range(20):
            col_sums = matrix.sum(axis=0).replace(0, np.nan)
            matrix = matrix.div(col_sums, axis=1)

            row_sums = matrix.sum(axis=1).replace(0, np.nan)
            matrix = matrix.div(row_sums, axis=0)

        return matrix.fillna(0.0)
