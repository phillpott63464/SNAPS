

import numpy as np
import pandas as pd
from refactor.dataclasses import AssignmentResult, ShiftData
from refactor.strategies.base import AssignmentStrategy
from refactor.strategies.registry import StrategyRegistry
from refactor.strategies.utils.output import make_assign_df
from scipy.stats import multivariate_normal, norm
from typing import Optional


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
            normalise_by="SS",
        )
        original_matrix = original_matrix.div(
            original_matrix.sum(axis=1), axis="rows"
        )

        prob_matrix = original_matrix
        if self.config.use_generic_predictions:
            generic_preds = self._generic_predictions(preds)
            generic_matrix = self.calc_prob_matrix(
                obs=obs,
                preds=generic_preds,
                delta_correlation=False,
                atom_res_sd_file=self.config.generic_shift_stdev_file,
                glycine_cb_penalty=1.0,
            )
            prob_matrix = original_matrix * generic_matrix
            prob_matrix = prob_matrix.div(
                prob_matrix.sum(axis=1), axis="rows"
            )
            prob_matrix = self._normalise(prob_matrix)
        else:
            prob_matrix = self._normalise(prob_matrix)
        original_log_prob = self._calc_log_prob_matrix(prob_matrix)

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
                output_log_prob=original_log_prob,
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
            prob_matrix = self.calc_triplet_prob_matrix(
                prob_matrix=prob_matrix,
                allowed_links_matrix=allowed_links,
                normalise_by="SS",
            )

            prob_matrix = self._normalise(prob_matrix)
            log_prob_matrix = self._calc_log_prob_matrix(prob_matrix)

            self.config.logger.info(
                "Completed triplet iteration %d",
                iteration + 1,
            )

        return self._assign_from_matrix(
            obs=obs,
            preds=preds,
            log_prob=log_prob_matrix,
            output_log_prob=original_log_prob,
            mismatch_result=mismatch_result,
            iterations=iterations,
        )

    def _assign_from_matrix(
        self,
        obs: pd.DataFrame,
        preds: pd.DataFrame,
        log_prob: pd.DataFrame,
        output_log_prob: pd.DataFrame,
        mismatch_result: tuple | None,
        iterations: int,
    ) -> AssignmentResult:
        """Construct an assignment without using DirectStrategy."""

        # Hungarian assignment.
        matching = self.find_best_assignment(
            score_matrix=log_prob,
            maximise=self.config.maximise_assignments,
        )
        matching = matching.sort_values("Res_name")

        # Construct the assignment dataframe.
        assign_df = make_assign_df(
            config=self.config,
            obs=obs,
            preds=preds,
            log_prob_matrix=output_log_prob,
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

    def calc_prob_matrix(
        self,
        preds,
        obs,
        normalise_by: Optional[str] = None,
        delta_correlation: Optional[bool] = None,
        atom_res_sd_file=None,
        glycine_cb_penalty: float = 0.01,
    ):
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

        if atom_res_sd_file is not None:
            atom_res_sd = pd.read_csv(atom_res_sd_file, index_col=0)
        else:
            atom_res_sd = None

        if delta_correlation is None:
            delta_correlation = self.config.delta_correlation

        if delta_correlation:
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

            if delta_correlation:
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

        if delta_correlation:
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
                prob_matrix.loc[obs_CB, g] = prob_matrix.loc[obs_CB, g] * glycine_cb_penalty

            glycines_m1 = preds[preds.Res_type == "G"].Res_name
            obs_CB_m1 = obs[~obs.CB_m1.isna()].SS_name
            for g in glycines_m1:
                prob_matrix.loc[obs_CB_m1, g] = prob_matrix.loc[obs_CB_m1, g] * glycine_cb_penalty

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
        if normalise_by == "Res":
            prob_matrix = prob_matrix.div(prob_matrix.sum(axis=0), axis="columns")
        elif normalise_by == "SS":
            prob_matrix = prob_matrix.div(prob_matrix.sum(axis=1), axis="rows")

        self.config.logger.info("Calculated probability matrix (%dx%d)",
                         prob_matrix.shape[0], prob_matrix.shape[1])

        return (prob_matrix)

    def _generic_predictions(self, preds: pd.DataFrame) -> pd.DataFrame:
        """Create residue-type predictions for missing structural shifts."""
        generic = preds.copy()
        means = pd.read_csv(self.config.generic_shift_file, index_col=0)

        for atom in ("H", "N", "C", "CA", "CB", "HA"):
            if atom in means:
                generic[atom] = generic["Res_type"].map(means[atom])

        by_name = generic.set_index("Res_name")
        for atom in self.config.atom_set:
            if atom not in generic:
                continue
            base_atom = atom.removesuffix("_m1").removesuffix("_p1")
            if base_atom not in by_name:
                continue
            if atom.endswith("_m1"):
                generic[atom] = generic["Res_name_m1"].map(by_name[base_atom])
            elif atom.endswith("_p1"):
                generic[atom] = generic["Res_name_p1"].map(by_name[base_atom])

        for atom in self.config.atom_set.intersection(preds.columns):
            generic[atom] = generic[atom].where(preds[atom].isna())

        return generic

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

    def calc_triplet_prob_matrix(self, prob_matrix, allowed_links_matrix, normalise_by=None):
        """ Calculate the probability that each spin system is assigned to a given residue,
        summed over all possible triplets, accounting for allowed sequential links.

        """
        # Calculate matrixes for probability contribution of i-1 and i+1 residues
        left_matrix = allowed_links_matrix.transpose().dot(prob_matrix)
        right_matrix = allowed_links_matrix.dot(prob_matrix)

        # Shift values left or right
        left_matrix = left_matrix.shift(1, axis=1).fillna(1)
        right_matrix = right_matrix.shift(-1, axis=1).fillna(1)

        # Calculate the triplet probability matrix
        triplet_prob_matrix = left_matrix * prob_matrix * right_matrix
        triplet_prob_matrix.index.name = "SS_name"

        # Do the normalisation, if needed
        if normalise_by=="Res":
            triplet_prob_matrix = triplet_prob_matrix.div(triplet_prob_matrix.sum(axis=0), axis="columns")
        elif normalise_by=="SS":
            triplet_prob_matrix = triplet_prob_matrix.div(triplet_prob_matrix.sum(axis=1), axis="rows")

        return(triplet_prob_matrix)


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

    @staticmethod
    def _calc_log_prob_matrix(prob_matrix: pd.DataFrame) -> pd.DataFrame:
        """Convert probabilities using Alex's zero-probability replacement."""
        positive = prob_matrix[prob_matrix > 0]
        replacement = min(positive.min().min(), 1e-200)
        matrix = prob_matrix.mask(prob_matrix == 0, replacement)
        return np.log10(matrix)
