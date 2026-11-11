from typing import Optional

import numpy as np
import pandas as pd
from refactor.strategies.utils.misc import df_lookup
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal, norm

from refactor.dataclasses import AssignmentResult, ShiftData
from refactor.strategies.base import AssignmentStrategy
from refactor.strategies.registry import StrategyRegistry


@StrategyRegistry.register("triplet")
class TripletStrategy(AssignmentStrategy):
    """Iterative triplet-informed assignment using sequential links."""

    def assign(self, data: ShiftData) -> AssignmentResult:
        self.validate_data(data.obs_shifts, data.pred_shifts, data.sequence)
        # Algorith from snaps.py
        obs, preds = self.preprocess_obs_preds(data.obs_shifts, data.pred_shifts)
        prob_matrix = self.calc_prob_matrix(
            obs=obs, preds=preds, sequence=data.sequence
        )
        log_prob_matrix = self.calc_log_prob_matrix(prob_matrix)
        assign_df, _ = self.assign_from_preds(
            obs=obs, preds=preds, log_prob_matrix=log_prob_matrix
        )
        assign_df, _ = self.add_consistency_info(
            assign_df=assign_df,
            obs=obs,
            preds=preds,
            threshold=self.config.seq_link_threshold,
        )

        triplet_prob_matrix = prob_matrix.copy()
        triplet_log_prob_matrix = log_prob_matrix.copy()
        triplet_assign_df, triplet_matching = self.assign_from_preds(
            obs, preds, triplet_log_prob_matrix
        )
        triplet_assign_df, triplet_allowed_links_matrix = self.add_consistency_info(
            assign_df=triplet_assign_df,
            obs=obs,
            preds=preds,
            threshold=self.config.seq_link_threshold,
        )

        triplet_prob_list = [triplet_prob_matrix]
        triplet_log_prob_list = [triplet_log_prob_matrix]
        triplet_matching_list = [triplet_matching]
        triplet_assign_df_list = [triplet_assign_df]
        triplet_allowed_links_list = [triplet_allowed_links_matrix]
        for i in range(self.config.triplet_prob_iterations):
            triplet_prob_matrix = self.calc_triplet_prob_matrix(
                prob_matrix=triplet_prob_list[i],
                allowed_links_matrix=triplet_allowed_links_list[i],
                normalise_by="SS",
            )
            # Test of probability normalisation
            for _ in range(20):
                triplet_prob_matrix = triplet_prob_matrix.div(
                    triplet_prob_matrix.sum(axis=0), axis="columns"
                )
                triplet_prob_matrix = triplet_prob_matrix.div(
                    triplet_prob_matrix.sum(axis=1), axis="rows"
                )

            triplet_log_prob_matrix = self.calc_log_prob_matrix(triplet_prob_matrix)
            triplet_assign_df, triplet_matching = self.assign_from_preds(
                obs=obs, preds=preds, log_prob_matrix=triplet_log_prob_matrix
            )
            triplet_assign_df, triplet_allowed_links_matrix = self.add_consistency_info(
                assign_df=triplet_assign_df,
                obs=obs,
                preds=preds,
                threshold=self.config.seq_link_threshold,
            )

            triplet_prob_list += [triplet_prob_matrix]
            triplet_log_prob_list += [triplet_log_prob_matrix]
            triplet_matching_list += [triplet_matching]
            triplet_assign_df_list += [triplet_assign_df]
            triplet_allowed_links_list += [triplet_allowed_links_matrix]

        return AssignmentResult(
            assignments=triplet_matching_list[-1],
            log_prob_matrix=triplet_log_prob_list[-1],
            assign_df=triplet_assign_df_list[-1],
            assign_dfs=triplet_assign_df_list,
            # mismatch_matrix=mismatch_matrix,
            confidence=assign_df[["SS_name", "Res_name", "Confidence"]]
            if "Confidence" in assign_df
            else None,
            metadata={
                "strategy": "triplet",
                "config": self.config,
            },
        )

    def assign_from_preds(
        self, obs, preds, log_prob_matrix, set_assign_df=False, inc=None, exc=None
    ):
        """Assign the observed spin systems using predicted shifts only

        This function essentially wraps around find_best_assignment() and
        make_assign_df()
        inc: a DataFrame of (row, col) pairs which must be part of the assignment.
            First column has the index names, second has the column names.
        exc: a DataFrame of (row, col) pairs which may not be part of the assignment.
        """
        log_prob_matrix = log_prob_matrix.copy()
        log_prob_matrix.index.name = "SS_name"
        log_prob_matrix.columns.name = "Res_name"
        matching = self.find_best_assignment(
            log_prob_matrix, maximise=True, inc=inc, exc=exc
        )
        if not hasattr(self, "_assignment_log_prob_matrix"):
            self._assignment_log_prob_matrix = log_prob_matrix.copy()
        assign_df = self.make_assign_df(
            obs=obs,
            preds=preds,
            log_prob_matrix=self._assignment_log_prob_matrix,
            matching=matching,
        )

        self.config.logger.info(
            "Finished calculating best assignment based on predictions"
        )
        return (assign_df, matching)

    def add_consistency_info(self, assign_df, obs, preds, threshold=0.2):
        """Find maximum mismatch and number of 'significant' mismatches for
        each residue

        threshold: Minimum carbon shift difference for sequential residues to
            count as mismatched
        """

        assign_df = assign_df.copy()

        # First check if there are any sequential atoms
        if self.sequential_atoms_present(assign_df.columns):
            tmp, allowed_links_matrix = self.check_matching_consistency(
                assign_df, obs, preds, threshold
            )
            assign_df = assign_df.merge(tmp, how="left")
            self.config.logger.info("Finished checking assignment consistency")
        else:
            # You can't do a comparison without sequential atoms
            assign_df["Max_mismatch_m1"] = np.nan
            assign_df["Max_mismatch_p1"] = np.nan
            assign_df["Num_good_links_m1"] = 0
            assign_df["Num_good_links_p1"] = 0
            assign_df["Confidence"] = "Undefined"

            self.config.logger.warning(
                "Couldn't calculate assignment confidence: "
                + "need sequential atom information"
            )
            allowed_links_matrix = None

        return (assign_df, allowed_links_matrix)

    def sequential_atoms_present(self, atom_list):
        """Returns true if a pair of atoms in atom_list are sequential, otherwise False

        Acceptable pairs are (C, C_m1), (CA, C_Am1), (CB,C_Bm1)"""

        i_atoms = pd.Series(["C", "CA", "CB"])
        i_m1_atoms = pd.Series(["C_m1", "CA_m1", "CB_m1"])
        seq_atoms = i_atoms.isin(atom_list) & i_m1_atoms.isin(atom_list)

        return seq_atoms.any()

    def check_matching_consistency(self, matching, obs, preds, threshold=0.2):
        """Calculate mismatch scores and assignment confidence for a given matching

        Returns a DataFrame with Res_name, SS_name and columns containing the
        mismatch with adjacent residues, the number of good links, and the
        assignment confidence based on these factors

        Parameters
        matching: a DataFrame with Res_name and SS_name columns
        threshold: the maximum allowed mismatch for a good sequential link
        """
        # Create mismatch matrix if it doesn't already exist
        mismatch_matrix, consistent_links_matrix, allowed_links_matrix = (
            self.calc_mismatch_matrix(obs)
        )

        # Add Res_name_m1 and Res_name_p1 columns to matching DataFrame
        matching = matching.copy()
        matching.index = matching["Res_name"]
        matching.index.name = None

        tmp = pd.concat([matching, preds[["Res_name_m1", "Res_name_p1"]]], axis=1)

        # Add a SS_name_m1 and SS_name_p1 columns
        tmp = tmp.merge(
            matching[["SS_name"]],
            how="left",
            left_on="Res_name_m1",
            right_index=True,
            suffixes=("", "_m1"),
        )
        tmp = tmp.merge(
            matching[["SS_name"]],
            how="left",
            left_on="Res_name_p1",
            right_index=True,
            suffixes=("", "_p1"),
        )

        # Filter out rows with NaN's in SS_name_m1/p1
        tmp_m1 = tmp.dropna(subset=["SS_name_m1"])
        tmp_p1 = tmp.dropna(subset=["SS_name_p1"])

        # Get the mismatches
        tmp["Max_mismatch_m1"] = pd.Series(
            list(df_lookup(mismatch_matrix, tmp_m1["SS_name_m1"], tmp_m1["SS_name"])),
            index=tmp_m1.index,
        )
        tmp["Max_mismatch_p1"] = pd.Series(
            list(df_lookup(mismatch_matrix, tmp_p1["SS_name"], tmp_p1["SS_name_p1"])),
            index=tmp_p1.index,
        )
        tmp["Max_mismatch"] = tmp[["Max_mismatch_m1", "Max_mismatch_p1"]].max(axis=1)

        tmp["Num_good_links_m1"] = pd.Series(
            list(
                df_lookup(
                    consistent_links_matrix, tmp_m1["SS_name_m1"], tmp_m1["SS_name"]
                )
            ),
            index=tmp_m1.index,
        )
        tmp["Num_good_links_p1"] = pd.Series(
            list(
                df_lookup(
                    consistent_links_matrix, tmp_p1["SS_name"], tmp_p1["SS_name_p1"]
                )
            ),
            index=tmp_p1.index,
        )
        tmp["Num_good_links_m1"] = tmp["Num_good_links_m1"].fillna(0)
        tmp["Num_good_links_p1"] = tmp["Num_good_links_p1"].fillna(0)
        tmp["Num_good_links"] = tmp["Num_good_links_m1"] + tmp["Num_good_links_p1"]

        # Add an assignment confidence column
        tmp["Conf_m1"] = "N"
        tmp.loc[tmp["Num_good_links_m1"] > 0, "Conf_m1"] = "W"
        tmp.loc[tmp["Num_good_links_m1"] > 1, "Conf_m1"] = "S"
        tmp.loc[tmp["Max_mismatch_m1"] > threshold, "Conf_m1"] = "X"

        tmp["Conf_p1"] = "N"
        tmp.loc[tmp["Num_good_links_p1"] > 0, "Conf_p1"] = "W"
        tmp.loc[tmp["Num_good_links_p1"] > 1, "Conf_p1"] = "S"
        tmp.loc[tmp["Max_mismatch_p1"] > threshold, "Conf_p1"] = "X"

        tmp["Conf2"] = tmp["Conf_m1"] + tmp["Conf_p1"]
        # Reorder letters eg WS -> SW
        tmp["Conf2"] = tmp["Conf2"].replace(
            ["WS", "NS", "XS", "NW", "XW", "XN"], ["SW", "SN", "SX", "WN", "WX", "NX"]
        )

        tmp["Confidence"] = tmp["Conf2"]
        tmp["Confidence"] = tmp["Confidence"].replace(["SS", "SW", "SN", "WW"], "High")
        tmp["Confidence"] = tmp["Confidence"].replace(["SX", "WN"], "Medium")
        tmp["Confidence"] = tmp["Confidence"].replace("WX", "Low")
        tmp["Confidence"] = tmp["Confidence"].replace(["NX", "XX"], "Unreliable")
        tmp["Confidence"] = tmp["Confidence"].replace("NN", "Undefined")

        # Summarise confidence of results
        summary_str = ", ".join(
            [
                str(count) + " " + conf
                for conf, count in tmp["Confidence"]
                .value_counts()
                .sort_values(ascending=False)
                .items()
            ]
        )
        self.config.logger.info("Calculated assignment confidence: " + summary_str)

        return (
            tmp.loc[
                :,
                [
                    "SS_name",
                    "Res_name",
                    "Max_mismatch_m1",
                    "Max_mismatch_p1",
                    "Max_mismatch",
                    "Num_good_links_m1",
                    "Num_good_links_p1",
                    "Num_good_links",
                    "Confidence",
                ],
            ],
            allowed_links_matrix,
        )

    def calc_prob_matrix(self, obs, preds, sequence):
        prob_matrix = self._calc_prob_matrix(obs=obs, preds=preds, normalise_by="SS")

        if self.config.use_generic_predictions:
            generic_preds = self.create_generic_predictions(
                preds=preds, sequence=sequence
            )
            generic_prob_matrix = self.calc_generic_prob_matrix(
                obs=obs, preds=preds, generic_preds=generic_preds, sequence=sequence
            )
            # breakpoint()
            prob_matrix *= generic_prob_matrix
            prob_matrix = prob_matrix.div(prob_matrix.sum(axis=1), axis="rows")

        # Test of probability normalisation
        for i in range(20):
            prob_matrix = prob_matrix.div(prob_matrix.sum(axis=0), axis="columns")
            prob_matrix = prob_matrix.div(prob_matrix.sum(axis=1), axis="rows")

        return prob_matrix

    def _calc_prob_matrix(
        self,
        obs,
        preds,
        atom_res_sd=None,
        glycine_CB_penalty=None,
        normalise_by=None,
        delta_correlation=None,
    ):
        """Calculate a matrix of probabilities based on match between observed
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

        obs = obs.copy()
        preds = preds.copy()
        atoms = sorted(self.config.atom_set.intersection(obs.columns))
        prob_matrix = pd.DataFrame(1.0, index=obs.index, columns=preds.index)
        prob_matrix.index.name = "SS_name"
        prob_matrix.columns.name = "Res_name"
        if delta_correlation == None:
            delta_correlation = self.config.delta_correlation
        atom_sd = self.config.atom_sd
        if glycine_CB_penalty == None:
            glycine_CB_penalty = self.config.glycine_CB_penalty

        if delta_correlation:
            # Import parameters describing the delta correlations
            # Note: this also sorts the atom types in d_mean and c_cov into the
            # same order as the 'atoms' list defined above.
            d_mean = pd.read_csv(
                self.config.delta_correlation_mean_file, header=None, index_col=0
            ).loc[atoms, 1]
            d_cov = pd.read_csv(
                self.config.delta_correlation_cov_file, index_col=0
            ).loc[atoms, atoms]
            self.config.logger.info(
                "Imported delta_correlation info from %s and %s"
                % (
                    self.config.delta_correlation_mean_file,
                    self.config.delta_correlation_cov_file,
                )
            )

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
            obs_atom = pd.DataFrame(
                obs[atom].repeat(len(obs.index)).values.reshape([len(obs.index), -1]),
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

            delta_atom = preds_atom - obs_atom

            # Make sure delta_atom is entirely numeric, with any NAs replaced with 0
            delta_atom = delta_atom.fillna(0.0).astype(float)

            if delta_correlation:
                # Store delta matrix for this atom type for later analysis
                delta_list = delta_list + [delta_atom.values]
            else:
                # Calculate probability density and apply to probability matrix
                if atom_res_sd is None:
                    prob_atom = pd.DataFrame(
                        norm.pdf(delta_atom, scale=atom_sd[atom]),
                        index=obs.index,
                        columns=preds.index,
                    )
                else:
                    # If atom is from a neighbouring residue, choose the appropriate atom type.
                    atom2 = atom
                    if atom2 == "C_m1":
                        atom2 = "C"
                    if atom2 == "CA_m1":
                        atom2 = "CA"
                    if atom2 == "CB_m1":
                        atom2 = "CB"
                    if atom2 in ["N_m1", "N_p1"]:
                        atom2 = "N"

                    prob_atom = pd.DataFrame(1.0, index=obs.index, columns=preds.index)
                    for res in atom_res_sd.index:
                        mask = preds.Res_type == res
                        tmp = norm.pdf(
                            delta_atom.loc[:, mask], scale=atom_res_sd.loc[res, atom2]
                        )
                        prob_atom.loc[:, mask] = pd.DataFrame(
                            tmp, index=obs.index, columns=preds.index[mask]
                        )

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
            prob_matrix = pd.DataFrame(
                mvn.pdf(delta_mat), index=obs.index, columns=preds.index
            )

        # Penalise matches to glycine residues if SS has a CB
        if "CB" in obs.columns:
            glycines = preds[preds.Res_type == "G"].Res_name
            obs_CB = obs[~obs.CB.isna()].SS_name
            for g in glycines:
                prob_matrix.loc[obs_CB, g] = (
                    prob_matrix.loc[obs_CB, g] * glycine_CB_penalty
                )

            glycines_m1 = preds[preds.Res_type == "G"].Res_name
            obs_CB_m1 = obs[~obs.CB_m1.isna()].SS_name
            for g in glycines_m1:
                prob_matrix.loc[obs_CB_m1, g] = (
                    prob_matrix.loc[obs_CB_m1, g] * glycine_CB_penalty
                )

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

        self.config.logger.info(
            "Calculated probability matrix (%dx%d)",
            prob_matrix.shape[0],
            prob_matrix.shape[1],
        )

        # self.prob_matrix = prob_matrix
        return prob_matrix

    def create_generic_predictions(self, preds, sequence):
        """Create a set of predicted shifts based only on residue type.
        These can be used where there are missing predictions from the structure.

        Returns a dataframe with the generic shift predictions for each residue.
        """
        generic_preds = preds.copy()
        generic_preds = generic_preds.loc[:, sequence.columns]

        # Import the average shifts for each residue type
        residue_shifts = pd.read_csv(self.config.generic_shift_file, index_col=0)

        # Create the generic shift predictions
        atom_list = ["H", "N", "C", "CA", "CB", "HA"]
        for atom in atom_list:
            generic_preds[atom] = np.nan

        for res in residue_shifts.index:
            for atom in atom_list:
                generic_preds.loc[generic_preds.Res_type == res, atom] = (
                    residue_shifts.loc[res, atom]
                )

        # Create the i-1 and i+1 columns
        generic_preds.index = generic_preds.Res_name
        generic_preds.index.name = None
        preds_m1 = generic_preds[
            list(
                {
                    "C",
                    "CA",
                    "CB",
                    "N",
                    "Res_type",
                    "Res_name",
                    "Res_name_p1",
                }.intersection(generic_preds.columns)
            )
        ].copy()
        preds_m1.index = preds_m1.Res_name_p1  # Note _p1, because we need to transfer the chemical shift to the i+1 residue!
        generic_preds = pd.merge(
            generic_preds,
            preds_m1[["C", "CA", "CB", "N"]],
            how="left",
            left_index=True,
            right_index=True,
            suffixes=["", "_m1"],
        )

        preds_p1 = generic_preds[
            list(
                {"N", "Res_type", "Res_name", "Res_name_m1"}.intersection(
                    generic_preds.columns
                )
            )
        ].copy()
        preds_p1.index = preds_p1.Res_name_m1  # Note _m1, because we need to transfer the chemical shift to the i-1 residue!
        generic_preds = pd.merge(
            generic_preds,
            preds_p1[["N"]],
            how="left",
            left_index=True,
            right_index=True,
            suffixes=["", "_p1"],
        )

        # Restrict to atom types in preds
        col_mask = generic_preds.columns.isin(preds.columns)
        generic_preds = generic_preds.loc[:, col_mask]

        return generic_preds

    def calc_generic_prob_matrix(
        self,
        obs,
        preds,
        generic_preds,
        sequence,
        only_missing_preds=True,
        normalise_by=None,
    ):
        """Calculate a probability matrix using generic shift predictions.
        This can be used in place of the probability matrix from predictions,
        or used to compensate for missing predictions.

        Parameters:
        only_missing_preds: if True, only generic predictions from atoms with
        missing predicted shifts are used.

        Returns
        The probability matrix"""
        generic_preds = generic_preds.copy()
        preds = preds.copy()
        obs = obs.copy()

        # Remove generic predictions for atoms with a real prediction
        if only_missing_preds:
            cols = generic_preds.columns[
                generic_preds.columns.isin(self.config.atom_set)
            ]
            generic_preds[~preds.loc[:, cols].isna()] = np.nan
            # breakpoint()

        # Import average generic shifts and standard deviations
        generic_stdev = pd.read_csv(self.config.generic_shift_stdev_file, index_col=0)

        # Calculate the probability matrix
        generic_prob_matrix = self._calc_prob_matrix(
            obs=obs,
            preds=generic_preds,
            atom_res_sd=generic_stdev,
            glycine_CB_penalty=1.0,
            delta_correlation=False,
        )

        # Do the normalisation, if needed
        if normalise_by == "Res":
            generic_prob_matrix = generic_prob_matrix.div(
                generic_prob_matrix.sum(axis=0), axis="columns"
            )
        elif normalise_by == "SS":
            generic_prob_matrix = generic_prob_matrix.div(
                generic_prob_matrix.sum(axis=1), axis="rows"
            )

        return generic_prob_matrix

    def calc_log_prob_matrix(self, prob_matrix=None):
        """Calculate a log probability matrix from a probability matrix

        By default uses the prob_matrix stored in the SNAPS_assigner object,
        but can also be passed a different prob_matrix.

        Returns
        A DataFrame containing the log probabilities
        """
        # Remove any zeroes, by setting them to the minimum value in the matrix
        replacement_value = min(prob_matrix[prob_matrix > 0].min().min(), 1e-200)
        prob_matrix[prob_matrix == 0] = replacement_value

        log_prob_matrix = np.log10(prob_matrix)

        return log_prob_matrix

    def calc_mismatch_matrix(self, obs):
        """Calculate matrix of the mismatch between i and i-1 observed carbon
        shifts for all spin system pairs. Also, make matrix of number of
        consistent sequential links for all spin system pairs.

        Parameters
        threshold: The cutoff value beyond which a seq link is considered bad
        """
        obs = obs.copy()
        threshold = self.config.seq_link_threshold

        # First check if there are any sequential atoms
        carbons = pd.Series(["C", "CA", "CB"])
        carbons_m1 = carbons + "_m1"
        seq_atoms = carbons[carbons.isin(obs.columns) & carbons_m1.isin(obs.columns)]

        if seq_atoms.size == 0:
            # You can't make a mismatch matrix
            self.config.logger.warning(
                "No sequential links in data - mismatch matrix not calculated."
            )
            return (None, None, None)
        else:
            # Start with empty mismatch and consistent_links matrixes
            mismatch_matrix = pd.DataFrame(
                0, index=obs["SS_name"], columns=obs["SS_name"]
            )
            mismatch_matrix.columns.name = "i"
            mismatch_matrix.index.name = "i_m1"

            consistent_links_matrix = mismatch_matrix.copy()

            for atom in seq_atoms:
                # Repeat the i and i-1 observed shifts into matrixes, and
                # transpose one of them.
                # This is the fastest way I've found to calculate this.
                i_atom = (
                    obs[atom]
                    .repeat(len(obs.index))
                    .values.reshape([len(obs.index), -1])
                )
                i_m1_atom = (
                    obs[atom + "_m1"]
                    .repeat(len(obs.index))
                    .values.reshape([len(obs.index), -1])
                    .transpose()
                )

                mismatch_atom = pd.DataFrame(i_atom - i_m1_atom)

                mismatch_atom.columns = obs["SS_name"]
                mismatch_atom.columns.name = "i"
                mismatch_atom.index = obs["SS_name"]
                mismatch_atom.index.name = "i_m1"

                consistent_links_atom = abs(mismatch_atom) < threshold

                # Make a note of NA positions, and set them to default value
                mismatch_atom = mismatch_atom.fillna(0.0)
                consistent_links_atom = consistent_links_atom.fillna(0.0)

                # Update mismatch and consistent links matrixes
                mismatch_matrix = mismatch_matrix.combine(
                    abs(mismatch_atom), np.maximum
                )
                consistent_links_matrix = (
                    consistent_links_matrix + consistent_links_atom
                )

            allowed_links_matrix = (mismatch_matrix < 0.2).astype(int)
            for i in allowed_links_matrix.index:
                allowed_links_matrix.loc[i, i] = 0  # Set diagonal to zero

            self.config.logger.info(
                "Calculated mismatch and consistent_links matrixes (%dx%d)",
                mismatch_matrix.shape[0],
                mismatch_matrix.shape[1],
            )

            return (mismatch_matrix, consistent_links_matrix, allowed_links_matrix)

    def calc_triplet_prob_matrix(
        self, prob_matrix, allowed_links_matrix, normalise_by=None
    ):
        """Calculate the probability that each spin system is assigned to a given residue,
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
        if normalise_by == "Res":
            triplet_prob_matrix = triplet_prob_matrix.div(
                triplet_prob_matrix.sum(axis=0), axis="columns"
            )
        elif normalise_by == "SS":
            triplet_prob_matrix = triplet_prob_matrix.div(
                triplet_prob_matrix.sum(axis=1), axis="rows"
            )

        return triplet_prob_matrix

    def make_assign_df(self, obs, preds, log_prob_matrix, matching):
        """Make a dataframe with full assignment information, given a dataframe
        of SS_name and Res_name.

        Matching may have additional columns, which will also be kept.
        """
        obs = obs.copy()
        obs.index.name = "name"  # Needed to avoid error when merging dataframes.
        preds = preds.copy()
        log_prob_matrix = log_prob_matrix.copy()
        valid_atoms = list(self.config.atom_set)
        extra_cols = set(matching.columns).difference({"SS_name", "Res_name"})

        if not {"SS_name", "Res_name"}.issubset(matching.columns):
            self.config.logger.warning(
                "Cannot make assignment dataframe, as matching"
                + " dataframe does not have the correct columns"
            )

        assign_df = pd.merge(
            matching,
            preds.loc[
                :, ["Res_N", "Res_type", "Res_name", "Pred_N", "Pred_name", "Dummy_res"]
            ],
            on="Res_name",
            how="left",
        )
        assign_df = assign_df[
            [
                "Res_name",
                "Res_N",
                "Res_type",
                "SS_name",
                "Pred_N",
                "Pred_name",
                "Dummy_res",
            ]
            + list(extra_cols)
        ]
        assign_df = pd.merge(
            assign_df,
            obs.loc[:, obs.columns.isin(["SS_name", "Dummy_SS"] + valid_atoms)],
            on="SS_name",
            how="left",
        )
        assign_df = pd.merge(
            assign_df,
            preds.loc[:, preds.columns.isin(valid_atoms + ["Res_name"])],
            on="Res_name",
            suffixes=("", "_pred"),
            how="left",
        )

        assign_df.index = assign_df.SS_name  # Needed to match Log_prob to correct row.
        assign_df.index.names = ["index"]
        assign_df["Log_prob"] = df_lookup(
            log_prob_matrix, assign_df["SS_name"], assign_df["Res_name"]
        )
        # Careful above not to get rows/columns confused

        assign_df = assign_df.sort_values(by="Res_N")

        return assign_df

    def find_best_assignment(
        self,
        score_matrix,
        maximise=True,
        inc=None,
        exc=None,
        exc_mask=None,
        dummy_rows=[],
        dummy_cols=[],
        return_none_all_dummy=False,
    ):
        """Use the Hungarian algorithm to find the highest scoring assignment,
        with constraints. Generalised so it can be used for assigning either
        sequentially or based on predictions.

        Returns a data frame with the index and column names of the matching.

        Parameters
        score_matrix: a pandas dataframe containing the scores, and with row and
            column labels
        maximise: if True, returns the assignment with the maximum score, otherwise return
            assignment with the minimum score
        inc: a DataFrame of (row, col) pairs which must be part of the assignment.
            First column has the index names, second has the column names.
        exc: a DataFrame of (row, col) pairs which may not be part of the assignment.
        exc_mask: a DataFrame with the same size and indexes as score_matrix.
            Contains boolean values, with True meaning an assignment is to be excluded.
            Note that it is possible to provide both exc and exc_mask
        """
        score_matrix = score_matrix.copy()

        row_name = score_matrix.index.name
        col_name = score_matrix.columns.name

        self.config.logger.debug("Started linear assignment")

        if maximise:
            penalty = -2 * score_matrix.abs().max().max()
        else:
            penalty = 2 * score_matrix.abs().max().max()
        if exc_mask is not None:
            score_matrix = score_matrix.mask(exc_mask, other=penalty)

        if inc is not None:
            # Check for conflicting entries in inc
            conflicts = inc[row_name].duplicated(keep=False) | inc[col_name].duplicated(
                keep=False
            )
            if any(conflicts):
                self.config.logger.warning(
                    "Warning: entries in inc conflict with one another - dropping conflicts"
                )
                inc = inc[~conflicts]

            if exc is not None:
                # Check constraints are consistent
                # Get rid of any entries in exc which share a Res or SS with inc
                exc_in_inc = exc[row_name].isin(inc[row_name]) | exc[col_name].isin(
                    inc[col_name]
                )
                if any(exc_in_inc):
                    self.config.logger.warning(
                        "Some values in exc are also found in inc, so are redundant."
                    )
                    exc = exc.loc[~exc_in_inc, :]

            # Removed fixed assignments from probability matrix and obs, preds
            # dataframes. Latter is needed if inc includes any dummy SS/res,
            # and to detect if the reduced data is entirely dummies
            score_matrix_reduced = score_matrix.drop(index=inc[row_name]).drop(
                columns=inc[col_name]
            )
            self.config.logger.debug(
                "%d assignments were fixed, %d remain to be assigned"
                % (len(inc), len(score_matrix_reduced.index))
            )
        else:
            score_matrix_reduced = score_matrix

        # If the matrix consists entirely of dummy rows or columns, the
        # assignment will not be meaningful. In this case, may want to return None
        if return_none_all_dummy:
            if dummy_rows is not None:
                if set(score_matrix_reduced.index).issubset(dummy_rows):
                    self.config.logger.debug("Score matrix includes only dummy rows")
                    return None
            if dummy_cols is not None:
                if set(score_matrix_reduced.columns).issubset(dummy_cols):
                    self.config.logger.debug("Score matrix includes only dummy columns")
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
                if r[col_name] in dummy_cols:
                    score_matrix_reduced.loc[r[row_name], dummy_cols] = penalty
                elif r[row_name] in dummy_rows:
                    score_matrix_reduced.loc[dummy_rows, r[col_name]] = penalty
                else:
                    score_matrix_reduced.loc[r[row_name], r[col_name]] = penalty

            self.config.logger.debug(
                "Penalised %d excluded row,column pairs" % len(exc.index)
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
            matching = pd.concat([inc, matching_reduced]).sort_values("Res_name")
            return matching.reset_index(drop=True)
        else:
            matching_reduced = matching_reduced.sort_values("Res_name")
            return matching_reduced.reset_index(drop=True)

    def preprocess_obs_preds(self, obs, preds):
        obs, preds = super().preprocess_obs_preds(obs, preds)
        # Remove N_m1 and N_p1, not sure why they're there in the old one and not this one but they're not good
        preds = preds.drop(["N_m1", "N_p1"], axis=1)
        return obs, preds

    def validate_data(
        self, obs: pd.DataFrame, preds: pd.DataFrame, sequence: Optional[pd.DataFrame]
    ) -> None:
        if sequence is None:
            raise ValueError(
                "Triplet strategy requires sequence, please add a sequence using --sequence_file, use a nef file that contains a sequence, or use a different strategy"
            )
