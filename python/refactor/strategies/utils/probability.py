from math import log10

import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal

from refactor.SNAPSconfig import SNAPSConfig


def calc_log_prob_matrix(
    config: SNAPSConfig,
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
    atoms = sorted(config.atom_set.intersection(obs.columns))

    if config.pred_correction:
        # Import parameters for correcting the shifts
        lm_pars = pd.read_csv(config.pred_correction_file, index_col=0)
        config.logger.info(
            "Imported pred_correction info from %s"
            % config.pred_correction_file
        )

    if config.delta_correlation:
        # Import parameters describing the delta correlations
        # Note: this also sorts the atom types in d_mean and c_cov into the
        # same order as the 'atoms' list defined above.
        if config.pred_correction:
            d_mean = pd.read_csv(
                config.delta_correlation_mean_corrected_file,
                header=None,
                index_col=0,
            ).loc[atoms, 1]
            d_cov = pd.read_csv(
                config.delta_correlation_cov_corrected_file,
                index_col=0,
            ).loc[atoms, atoms]
            config.logger.info(
                "Imported delta_correlation info from %s and %s"
                % (
                    config.delta_correlation_mean_corrected_file,
                    config.delta_correlation_cov_corrected_file,
                )
            )
        else:
            d_mean = pd.read_csv(
                config.delta_correlation_mean_file,
                header=None,
                index_col=0,
            ).loc[atoms, 1]
            d_cov = pd.read_csv(
                config.delta_correlation_cov_file, index_col=0
            ).loc[atoms, atoms]
            config.logger.info(
                "Imported delta_correlation info from %s and %s"
                % (
                    config.delta_correlation_mean_file,
                    config.delta_correlation_cov_file,
                )
            )

        # For the delta_correlation method, we need to store the prediction
        # errors for each atom type for analysis at the end.
        delta_list = []

    log_prob_matrix = pd.DataFrame(0, index=obs.index, columns=preds.index)
    log_prob_matrix.index.name = "SS_name"
    log_prob_matrix.columns.name = "Res_name"

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

        # If correcting the predictions, subtract a linear function of the
        # observed shift from each predicted shift
        if config.pred_correction:
            config.logger.info("Calculating corrections to predicted shifts")
            preds_corr_atom = preds_atom
            for res in preds["Res_type"].dropna().unique():
                if (atom + "_" + res) in lm_pars.index:
                    # Find shifts of the current atom/residue combination
                    mask = (lm_pars["Atom_type"] == atom) & (
                        lm_pars["Res_type"] == res
                    )

                    # Look up model parameters
                    grad = lm_pars.loc[mask, "Grad"].tolist()[0]
                    offset = lm_pars.loc[mask, "Offset"].tolist()[0]

                    # Make the correction
                    if atom in ("C_m1", "CA_m1", "CB_m1"):
                        mask = preds["Res_type_m1"] == res
                        preds_corr_atom.loc[:, mask] = (
                            preds_atom.loc[:, mask]
                            - grad * obs_atom.loc[:, mask]
                            - offset
                        )
                    else:
                        mask = preds["Res_type"] == res
                        preds_corr_atom.loc[:, mask] = (
                            preds_atom.loc[:, mask]
                            - grad * obs_atom.loc[:, mask]
                            - offset
                        )

            delta_atom = preds_corr_atom - obs_atom
        else:
            delta_atom = preds_atom - obs_atom

        if config.delta_correlation:
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
                    scale=config.atom_sd[atom]
                    * config.shift_scale_factor,
                ),
                index=obs.index,
                columns=preds.index,
            )

            # Replace former NA values with a default value
            prob_atom[na_mask] = log10(config.default_probability)

            # Add to the log prob matrix
            log_prob_matrix = log_prob_matrix + prob_atom

    if config.delta_correlation:
        config.logger.info("Accounting for correlated prediction errors")

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
            log_prob_matrix + log10(config.default_probability) * na_matrix
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
    log_prob_matrix[log_prob_matrix.isna()] = 2 * np.nanmin(log_prob_matrix.values)
    log_prob_matrix.loc[obs["Dummy_SS"], :] = 0
    log_prob_matrix.loc[:, preds["Dummy_res"]] = 0

    log_prob_matrix.index.name = "SS_name"
    log_prob_matrix.columns.name = "Res_name"

    config.logger.info(
        "Calculated log probability matrix (%dx%d)",
        log_prob_matrix.shape[0],
        log_prob_matrix.shape[1],
    )

    return log_prob_matrix
