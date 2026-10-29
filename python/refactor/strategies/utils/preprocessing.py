import numpy as np
import pandas as pd

from refactor.SNAPSconfig import SNAPSConfig


def preprocess_obs_preds(config: SNAPSConfig, obs: pd.DataFrame, preds: pd.DataFrame):
    """Perform preprocessing on the observed and predicted shifts to prepare
    them for calculation of the log probability matrix.

    1) Remove proline residues from predictions
    2) Discard atom types that are not used or aren't present in both
    3) Add dummy rows to obs or preds to bring them to the same length
    """

    #### Delete any prolines in preds
    print("Removing %d prolines from predictions" % sum(preds["Res_type"] == "P"))
    preds = preds.drop(preds.index[preds["Res_type"] == "P"])

    # Remove references to deleted residues from Res_name_m1/p1
    preds.loc[~preds["Res_name_m1"].isin(preds["Res_name"]), "Res_name_m1"] = np.nan
    preds.loc[~preds["Res_name_p1"].isin(preds["Res_name"]), "Res_name_p1"] = np.nan

    #### Restrict atom types
    # self.pars["atom_set"] is the set of atoms to be used in the analysis

    # Get all non-atom columns
    all_atoms = config.supported_atom_types
    obs_metadata = sorted(set(obs.columns).difference(all_atoms))
    preds_metadata = sorted(set(preds.columns).difference(all_atoms))
    # Get atoms that are in atom_set *and* present in both obs and preds
    shared_atoms = sorted(
        list(
            config.atom_set.intersection(obs.columns).intersection(
                preds.columns
            )
        )
    )  # It has to be a set before this point but has to be a list past this point, oddly
    obs = obs.loc[:, obs_metadata + shared_atoms]
    preds = preds.loc[:, preds_metadata + shared_atoms]

    print("Restricted atom types to %s" % ",".join(shared_atoms))

    #### Create dummy rows
    # Create columns to keep track of dummy status
    preds["Dummy_res"] = False
    obs["Dummy_SS"] = False

    N = len(obs.index)
    M = len(preds.index)

    if N > M:  # If there are more spin systems than predictions
        dummies = pd.DataFrame(
            np.nan,
            columns=preds.columns,
            index=[
                config.dummy_predicted_prefix + str(i)
                for i in 1 + np.arange(N - M)
            ],
        )
        dummies["Res_name"] = dummies.index
        dummies["Dummy_res"] = True
        # preds = preds.append(dummies)
        preds = pd.concat([preds, dummies])
        print("Added %d dummy predicted residues" % len(dummies.index))
    elif M > N:
        dummies = pd.DataFrame(
            np.nan,
            columns=obs.columns,
            index=[
                config.dummy_observed_prefix + str(i)
                for i in 1 + np.arange(M - N)
            ],
        )
        dummies["SS_name"] = dummies.index
        dummies["Dummy_SS"] = True
        # obs = obs.append(dummies)
        obs = pd.concat([obs, dummies])
        print("Added %d dummy observed residues" % len(dummies.index))

    return (obs, preds)
