import pandas as pd
from modules.assigner_functions import df_lookup


def make_assign_df(
    config,
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
    obs.index.name = "name"  # Needed to avoid error when merging dataframes.
    valid_atoms = sorted(config.atom_set)
    extra_cols = set(matching.columns).difference({"SS_name", "Res_name"})

    if not {"SS_name", "Res_name"}.issubset(matching.columns):
        config.logger.warning(
            "Cannot make assignment dataframe, as matching"
            + " dataframe does not have the correct columns"
        )

    assign_df = pd.merge(
        matching,
        preds.loc[:, ["Res_N", "Res_type", "Res_name", "Dummy_res"]],
        on="Res_name",
        how="left",
    )
    assign_df = assign_df[
        ["Res_name", "Res_N", "Res_type", "SS_name", "Dummy_res"] + sorted(extra_cols)
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

    # assign_df["Log_prob"] = log_prob_matrix.lookup(
    #                                     assign_df["SS_name"],
    #                                     assign_df["Res_name"])
    assign_df["Log_prob"] = df_lookup(
        log_prob_matrix, assign_df["SS_name"], assign_df["Res_name"]
    )
    # Careful above not to get rows/columns confused

    assign_df = assign_df.sort_values(by="Res_N")

    return assign_df
