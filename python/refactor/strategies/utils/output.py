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
    of SS_name and Res_name. Matching may have additional columns, which
    will also be kept.
    """
    obs = obs.copy()
    preds = preds.copy()
    obs.index.name = "name"  # Needed to avoid error when merging dataframes.
    valid_atoms = sorted(config.atom_set)
    extra_cols = set(matching.columns).difference({"SS_name", "Res_name"})
    is_triplet = config.strategy == "triplet"

    if not {"SS_name", "Res_name"}.issubset(matching.columns):
        config.logger.warning(
            "Cannot make assignment dataframe, as matching"
            + " dataframe does not have the correct columns"
        )

    pred_info = ["Res_N", "Res_type", "Res_name", "Dummy_res"]
    pred_name_columns = ["Pred_N", "Pred_type", "Pred_name"]
    include_pred_name_columns = is_triplet and set(pred_name_columns).issubset(
        preds.columns
    )
    if include_pred_name_columns:
        pred_info += pred_name_columns

    assign_df = pd.merge(
        matching,
        preds.loc[:, pred_info].reset_index(drop=True),
        on="Res_name",
        how="left",
    )

    if include_pred_name_columns:
        assignment_columns = [
            "Res_name",
            "Res_N",
            "Res_type",
            "SS_name",
            "Pred_N",
            "Pred_name",
            "Dummy_res",
        ]
    else:
        assignment_columns = [
            "Res_name",
            "Res_N",
            "Res_type",
            "SS_name",
            "Dummy_res",
        ]

    assign_df = assign_df[assignment_columns + sorted(extra_cols)]
    assign_df = pd.merge(
        assign_df,
        obs.loc[:, obs.columns.isin(["SS_name", "Dummy_SS"] + valid_atoms)],
        on="SS_name",
        how="left",
    )
    assign_df = pd.merge(
        assign_df,
        preds.loc[:, preds.columns.isin(valid_atoms + ["Res_name"])].reset_index(
            drop=True
        ),
        on="Res_name",
        suffixes=("", "_pred"),
        how="left",
    )

    original_index = assign_df.index
    assign_df.index = assign_df["SS_name"]
    assign_df["Log_prob"] = df_lookup(
        log_prob_matrix, assign_df["SS_name"], assign_df["Res_name"]
    )
    assign_df.index = original_index

    assign_df = assign_df.sort_values(
        by="Res_N",
        na_position="last" if is_triplet else "last",
    )

    return assign_df
