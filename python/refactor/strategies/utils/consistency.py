import numpy as np
import pandas as pd

from modules.assigner_functions import df_lookup


def calc_mismatch_matrix(obs, threshold=0.2, logger=None):
    """Calculate pairwise sequential carbon mismatches for observed shifts."""
    carbons = pd.Series(["C", "CA", "CB"])
    seq_atoms = carbons[
        carbons.isin(obs.columns) & (carbons + "_m1").isin(obs.columns)
    ]

    if seq_atoms.empty:
        if logger:
            logger.warning(
                "No sequential links in data - mismatch matrix not calculated."
            )
        return None

    mismatch_matrix = pd.DataFrame(0, index=obs["SS_name"], columns=obs["SS_name"])
    mismatch_matrix.columns.name = "i"
    mismatch_matrix.index.name = "i_m1"
    consistent_links_matrix = mismatch_matrix.copy()

    for atom in seq_atoms:
        mismatch_atom = pd.DataFrame(
            obs[atom].to_numpy()[:, None] - obs[f"{atom}_m1"].to_numpy()[None, :],
            index=obs["SS_name"],
            columns=obs["SS_name"],
        )
        mismatch_atom.columns.name = "i"
        mismatch_atom.index.name = "i_m1"

        consistent_links_atom = abs(mismatch_atom) < threshold
        na_mask = mismatch_atom.isna()
        mismatch_atom = mismatch_atom.mask(na_mask, 0)
        consistent_links_atom = consistent_links_atom.mask(na_mask, 0)

        mismatch_matrix = mismatch_matrix.combine(abs(mismatch_atom), np.maximum)
        consistent_links_matrix = consistent_links_matrix + consistent_links_atom

    if logger:
        logger.info(
            "Calculated mismatch and consistent_links matrixes (%dx%d)",
            mismatch_matrix.shape[0],
            mismatch_matrix.shape[1],
        )
    return mismatch_matrix, consistent_links_matrix


def add_consistency_info(
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
        "C",
        "CA",
        "CB",
        "C_m1",
        "CA_m1",
        "CB_m1",
    }.issubset(assign_df.columns) or any(
        atom in assign_df.columns and f"{atom}_m1" in assign_df.columns
        for atom in ("C", "CA", "CB")
    )

    if not sequential_atoms:
        assign_df["Max_mismatch_m1"] = np.nan
        assign_df["Max_mismatch_p1"] = np.nan
        assign_df["Num_good_links_m1"] = 0
        assign_df["Num_good_links_p1"] = 0
        assign_df["Confidence"] = "Undefined"
        if logger:
            logger.warning(
                "Couldn't calculate assignment confidence: "
                "need sequential atom information"
            )
        return assign_df

    if mismatch_matrix is None or consistent_links_matrix is None:
        raise ValueError(
            "mismatch_matrix and consistent_links_matrix are required "
            "when adding consistency information"
        )

    matching = assign_df.copy()
    matching.index = matching["Res_name"]
    matching.index.name = None
    tmp = pd.concat(
        [matching, preds[["Res_name_m1", "Res_name_p1"]]],
        axis=1,
    )
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

    tmp_m1 = tmp.dropna(subset=["SS_name_m1"])
    tmp_p1 = tmp.dropna(subset=["SS_name_p1"])
    tmp["Max_mismatch_m1"] = pd.Series(
        list(
            df_lookup(
                mismatch_matrix,
                tmp_m1["SS_name_m1"],
                tmp_m1["SS_name"],
            )
        ),
        index=tmp_m1.index,
    )
    tmp["Max_mismatch_p1"] = pd.Series(
        list(
            df_lookup(
                mismatch_matrix,
                tmp_p1["SS_name"],
                tmp_p1["SS_name_p1"],
            )
        ),
        index=tmp_p1.index,
    )
    tmp["Max_mismatch"] = tmp[["Max_mismatch_m1", "Max_mismatch_p1"]].max(axis=1)
    tmp["Num_good_links_m1"] = pd.Series(
        list(
            df_lookup(
                consistent_links_matrix,
                tmp_m1["SS_name_m1"],
                tmp_m1["SS_name"],
            )
        ),
        index=tmp_m1.index,
    )
    tmp["Num_good_links_p1"] = pd.Series(
        list(
            df_lookup(
                consistent_links_matrix,
                tmp_p1["SS_name"],
                tmp_p1["SS_name_p1"],
            )
        ),
        index=tmp_p1.index,
    )
    tmp["Num_good_links_m1"] = tmp["Num_good_links_m1"].fillna(0)
    tmp["Num_good_links_p1"] = tmp["Num_good_links_p1"].fillna(0)
    tmp["Num_good_links"] = tmp["Num_good_links_m1"] + tmp["Num_good_links_p1"]

    for direction in ("m1", "p1"):
        tmp[f"Conf_{direction}"] = "N"
        tmp.loc[tmp[f"Num_good_links_{direction}"] > 0, f"Conf_{direction}"] = "W"
        tmp.loc[tmp[f"Num_good_links_{direction}"] > 1, f"Conf_{direction}"] = "S"
        tmp.loc[
            tmp[f"Max_mismatch_{direction}"] > threshold,
            f"Conf_{direction}",
        ] = "X"

    tmp["Conf2"] = tmp["Conf_m1"] + tmp["Conf_p1"]
    tmp["Conf2"] = tmp["Conf2"].replace(
        ["WS", "NS", "XS", "NW", "XW", "XN"],
        ["SW", "SN", "SX", "WN", "WX", "NX"],
    )
    tmp["Confidence"] = tmp["Conf2"].replace(
        {
            "SS": "High",
            "SW": "High",
            "SN": "High",
            "WW": "High",
            "SX": "Medium",
            "WN": "Medium",
            "WX": "Low",
            "NX": "Unreliable",
            "XX": "Unreliable",
            "NN": "Undefined",
        }
    )
    if logger:
        summary = ", ".join(
            f"{count} {confidence}"
            for confidence, count in tmp["Confidence"]
            .value_counts()
            .sort_values(ascending=False)
            .items()
        )
        logger.info("Calculated assignment confidence: " + summary)

    consistency = tmp[
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
        ]
    ]
    return assign_df.merge(consistency, how="left")
