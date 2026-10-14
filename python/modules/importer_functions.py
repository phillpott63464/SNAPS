import pandas as pd
from NEF_reader import read_nef_obs_shifts_from_file_to_pandas

def import_observed_shifts(filename, filetype, SS_num=False, chain='A'):
    """ Import a chemical shift list

    filename: Path to text file containing chemical shifts.
    filetype: Allowed values are "snaps", "ccpn", "sparky", "xeasy",
        "nmrpipe" or "mars"
        The "ccpn" option is for importing a Resonance table exported from
        Analysis v2.x. The "snaps" option is for importing an unassigned
        shift table previously exported from SNAPS
    SS_num: If true, will extract the longest number from the SS_name and
    treat it as the residue number. Without this, it is not possible to get
    the i-1 shifts for each spin system.

    """
    # Import from file
    if filetype == "snaps":
        obs = pd.read_table(filename)
    elif filetype == "ccpn":
        obs = pd.read_table(filename)
        obs = obs.loc[:, ["Residue", "Assign Name", "Shift"]]
        obs.columns = ["SS_name", "Atom_type", "Shift"]
        obs["Atom_type"] = obs["Atom_type"].str.upper()
    elif filetype == "sparky":
        obs = pd.read_table(filename, sep=r"\s+")
        obs = obs.loc[:, ["Group", "Atom", "Shift"]]
        obs.columns = ["SS_name", "Atom_type", "Shift"]
        obs.loc[obs["Atom_type"] == "HN", "Atom_type"] = "H"
    elif filetype == "xeasy":
        obs = pd.read_table(filename, sep=r"\s+",
                            header=None, na_values="999.000",
                            names=["i", "Shift", "SD", "Atom_type", "SS_name"])
        obs = obs.loc[:, ["SS_name", "Atom_type", "Shift"]]
        obs["SS_name"] = obs["SS_name"].astype(str)
        obs = obs.dropna(subset=["Shift"])
        obs.loc[obs["Atom_type"] == "HN", "Atom_type"] = "H"
    elif filetype == "nmrpipe":
        # Work out where the column names and data are
        with open(filename, 'r') as f:
            for num, line in enumerate(f, 1):
                if line.find("VARS") > -1:
                    colnames_line = num

        obs = pd.read_table(filename, sep=r"\s+", skiprows=colnames_line + 1,
                            names=["SS_name", "Res_type", "Atom_type", "Shift"])
        obs = obs.loc[:, ["SS_name", "Atom_type", "Shift"]]
        obs["SS_name"] = obs["SS_name"].astype(str)
        obs.loc[obs["Atom_type"] == "HN", "Atom_type"] = "H"
    elif filetype == "mars":
        obs_wide = pd.read_table(filename, sep=r"\s+", na_values="-")
        obs_wide = obs_wide.rename(columns={"CO": "C", "CO-1": "C_m1",
                                            "CA-1": "CA_m1", "CB-1": "CB_m1"})
        obs_wide = obs_wide.drop(columns="HA-1")
        obs_wide["SS_name"] = obs_wide.index.astype(str)
        obs = obs_wide.melt(id_vars="SS_name",
                            value_vars=["H", "HA", "N", "C", "CA",
                                        "CB", "C_m1", "CA_m1", "CB_m1"],
                            var_name="Atom_type", value_name="Shift")
    elif filetype == 'nef':
        obs = read_nef_obs_shifts_from_file_to_pandas(filename, chain)

    #        elif filetype == "nmrstar":
    #            tmp = nmrstarlib.read_files(filename)
    #            return(tmp)
    else:
        print("import_obs_shifts: invalid filetype '%s'." % (filetype))
        return (None)

    # Restrict to backbone atom types
    obs = obs.loc[obs["Atom_type"].isin(["H", "HA", "N", "C", "CA", "CB",
                                         "C_m1", "CA_m1", "CB_m1"]), :]

    # Convert from long to wide
    obs = obs.pivot(index="SS_name", columns="Atom_type", values="Shift")
    obs.insert(0, "SS_name", obs.index.values)

    # Extract residue number from SS_name (if present), and get m1 shifts
    if SS_num:
        obs["Res_N"] = obs["SS_name"].str.extract(r"(\d+)").astype(int)

        obs.index = obs["Res_N"]
        obs_m1 = obs[list({"C", "CA", "CB"}.intersection(obs.columns))]
        obs_m1.index = obs_m1.index + 1
        obs_m1.columns = obs_m1.columns + "_m1"
        obs = pd.merge(obs, obs_m1, how="left",
                       left_index=True, right_index=True)
        obs = obs.drop(columns="Res_N")

        # Set index back to SS_name
        obs.index = obs["SS_name"]
        obs.index.name = None

    # the index name shouldn't match a column name!
    obs.index.name = None
    return obs