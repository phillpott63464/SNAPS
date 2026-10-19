#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines a class with functions related to the actual assignment process, and
outputting the results. The class also stores intermediate variables (such as
the imported chemical shifts and the log probability matrix).

@author: aph516
"""

import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import gridplot
from bokeh.models.ranges import Range1d
from bokeh.models import WheelZoomTool, LabelSet, ColumnDataSource, Span
from bokeh.io import export_png
from bokeh.embed import json_item
from scipy.stats import norm, multivariate_normal
from scipy.optimize import linear_sum_assignment
from math import log10, sqrt
from copy import deepcopy
from pathlib import Path
#from Bio.SeqUtils import seq1
from Bio import SeqIO
from distutils.util import strtobool
from collections import namedtuple
from sortedcontainers import SortedListWithKey
#from textwrap import dedent
import logging
import yaml
from pathlib import Path

def df_lookup(df, row_labels, col_labels, index="rows"):
    """Look up a series of locations in a data frame df, with the row and
    column indicies given by row_labels and col_labels. This replaces the
    DataFrame.lookup() that was part of previous versions of pandas.

    Returns
    A series containing the looked-up values

    Parameters
    df
    row_labels: the row indicies of the values to be fetched
    col_labels: the column indicies of the values to be fetched
    eg. if row_labels=[1,3] and col_labels=['b','e'], the function would return a Series
    consisting of df.loc[1,'b'] and df.loc[3,'e']

    Taken from https://stackoverflow.com/questions/71156253/alternative-to-df-lookuprow-col?noredirect=1&lq=1
    """
    if index=="cols":
        return(pd.Series(df.to_numpy()[df.index.get_indexer(row_labels), df.columns.get_indexer(col_labels)], index=col_labels))
    else:
        return(pd.Series(df.to_numpy()[df.index.get_indexer(row_labels), df.columns.get_indexer(col_labels)], index=row_labels))


class SNAPS_assigner:
    # Functions
    def __init__(self):
        self.obs = None
        self.preds = None
        self.seq_df = None
        self.all_preds = None
        self.log_prob_matrix = None
        self.mismatch_matrix = None
        self.consistent_links_matrix = None
        self.assign_df = None
        self.alt_assign_df = None
        self.best_match_indexes = None
        self.pars = {"pred_correction": False,
                "delta_correlation": False,
                "atom_set": {"H","N","HA","C","CA","CB","C_m1","CA_m1","CB_m1"},
                "atom_sd": {'H':0.1711, 'N':1.1169, 'HA':0.1231,
                            'C':0.5330, 'CA':0.4412, 'CB':0.5163,
                            'C_m1':0.5530, 'CA_m1':0.4412, 'CB_m1':0.5163},
                "seq_link_threshold": 0.2}
        self.logger = logging.getLogger("SNAPS.assigner")

        if False:
            # To suppress logging messages from within this class, set this block to True
            log_handler = logging.NullHandler()
            self.logger.addHandler(log_handler)
            self.logger.propagate = False

    def copy(self):
        """Create a deep copy of the data frame, including new copies of each dataframe

        Needed because copy.deepcopy() can't seem to copy loggers that are in use.
        """

        tmp = self.logger

        self.logger = None
        a = deepcopy(self)

        self.logger = tmp
        a.logger = tmp

        return(a)

    def read_config_file(self, filename):
        """Read a configuration file written in YAML format

        Returns
        A dictionary containing the imported parameters.

        Parameters
        filename: A path to the configuration file"""
        f = open(filename, 'r')

        self.pars = yaml.safe_load(f)
        self.pars["atom_set"] = set(self.pars["atom_set"])

        # Check whether all necessary parameters have been imported
        required_pars = {"atom_set", "atom_sd",
                         "iterate_until_consistent","seq_link_threshold",
                         "delta_correlation",
                         "delta_correlation_mean_file",
                         "delta_correlation_cov_file",
                         "pred_correction",
                         "pred_correction_file",
                         "delta_correlation_mean_corrected_file",
                         "delta_correlation_cov_corrected_file"}
        #TODO: Need to add parameters relating to alt_assignments
        if required_pars.issubset(set(self.pars.keys())):
            self.logger.info("All required parameters imported")
        else:
            missing_pars = required_pars.difference(set(self.pars.keys()))
            self.logger.error("Some required parameters were missing/not imported: %s",
                                ", ".join(missing_pars))

        par_dir = Path(filename).parent

        for par_name, par in self.pars.items():
            if par_name.endswith('_file'):
                self.pars[par_name] = str(par_dir / par)

        self.logger.info("Finished reading in config parameters from %s" 
                         % filename)
        return(self.pars)

    def import_pred_shifts(self, filename, filetype, offset=0):
        from modules.assigner_functions import import_predicted_shifts

        preds, seq_df = (import_predicted_shifts(filename=filename, filetype=filetype, seq_df=self.seq_df, logger=self.logger, offset=offset))

        self.preds = preds
        self.seq_df = seq_df # This is not used in any of the further code - it is only used in import_sequence, which is never called in SNAPS.py. Not sure why it exists.

    def prepare_obs_preds(self):
        """Perform preprocessing on the observed and predicted shifts to prepare
        them for calculation of the log probability matrix.

        1) Remove proline residues from predictions
        2) Discard atom types that are not used or aren't present in both
        3) Add dummy rows to obs or preds to bring them to the same length

        Returns the modified obs and preds DataFrames.
        """
        from modules.assigner_functions import prepare_obs_preds

        obs, preds = prepare_obs_preds(obs=self.obs, preds=self.preds, atom_set=self.pars['atom_set'], logger=self.logger)

        self.obs = obs

        self.preds = preds

        return(self.obs, self.preds)

    def calc_log_prob_matrix(self, atom_sd=None, sf=1, default_prob=0.01):
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
        sf: A scale factor that multiplies the atom_sd
        default_prob: penalty for missing data
        """

        # Use default atom_sd values if not defined
        if atom_sd==None:
            atom_sd = self.pars["atom_sd"]

            from modules.assigner_functions import calc_log_prob_matrix

        obs = self.obs.copy()
        preds = self.preds.copy()
        pred_correction = self.pars["pred_correction"]
        pred_correction_file = self.pars["pred_correction_file"]
        delta_correlation = self.pars["delta_correlation"]
        delta_correlation_mean_file = self.pars["delta_correlation_mean_file"]
        delta_correlation_cov_file = self.pars["delta_correlation_cov_file"]
        delta_correlation_mean_corrected_file = self.pars["delta_correlation_mean_corrected_file"]
        delta_correlation_cov_corrected_file = self.pars["delta_correlation_cov_corrected_file"]

        print(delta_correlation_mean_file)

        self.log_prob_matrix = calc_log_prob_matrix(
            obs=obs,
            preds=preds,
            pred_correction=pred_correction,
            delta_correlation=delta_correlation,
            pred_correction_file=pred_correction_file,
            delta_correlation_mean_file=delta_correlation_mean_file,
            delta_correlation_cov_file=delta_correlation_cov_file,
            delta_correlation_mean_corrected_file=delta_correlation_mean_corrected_file,
            delta_correlation_cov_corrected_file=delta_correlation_cov_corrected_file,
            logger=self.logger,
        )

        self.log_prob_matrix.to_pickle('output/log_prog.pickle')

        return self.log_prob_matrix

    def calc_mismatch_matrix(self, threshold=0.2):
        """Calculate matrix of the mismatch between i and i-1 observed carbon
        shifts for all spin system pairs. Also, make matrix of number of
        consistent sequential links for all spin system pairs.

        Parameters
        threshold: The cutoff value beyond which a seq link is considered bad
        """
        obs = self.obs.copy()

        # First check if there are any sequential atoms
        carbons = pd.Series(["C","CA","CB"])
        carbons_m1 = carbons + "_m1"
        seq_atoms = carbons[carbons.isin(obs.columns) &
                            carbons_m1.isin(obs.columns)]

        if seq_atoms.size==0:
            # You can't make a mismatch matrix
            self.logger.warning("No sequential links in data - mismatch matrix not calculated.")
            return(None)
        else:
            # Start with empty mismatch and consistent_links matrixes
            mismatch_matrix = pd.DataFrame(0,
                                           index=obs["SS_name"],
                                           columns=obs["SS_name"])
            mismatch_matrix.columns.name = "i"
            mismatch_matrix.index.name = "i_m1"

            consistent_links_matrix = mismatch_matrix.copy()

            for atom in seq_atoms:
                # Repeat the i and i-1 observed shifts into matrixes, and
                # transpose one of them.
                # This is the fastest way I've found to calculate this.
                i_atom = (obs[atom].repeat(len(obs.index)).values.
                          reshape([len(obs.index),-1]))
                i_m1_atom = (obs[atom+"_m1"].repeat(len(obs.index)).values.
                          reshape([len(obs.index),-1]).transpose())

                mismatch_atom = pd.DataFrame(i_atom - i_m1_atom)

                mismatch_atom.columns = obs["SS_name"]
                mismatch_atom.columns.name = "i"
                mismatch_atom.index = obs["SS_name"]
                mismatch_atom.index.name = "i_m1"

                consistent_links_atom = (abs(mismatch_atom) < threshold)

                # Make a note of NA positions, and set them to default value
                na_mask = np.isnan(mismatch_atom)
                mismatch_atom[na_mask] = 0
                consistent_links_atom[na_mask] = 0

                # Update mismatch and consistent links matrixes
                mismatch_matrix = mismatch_matrix.combine(abs(mismatch_atom), np.maximum)
                consistent_links_matrix = consistent_links_matrix + consistent_links_atom

            self.logger.info("Calculated mismatch and consistent_links matrixes (%dx%d)",
                         mismatch_matrix.shape[0], mismatch_matrix.shape[1])

            self.mismatch_matrix = mismatch_matrix
            self.consistent_links_matrix = consistent_links_matrix
            return(self.mismatch_matrix, consistent_links_matrix)

    def find_best_assignment(self, score_matrix, maximise=True, inc=None, exc=None,
                             dummy_rows=None, dummy_cols=None, return_none_all_dummy=False):
        from modules.assigner_functions import find_best_assignment
        matching = find_best_assignment(
            score_matrix=score_matrix,
            maximise=maximise,
            inc=inc,
            exc=exc,
            dummy_rows=dummy_rows,
            dummy_cols=dummy_cols,
            return_none_all_dummy=return_none_all_dummy,
            logger=self.logger,
        )
        return matching

    def make_assign_df(self, matching, set_assign_df=False):
        """Make a dataframe with full assignment information, given a dataframe
        of SS_name and Res_name.

        Matching may have additional columns, which will also be kept.
        """
        obs = self.obs.copy()
        obs.index.name = "name" # Needed to avoid error when merging dataframes.
        preds = self.preds.copy()
        log_prob_matrix = self.log_prob_matrix.copy()
        valid_atoms = list(self.pars["atom_set"])
        extra_cols = set(matching.columns).difference({"SS_name","Res_name"})

        if not {"SS_name","Res_name"}.issubset(matching.columns):
            self.logger.warning("Cannot make assignment dataframe, as matching"
                                +" dataframe does not have the correct columns")

        assign_df = pd.merge(matching,
                             preds.loc[:,["Res_N","Res_type", "Res_name",
                                    "Dummy_res"]],
                             on="Res_name", how="left")
        assign_df = assign_df[["Res_name","Res_N","Res_type","SS_name",
                               "Dummy_res"]+list(extra_cols)]
        assign_df = pd.merge(assign_df,
                             obs.loc[:, obs.columns.isin(
                                     ["SS_name","Dummy_SS"]+valid_atoms)],
                             on="SS_name", how="left")
        assign_df = pd.merge(assign_df,
                             preds.loc[:, preds.columns.isin(
                                     valid_atoms+["Res_name"])],
                             on="Res_name", suffixes=("","_pred"), how="left")

        # assign_df["Log_prob"] = log_prob_matrix.lookup(
        #                                     assign_df["SS_name"],
        #                                     assign_df["Res_name"])
        assign_df["Log_prob"] = df_lookup(log_prob_matrix,
                                          assign_df["SS_name"],
                                          assign_df["Res_name"])
        # Careful above not to get rows/columns confused

        assign_df = assign_df.sort_values(by="Res_N")

        if set_assign_df:
            self.assign_df = assign_df

        return(assign_df)

    def assign_from_preds(self, set_assign_df=False):
        """Assign the observed spin systems using predicted shifts only

        This function essentially wraps around find_best_assignment() and
        make_assign_df()
        """
        matching = self.find_best_assignment(self.log_prob_matrix, maximise=True)
        assign_df = self.make_assign_df(matching, set_assign_df)

        self.logger.info("Finished calculating best assignment based on predictions")
        return(assign_df)

    def sequential_atoms_present(self, atom_list):
        """Returns true if a pair of atoms in atom_list are sequential, otherwise False

        Acceptable pairs are (C, C_m1), (CA, C_Am1), (CB,C_Bm1)"""

        i_atoms = pd.Series(["C","CA","CB"])
        i_m1_atoms = pd.Series(["C_m1","CA_m1","CB_m1"])
        seq_atoms = i_atoms.isin(atom_list) & i_m1_atoms.isin(atom_list)

        return (seq_atoms.any())

    def check_matching_consistency(self, matching, threshold=0.2):
        """Calculate mismatch scores and assignment confidence for a given matching

        Returns a DataFrame with Res_name, SS_name and columns containing the
        mismatch with adjacent residues, the number of good links, and the
        assignment confidence based on these factors

        Parameters
        matching: a DataFrame with Res_name and SS_name columns
        threshold: the maximum allowed mismatch for a good sequential link
        """
        # Create mismatch matrix if it doesn't already exist
        if self.mismatch_matrix is None:
            self.calc_mismatch_matrix()

        # Add Res_name_m1 and Res_name_p1 columns to matching DataFrame
        matching.index = matching["Res_name"]
        matching.index.name = None

        tmp = pd.concat([matching, self.preds[["Res_name_m1","Res_name_p1"]]], axis=1)

        # Add a SS_name_m1 and SS_name_p1 columns
        tmp = tmp.merge(matching[["SS_name"]], how="left", left_on="Res_name_m1",
                        right_index=True, suffixes=("","_m1"))
        tmp = tmp.merge(matching[["SS_name"]], how="left", left_on="Res_name_p1",
                        right_index=True, suffixes=("","_p1"))

        # Filter out rows with NaN's in SS_name_m1/p1
        tmp_m1 = tmp.dropna(subset=["SS_name_m1"])
        tmp_p1 = tmp.dropna(subset=["SS_name_p1"])

        # Get the mismatches
        tmp["Max_mismatch_m1"] = pd.Series(list(df_lookup(self.mismatch_matrix,
                                tmp_m1["SS_name_m1"], tmp_m1["SS_name"])), index=tmp_m1.index)
        tmp["Max_mismatch_p1"] = pd.Series(list(df_lookup(self.mismatch_matrix,
                                tmp_p1["SS_name"], tmp_p1["SS_name_p1"])), index=tmp_p1.index)
        tmp["Max_mismatch"] = tmp[["Max_mismatch_m1","Max_mismatch_p1"]].max(axis=1)

        tmp["Num_good_links_m1"] = pd.Series(list(df_lookup(self.consistent_links_matrix,
                                tmp_m1["SS_name_m1"], tmp_m1["SS_name"])), index=tmp_m1.index)
        tmp["Num_good_links_p1"] = pd.Series(list(df_lookup(self.consistent_links_matrix,
                                tmp_p1["SS_name"], tmp_p1["SS_name_p1"])), index=tmp_p1.index)
        tmp["Num_good_links_m1"].fillna(0, inplace=True)
        tmp["Num_good_links_p1"].fillna(0, inplace=True)
        tmp["Num_good_links"] = tmp["Num_good_links_m1"] + tmp["Num_good_links_p1"]

        # Add an assignment confidence column
        tmp["Conf_m1"] = "N"
        tmp.loc[tmp["Num_good_links_m1"]>0,"Conf_m1"] = "W"
        tmp.loc[tmp["Num_good_links_m1"]>1,"Conf_m1"] = "S"
        tmp.loc[tmp["Max_mismatch_m1"]>threshold,"Conf_m1"] = "X"

        tmp["Conf_p1"] = "N"
        tmp.loc[tmp["Num_good_links_p1"]>0,"Conf_p1"] = "W"
        tmp.loc[tmp["Num_good_links_p1"]>1,"Conf_p1"] = "S"
        tmp.loc[tmp["Max_mismatch_p1"]>threshold,"Conf_p1"] = "X"

        tmp["Conf2"] = tmp["Conf_m1"]+tmp["Conf_p1"]
        # Reorder letters eg WS -> SW
        tmp["Conf2"] = tmp["Conf2"].replace(["WS","NS","XS","NW","XW","XN"],
                                           ["SW","SN","SX","WN","WX","NX"])

        tmp["Confidence"] = tmp["Conf2"]
        tmp["Confidence"] = tmp["Confidence"].replace(["SS","SW","SN","WW"],"High")
        tmp["Confidence"] = tmp["Confidence"].replace(["SX","WN"],"Medium")
        tmp["Confidence"] = tmp["Confidence"].replace("WX","Low")
        tmp["Confidence"] = tmp["Confidence"].replace(["NX","XX"],"Unreliable")
        tmp["Confidence"] = tmp["Confidence"].replace("NN","Undefined")

        # Summarise confidence of results
        summary_str = ", ".join(
            [str(count) + " "+conf for conf, count in
             tmp["Confidence"].value_counts().sort_values(ascending=False).items()])
        self.logger.info("Calculated assignment confidence: "+summary_str)

        return(tmp.loc[:,["SS_name","Res_name","Max_mismatch_m1","Max_mismatch_p1","Max_mismatch",
                    "Num_good_links_m1","Num_good_links_p1","Num_good_links","Confidence"]])

    def add_consistency_info(self, input_assign_df=None, threshold=0.2):
        """ Find maximum mismatch and number of 'significant' mismatches for
        each residue

        threshold: Minimum carbon shift difference for sequential residues to
            count as mismatched
        """

        # If the user hasn't specified an assign_df, use one already calculated
        # for this SNAPS_assigner instance
        if input_assign_df is None:
            set_assign_df = True
            assign_df = self.assign_df.copy()
        else:
            set_assign_df = False
            assign_df = input_assign_df.copy()

        # First check if there are any sequential atoms
        if self.sequential_atoms_present(assign_df.columns):
            tmp = self.check_matching_consistency(assign_df, threshold)
            assign_df = assign_df.merge(tmp, how="left")
            self.logger.info("Finished checking assignment consistency")
        else:
            # You can't do a comparison without sequential atoms
            assign_df["Max_mismatch_m1"] = np.NaN
            assign_df["Max_mismatch_p1"] = np.NaN
            assign_df["Num_good_links_m1"] = 0
            assign_df["Num_good_links_p1"] = 0
            assign_df["Confidence"] = "Undefined"

            self.logger.warning("Couldn't calculate assignment confidence: "+
                                "need sequential atom information")

        if set_assign_df:
            self.assign_df = assign_df
        return(assign_df)

    def find_consistent_assignments(self, threshold=0.2, set_assign_df=False):
        """Try to find a consistent set of assignments by optimising both match
        to predictions and mismatches between adjacent residues

        Returns an assign_df DataFrame, but does not modify the class

        Parameters
        threshold: the maximum allowed mismatch for a good sequential link
        """

        self.logger.info("Started assigning based on predictions and sequential links")
        assign_df0 = self.assign_from_preds()
        assign_df0 = self.add_consistency_info(assign_df0, threshold)
        N_HM_conf0 = assign_df0["Confidence"].isin(["High","Medium"]).sum()
        self.logger.info("At the current stage, there are "+str(N_HM_conf0)+
                         " high or medium confidence assignments")

        while True:
            # Find all residues with an adjacent confident residue
            HM_conf_res = assign_df0.loc[assign_df0["Confidence"].isin(["High","Medium"]),
                                         "Res_name"]

            #Make a dataframe for looking up which spin system is assigned to each residue
            tmp = assign_df0[["Res_name","SS_name"]]
            tmp.index = tmp["Res_name"]

            # Limit their assignment options to consistent spin systems
            a = self.copy()

            for res in HM_conf_res:
                # Get the neighbouring residues
                res_m1 = self.preds.loc[res,"Res_name_m1"]
                res_p1 = self.preds.loc[res,"Res_name_p1"]

                ss = tmp.SS_name[res]

                # (Need to be careful with NaN values at this point)
                penalty = a.log_prob_matrix.min().min()
                if not pd.isna(res_m1):
                    #Get list of inconsistent i-1 spin systems
                    allowed_ss = self.mismatch_matrix.loc[:,ss]<=threshold
                    # Set the inconsistent spins systems to have a high log_prob
                    # penalty is added so in case no spin systems are allowed -
                    # this way, the predictions do still have an influence.
                    a.log_prob_matrix.loc[~allowed_ss, res_m1] += penalty
                    #print(res, self.mismatch_matrix.index[self.mismatch_matrix.loc[:,ss]<=threshold])
                if not pd.isna(res_p1):
                    #Get list of inconsistent i+1 spin systems
                    allowed_ss = self.mismatch_matrix.loc[ss,:]<=threshold
                    #Set the inconsistent spins systems to have a high log_prob
                    a.log_prob_matrix.loc[~allowed_ss, res_p1] += penalty
                    #print(res, self.mismatch_matrix.index[self.mismatch_matrix.loc[ss,:]<=threshold])

            # Rerun assignment and see how many are consistent now
            assign_df1 = a.assign_from_preds()
            assign_df1 = a.add_consistency_info(assign_df1, threshold)
            N_HM_conf1 = assign_df1["Confidence"].isin(["High","Medium"]).sum()
            self.logger.info("At the current stage, there are "+str(N_HM_conf1)+
                         " high or medium confidence assignments")

            # Check if the number of high/medium confidence assignments has increased
            # If yes, try another round of consrtained assignment
            # If no, stick with the previous best assignment.
            if N_HM_conf1 > N_HM_conf0:
                assign_df0 = assign_df1
                N_HM_conf0 = N_HM_conf1
            else:
                break

        self.logger.info("At the current stage, there are "+str(N_HM_conf0)+
                         " high or medium confidence assignments")

        if set_assign_df:
            self.assign_df = assign_df0

        return(assign_df0)

    def output_shiftlist(self, filepath, format="sparky",
                         confidence_list=["High","Medium","Low","Unreliable","Undefined"]):
        """Export a chemical shift list, in a variety of formats

        Parameters
        filepath: the file the shifts will be written to
        format: the format of the chemical shift file (sparky, xeasy or nmrpipe)
        confidence_list: Only residues with confidence in this list will be output
        """

        # Limit confidence levels and discard unneeded columns and rows
        atoms = {"H","N","HA","C","CA","CB"}.intersection(self.assign_df.columns)
        df_wide = self.assign_df.loc[self.assign_df["Confidence"].isin(confidence_list),:]
        self.logger.info("Chemical shift export: restricted confidence levels to %s"
                         % ", ".join(confidence_list))
        df_wide.loc[~(df_wide["Dummy_res"] | df_wide["Dummy_SS"]),
                    ["Res_N","Res_type"]+list(atoms)]
        #TODO: This discards all i-1 information, even though that may be useful
        # in some cases (eg for getting proline shifts). Is it possible to make
        # better use of this?

        # Check in case no shifts were selected
        if df_wide.empty:
            print("No chemical shifts selected for export.")
            self.logger.warning("Cannot export chemical shifts: No atom types selected")
            # Create an empty file
            df_wide.to_csv(filepath, sep="\t", float_format="%.3f",
                           index=True, header=False)
            return(None)

        # Reshape dataframe from wide to long format, sort and remove NAs
        df = df_wide.melt(id_vars=["Res_N","Res_type"], value_vars=atoms,
                          var_name="Atom_type", value_name="Shift")
        df = df.sort_values(["Res_N","Atom_type"])
        df = df.dropna(subset=["Res_N"])
        df["Res_N"] = df["Res_N"].astype(int)

        # Make format-specific modifications and export
        if format=="sparky":
            df["Group"] = df["Res_type"] + df["Res_N"].astype(str)

            df = df.rename(columns={"Atom_type":"Atom"})
            df.loc[df["Atom"]=="H","Atom"] = "HN"

            nuc_dict = {"HN":"1H", "N":"15N", "HA":"1H",
                        "C":"13C", "CA":"13C", "CB":"13C"}
            df["Nuc"] = [nuc_dict[a] for a in df["Atom"]]

            output_df = df[["Group","Atom","Nuc","Shift"]]
            output_df["Sdev"] = 0.0
            output_df["Assignments"] = int(1)

            output_df = output_df.dropna()

            if filepath is not None:
                output_df.to_csv(filepath, sep="\t", float_format="%.3f",
                                 index=False)
        elif format=="xeasy":
            df.loc[df["Atom_type"]=="H","Atom_type"] = "HN"

            output_df = df[["Shift","Atom_type","Res_N"]]
            output_df.insert(1, "Sdev", 0)

            output_df["Shift"] = output_df["Shift"].fillna(999.0)
            output_df = output_df.dropna()
            output_df = output_df.reset_index(drop=True)

            if filepath is not None:
                output_df.to_csv(filepath, sep="\t", float_format="%.3f",
                                 index=True, header=False)
        elif format=="nmrpipe":
            df.loc[df["Atom_type"]=="H","Atom_type"] = "HN"

            output_df = df[["Res_N", "Res_type", "Atom_type", "Shift"]]

            output_df["Shift"] = output_df["Shift"].fillna(9999.0)
            output_df = output_df.dropna()

            # NmrPipe requres a sequence, but we don't necessarily have a complete one
            # Piece togther what we can from all_preds, and put X at other positions
            tmp = pd.DataFrame({"Res_N":np.arange(self.all_preds["Res_N"].min(),
                                                  self.all_preds["Res_N"].max()+1)})
            tmp = tmp.merge(self.all_preds[["Res_N","Res_type"]], how="left", on="Res_N")
            tmp["Res_type"] = tmp["Res_type"].fillna("X")
            seq = tmp["Res_type"].sum()     # Convert to a string
            self.logger.info("Sequence contains %d residues, of which %d have unknown type"
                             % (len(seq), sum(tmp["Res_type"]=="X")))

            if filepath is not None:
                f = open(filepath, 'w+')
                f.write("REMARK Chemical shifts (automatically assigned using SNAPS)\n\n")
                f.write("DATA FIRST_RESID %d\n\n" % tmp["Res_N"].min())
                f.write("DATA SEQUENCE %s\n\n" % seq)
                f.write("VARS   RESID RESNAME ATOMNAME SHIFT\n")
                f.write("FORMAT %4d   %1s     %4s      %8.3f\n\n")

                for i in output_df.index:
                    f.write("%4d %1s %4s %8.3f\n" % (output_df.loc[i, "Res_N"],
                                                     output_df.loc[i, "Res_type"],
                                                     output_df.loc[i, "Atom_type"],
                                                     output_df.loc[i, "Shift"]))
                f.close()

        else:
            self.logger.warning("Cannot export chemical shifts: "+
                                "format string '%s' not recognised." % format)
            return(None)

        self.logger.info("Wrote %d chemical shifts from %d residues to %s" %
                         (len(df.index),
                          len(df["Res_N"].drop_duplicates()),
                          filepath))
        return(output_df)

    def plot_strips(self, outfile=None, format="html", return_json=True, plot_width=1000):
        """Make a strip plot of the assignment.

        Uses bokeh module for plotting. Returns the bokeh plot object.

        outfile: if defined, the plot will be saved to this location
        format: type of output. Either "html" or "png"
        return_json: if tue, return the plot as a json object
        plot_width: The width of the output plot in pixels
        """
        df = self.assign_df
        plotlist = []

        if not self.sequential_atoms_present(df.columns):
            # You can't draw a strip plot
            print("No sequential links in data - strip plot not drawn.")
            return(None)
        else:
            #### Make a dataframe containing information needed for plotting
            # Add residues from seq_df that aren't in assign_df (eg. prolines)
            df = pd.merge(self.seq_df, df, how="left", on=["Res_name","Res_N","Res_type"])
            # Replace NA values where necessary
            mask = df["Dummy_res"].isna()
            df.loc[mask,"Dummy_res"] = False
            df.loc[mask,"Dummy_SS"] = True

            # Create a temporary plot that defines the common x-axis
            tmp_plt = figure(x_range=df["Res_name"].tolist())

            #### Make the confidence plot
            plt = figure(title=("Confidence plot"
                                "(pan/zoom tools can be accessed at top right; "
                                "mouse over the plot to see observed spin system name)"),
                        x_range=tmp_plt.x_range,
                        y_range = Range1d(0, 1.5),
                        tools="xpan, xwheel_zoom,hover,save,reset",
                        tooltips=[("Pred", "@Res_name"),("Obs","@SS_name")],
                        height=100, width=plot_width)

            # Create a colour map based on confidence
            colourmap = {"High":"green",
                         "Medium":"yellowgreen",
                         "Low":"orange",
                         "Unreliable":"red",
                         "Undefined":"grey"}

            # Plot the peaks
            for k in colourmap.keys():
                tmp = ColumnDataSource(df[df["Confidence"]==k])
                plt.vbar(x="Res_name", top=1, width=1,
                         color=colourmap[k], legend_label=k, source=tmp)

            # Set legend properties
            plt.legend.orientation = "horizontal"
            plt.legend.location = "top_center"
            plt.legend.padding = 0
            plt.legend.margin = 0

            # Change axis label orientation
            plt.xaxis.major_label_orientation = 3.14159/2
            plt.axis.visible = False


            plotlist = plotlist + [plt]

            #### Make the mismatch plot
            plt = figure(title="Mismatch plot",
                            y_axis_label="Mismatch (ppm)",
                            x_range=tmp_plt.x_range,
                            tools="xpan, xwheel_zoom,save,reset",
                            height=200, width=plot_width)

            plt.vbar(x=df["Res_name"],
                     top=df[["Max_mismatch_m1","Max_mismatch_p1"]].max(axis=1),
                     width=1)

            # Draw a line showing the threshold for mismatches
            threshold_line = Span(location=self.pars["seq_link_threshold"],
                                  dimension="width", line_dash="dashed",
                                  line_color="red")
            plt.add_layout(threshold_line)

            # Change axis label orientation
            plt.xaxis.major_label_orientation = 3.14159/2
            plotlist = plotlist + [plt]

            #### Make the strip plot
            # Work out which sequential carbons are present
            carbons = pd.Series(["C","CA","CB"])
            carbons_m1 = carbons + "_m1"
            seq_atoms = carbons[carbons.isin(df.columns) &
                            carbons_m1.isin(df.columns)]

            for atom in seq_atoms:
                # Setup plot
                plt = figure(y_axis_label=atom+" (ppm)",
                             x_range=tmp_plt.x_range,
                             tools="xpan, xwheel_zoom,save,reset",
                             height=200, width=plot_width)

                #### Plot the vertical lines
                vlines = df.loc[~df["Dummy_res"].astype(bool), ["Res_name",atom,atom+"_m1"]]
                # Introduce NaN's to break the line into discontinuous segments
                # "_z" in name is to ensure it sorts after the atom+"_m1" shifts
                # eg. CA, CA_m1, CA_z
                vlines[atom+"_z"] = np.NaN
                # Convert from wide to long
                vlines = vlines.melt(id_vars=["Res_name"],
                                     value_vars=[atom, atom+"_m1", atom+"_z"],
                                     var_name="Atom_type",
                                     value_name="Shift")
                vlines = vlines.sort_values(["Res_name","Atom_type"],
                                            ascending=[True,False])
                plt.line(vlines["Res_name"], vlines["Shift"],
                         line_color="black", line_dash="dashed")

                #### Plot the horizontal lines
                hlines = df.loc[~df["Dummy_res"].astype(bool), ["Res_N","Res_name",atom,atom+"_m1"]]
                # Introduce NaN's to break the line into discontinuous segments
                # "_a" in name ensures it sorts between the atom and atom+"_m1" shifts
                # eg. CA, CA_a, CA_m1
                hlines[atom+"_a"] = np.NaN
                # Convert from wide to long
                hlines = hlines.melt(id_vars=["Res_name"],
                                     value_vars=[atom, atom+"_m1", atom+"_a"],
                                     var_name="Atom_type",
                                     value_name="Shift")
                hlines = hlines.sort_values(["Res_name","Atom_type"],
                                            ascending=[True,False])
                plt.line(hlines["Res_name"], hlines["Shift"],
                         line_color="black", line_dash="solid")

                # Draw circles at the observed chemical shifts
                plt.circle(df["Res_name"], df[atom], fill_color="blue", size=5)
                plt.circle(df["Res_name"], df[atom+"_m1"], fill_color="red", size=5)

                # Reverse the y axis and set range:
                plt.y_range = Range1d(
                        df[[atom,atom+"_m1"]].max().max()+5,
                        df[[atom,atom+"_m1"]].min().min()-5)

                # Change axis label orientation
                plt.xaxis.major_label_orientation = 3.14159/2
                plt.xaxis.visible = False  # Make axis invisible by default

                plotlist = plotlist + [plt]

            # Make x axis on the last strip plot visible
            plotlist[-1].xaxis.visible = True

            #### Put everything together
            p = gridplot(plotlist, ncols=1)

            if outfile is not None:
                Path(outfile).resolve().parents[1].mkdir(parents=True, exist_ok=True)
                if format=="html":
                    output_file(outfile)
                    save(p)
                elif format=="png":
                    export_png(p, outfile)

            if return_json:
                return(json_item(p, "strip_plot"))
            else:
                return(p)

    def plot_hsqc(self, outfile=None, format="html",
                  return_json=True, plot_width=750):
        """Plot an HSQC coloured by prediction confidence


        Uses bokeh module for plotting. Returns the bokeh plot object.

        outfile: if defined, the plot will be saved to this location
        format: type of output. Either "html" or "png"
        return_json: if tue, return the plot as a json object
        plot_width: The width of the output plot in pixels
        """

        assign_df = self.assign_df.copy()

        plt = figure(title="HSQC",
                    x_axis_label="1H (ppm)",
                    y_axis_label="15N (ppm)",
                    height=plot_width, width=plot_width)

        # Create a colour map based on confidence
        colourmap = {"High":"green",
                         "Medium":"yellowgreen",
                         "Low":"orange",
                         "Unreliable":"red",
                         "Undefined":"grey"}

        # Plot the peaks
        for k in colourmap.keys():
            tmp = assign_df[assign_df["Confidence"]==k]
            plt.circle(tmp["H"], tmp["N"], color=colourmap[k], legend=k)

        # Label the points
        df = ColumnDataSource(assign_df)
        labels = LabelSet(x="H", y="N", text="Res_name", source=df, text_font_size="10pt")
        plt.add_layout(labels)

        # Reverse the axes and set range
        plt.x_range = Range1d(assign_df["H"].max()+1, assign_df["H"].min()-1)
        plt.y_range = Range1d(assign_df["N"].max()+5, assign_df["N"].min()-5)

        # Change legend layout
        plt.legend.orientation = "horizontal"

        if outfile is not None:
            Path(outfile).resolve().parents[1].mkdir(parents=True, exist_ok=True)
            if format=="html":
                output_file(outfile)
                save(plt)
            elif format=="png":
                export_png(plt, outfile)

        if return_json:
            return(json_item(plt))
        else:
            return(plt)
