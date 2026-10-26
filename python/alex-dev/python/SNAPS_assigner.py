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
from Bio import SeqIO, Align
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

def heatmap(x, y, z, x_range=None, y_range=None, palette=None, z_range=(None, None),
            x_label="x", y_label="y", z_label="z", flip_x_axis=False, flip_y_axis=False,
            colorbar=True, plot_height=750, plot_width=750, **kwargs):
    """Create a heatmap for x, y, z data where the x and y data are categorial.
    Borrowed from https://justinbois.github.io/bootcamp/2023_epfl/exercise_solutions/exercise_5/exercise_5.5_solution.html

    Parameters
    ----------
    x : array_like
        x-values for heat map. Assumed to be categorical. Any entries are
        converted to strings.
    y : array_like
        y-values for heat map. Assumed to be categorical. Any entries are
        converted to strings.
    z : array_like
        z-values for heat map. These data are quantitative and displayed
        with color.
    x_range : array_like
        Array of unique values that determines the values of the x-axis.
    y_range : array_like
        Array of unique values that determines the values of the y-axis.
    palette : List of hex colors, default bokeh.palettes.Viridis256
        Color palette to use to make linear color mapper for heat map.
    z_range : 2-tuple, default (None, None)
        Range of allowed z-values. If an entry is None, the min or max
        is used.
    x_label : str, defualt "z"
        Label to be used in tool tips for the x-values.
    y_label : str, defualt "z"
        Label to be used in tool tips for the y-values.
    z_label : str, defualt "z"
        Label to be used in tool tips for the z-values.
    flip_x_axis : bool, default False
        If True, x-axis is reversed.
    flip_y_axis : bool, default False
        If True, y-axis is reversed.
    colorbar : bool, default True
        If True, display color bar.
    kwargs : dict
        All other kwargs are passed to bokeh.plotting.figure() when
        setting up the plot.

    Returns
    -------
    output : Bokeh plotting object
        Heatmap plot.
    """
    import bokeh.io
    import bokeh.models
    import bokeh.palettes
    import bokeh.plotting

    # Convert x and y values to strings; assuming evenly spread
    x_str = [str(x_val) for x_val in x]
    y_str = [str(y_val) for y_val in y]

    # Ranges of z-values
    z_min = z.min() if z_range[0] is None else z_range[0]
    z_max = z.max() if z_range[1] is None else z_range[1]

    # Categorical axis values
    if x_range is None:
        x_range = [str(x_val) for x_val in sorted(np.unique(x))]
    if y_range is None:
        y_range = [str(y_val) for y_val in sorted(np.unique(y))]
    if flip_x_axis:
        x_range = x_range[::-1]
    if flip_y_axis:
        y_range = y_range[::-1]

    # Set up defaults
    x_axis_label = kwargs.pop("x_axis_label", x_label)
    y_axis_label = kwargs.pop("y_axis_label", y_label)
    tools = kwargs.pop("tools", "pan,box_zoom,wheel_zoom,reset,hover,save")
    tooltips = kwargs.pop(
        "tooltips", [(x_label, "@x"), (y_label, "@y"), (z_label, "@z")]
    )
    toolbar_location = kwargs.pop("toolbar_location", "above")
    frame_height = kwargs.pop("frame_height", None)
    frame_width = kwargs.pop("frame_width", None)

    # Adjust frame heights and widths to have square rectangles
    if frame_width is not None:
        if frame_height is None:
            frame_height = frame_width * len(y_range) // len(x_range)
    else:
        if frame_height is None:
            frame_height = 250
        frame_width = frame_height * len(x_range) // len(y_range)

    # Data source
    source = bokeh.models.ColumnDataSource(
        dict(x_str=x_str, y_str=y_str, x=x, y=y, z=z)
    )

    # Color mapper
    if palette is None: palette = bokeh.palettes.Viridis256
    mapper = bokeh.models.LinearColorMapper(palette=palette, low=z_min, high=z_max)

    # Figure
    p = bokeh.plotting.figure(
        x_range=x_range,
        y_range=y_range,
        frame_width=frame_width,
        frame_height=frame_height,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        tools=tools,
        tooltips=tooltips,
        toolbar_location=toolbar_location,
        # height=plot_height,
        # width=plot_width,
        **kwargs
    )

    p.rect(
        x="x_str",
        y="y_str",
        width=frame_width / frame_height * len(y_range) / len(x_range),
        height=frame_height / frame_width * len(x_range) / len(y_range),
        source=source,
        fill_color={"field": "z", "transform": mapper},
        line_color=None,
    )

    # Add color bar
    color_bar = bokeh.models.ColorBar(
        color_mapper=mapper, major_label_text_font_size="8px", border_line_color=None,
    )
    p.add_layout(color_bar, "right")

    return(p)

class SNAPS_assigner:
    # Functions
    def __init__(self):
        self.obs = None
        self.preds = None
        self.aligned_preds = None
        self.seq_df = None
        self.all_preds = None
        self.alignment = None
        self.prob_matrix = None
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

        # Find the full paths for parameter files (eg d_mean.csv)
        # In the YAML file, only the file name is given, so find the parent folder 
        # of the config file and add it
        par_dir = Path(filename).parent

        for par_name, par in self.pars.items():
            if par_name.endswith('_file'):
                self.pars[par_name] = str(par_dir / par)

        self.logger.info("Finished reading in config parameters from %s" 
                         % filename)
        return(self.pars)

    def import_sequence(self, filename, seq_numbering=1, filetype="text"):
        """ Imports the protein sequence

        The sequence information is used to fill in any gaps if prediction
        information is missing. It can also be used if the residue numbering is
        discontinuous. The sequence file should be in FASTA format, and the
        sequence numbering should be either:
            1) the residue number of the first amino acid, or
            2) a comma separated list of residue ranges eg -5:10,15:100 (this is
            for proteins with discontinuous sequence numbering)
        If the file contains multiple sequences, only the first will be used.

        Parameters
        filename: path to file containing sequence information
        """

        # Parse the sequence
        if filetype=="fasta":
            fasta_records = SeqIO.parse(open(filename),"fasta")
            record1 = next(fasta_records)
            seq_list = list(str(record1.seq))
        elif filetype=="text":
            with open(filename, 'r') as file:
                seq_list = list("".join(file.read().split()))

        # Parse the sequence numbering information and make a list of residue numbers
        if type(seq_numbering)==int:
            res_N_start = seq_numbering
            res_N_list = list(range(res_N_start, res_N_start+len(seq_list)))
        else:
            tmp = seq_numbering.split(",")
            if len(tmp)==1:     # If id was "123" or "123-456"
                tmp2 = tmp[0].split(":")
                if len(tmp2)==1:        # If id was "123"
                    res_N_start = int(tmp2[0])
                    res_N_list = list(range(res_N_start, res_N_start+len(seq_list)))
                else:                   # If id was "123-456"
                    start, end = tmp2
                    res_N_list = list(range(int(start), int(end)+1))
            else:               # If id was "123-200,300-456"
                res_N_list = []
                for x in tmp:
                    start, end = x.split(":")
                    res_N_list += list(range(int(start), int(end)+1))

        # Check res_N_list and seq_list are the same length, and correct if not
        if len(res_N_list) > len(seq_list):
            res_N_list = res_N_list[0:len(seq_list)]
            self.logger.warning("The residue range provided is longer than the"+
                                " length of the sequence, so has been truncated")
        elif len(res_N_list) < len(seq_list):
            extra_needed = len(seq_list) - len(res_N_list)
            last_N = res_N_list[-1]
            res_N_list += list(range(last_N, last_N + extra_needed))
            self.logger.warning("The residue range provided is shorter than the"+
                                " length of the sequence, so has been extended")

        # Make a dataframe
        seq_df = pd.DataFrame({"Seq_N":list(range(1,1+len(res_N_list))), "Res_N":res_N_list,"Res_type":seq_list})
            # Seq_N numbers the residues starting from 1.
        seq_df["Res_name"] = seq_df["Res_N"].astype(str) + seq_df["Res_type"]
        seq_df["Res_name"] = seq_df["Res_name"].str.rjust(5)
                # Pad Res_name to constant length

        # Add columns for i-1 and i+1 residue names
        seq_df["Res_name_m1"] = seq_df["Res_name"].shift(1)
        seq_df["Res_name_p1"] = seq_df["Res_name"].shift(-1)

        seq_df.index = seq_df["Res_name"]
        seq_df.index.name = None

        self.seq_df = seq_df
        self.logger.info("Finished importing a sequence of length %d" %
                         len(seq_df.index))

        return(self.seq_df)

    def import_pred_shifts(self, filename, filetype, offset=0):
        """ Import predicted chemical shifts from a ShiftX2 results file.

        Returns
        A DataFrame containing the predicted shifts, or None if the import failed.

        Parameters
        filename: path to file containing predicted shifts
        filetype: either "shiftx2" or "sparta+"
        offset: an optional integer to add to the residue number.
        """

        #### Import the raw data
        if filetype == "shiftx2":
            preds_long = pd.read_csv(filename)
            if any(preds_long.columns == "CHAIN"):
                if len(preds_long["CHAIN"].unique())>1:
                    self.logger.warning(
                            """Chain identifier dropped - if multiple chains are
                            present in the predictions, they will be merged.""")
                preds_long = preds_long.drop("CHAIN", axis=1)
            preds_long = preds_long.reindex(columns=["NUM","RES","ATOMNAME",
                                                     "SHIFT"])
            preds_long.columns = ["Res_N","Res_type","Atom_type","Shift"]
        elif filetype == "sparta+":
            # Work out where the column names and data are
            with open(filename, 'r') as f:
                for num, line in enumerate(f, 1):
                    if line.find("VARS")>-1:
                        colnames_line = num
                        colnames = line.split()[1:]
                        break

            preds_long = pd.read_table(filename, sep="\s+", names=colnames,
                                       skiprows=colnames_line+1)
            preds_long = preds_long.reindex(columns=["RESID","RESNAME",
                                                     "ATOMNAME","SHIFT"])
            preds_long.columns = ["Res_N","Res_type","Atom_type","Shift"]

            # Sparta+ uses HN for backbone amide proton - convert to H
            preds_long.loc[preds_long["Atom_type"]=="HN", "Atom_type"] = "H"
        else:
            self.logger.error("""Invalid predicted shift type: '%s'. Allowed
                              options are 'shiftx2' or 'sparta+'""" % (filetype))
            return(None)

        self.logger.info("Imported %d predicted chemical shifts from %s"
                         % (len(preds_long.index), filename))

        #### Initial processing and conversion from long to wide
        # Add sequence number offset and create residue names
        preds_long["Res_N"] = preds_long["Res_N"] + offset
        preds_long.insert(1, "Res_name", (preds_long["Res_N"].astype(str) +
                  preds_long["Res_type"]))
        # Left pad with spaces to a constant length (helps with sorting)
        preds_long["Res_name"] = preds_long["Res_name"].str.rjust(5)

        # Convert from long to wide format
        preds = preds_long.pivot(index="Res_N", columns="Atom_type",
                                 values="Shift")
        preds.index.name = None

        # Add the other residue name and type back in
        tmp = preds_long[["Res_N","Res_type","Res_name"]]
        tmp = tmp.drop_duplicates(subset="Res_name")
        tmp.index = tmp["Res_N"]
        tmp.index.name = None
        preds = pd.concat([tmp, preds], axis=1)

        #### Make consistent with seq_df (and create if it doesn't already exist)
        # TODO: Maybe this should be split off into a separate function?
        # If seq_df is missing, create it based on preds
        seq_df = self.seq_df

        if seq_df is None:
            seq_df = preds.copy()[["Res_N","Res_type","Res_name"]]

            # If there are any missing residue numbers, create them
            min_N = seq_df["Res_N"].min()
            max_N = seq_df["Res_N"].max()
            missing_residue_numbers = (set(range(min_N, max_N+1)).
                                       difference(seq_df["Res_N"]))
            if len(missing_residue_numbers)>0:
                tmp = pd.DataFrame({"Res_N":list(missing_residue_numbers),
                                    "Res_type":"X","Res_name":np.nan})
                tmp["Res_name"] = tmp["Res_N"].astype(str) + tmp["Res_type"]
                tmp["Res_name"] = tmp["Res_name"].str.rjust(5)
                tmp.index = tmp["Res_N"]
                tmp.index.name = None
                #seq_df = seq_df.append(tmp).sort_index()
                seq_df = pd.concat([seq_df, tmp]).sort_index()

        # Add/delete residues from preds so it matches seq_df
        # Only keep predictions that are in seq_df
        tmp = len(preds.index)
        preds = preds[preds["Res_N"].isin(seq_df["Res_N"])]
        tmp2 = tmp - len(preds.index)
        if tmp2>0:
            self.logger.info(("Predictions for %d residues were discarded "
                                "because they were not present in the imported "
                                "sequence file") % tmp2)

        # Add predictions for any residue number that is in seq_df but not preds
        tmp = pd.DataFrame(data=seq_df[~seq_df["Res_N"].isin(preds["Res_N"])],columns=preds.columns)
        #preds = preds.append(tmp)
        preds = pd.concat([preds, tmp])
        if len(tmp.index)>0:
            self.logger.info(("%d residues from the sequence were missing "
                                "from the predictions") % len(tmp.index))

        # If Res_name is inconsistent between seq_df and preds, make a compromise
        preds.index = preds["Res_N"]
        preds = preds.sort_index()
        seq_df.index = seq_df["Res_N"]
        seq_df = seq_df.sort_index()
        mask = preds["Res_name"] != seq_df["Res_name"]

        if any(mask):
            ambiguous_res_names = (seq_df.loc[mask, "Res_name"] + "(" +
                                   preds.loc[mask, "Res_type"] + "?)")
            preds.loc[mask, "Res_name"] = ambiguous_res_names
            seq_df.loc[mask, "Res_name"] = ambiguous_res_names
            # Note we don't update the Res_type in preds, because this is potentially
            # used for correcting the chemical shift
            self.logger.warning(
                    ("There were inconsistencies between the provided sequence "
                    "file and the predicted shifts. The following residues had "
                    "inconsistent amino acid types: %s") %
                    ", ".join(ambiguous_res_names))

#        preds.index = preds["Res_name"]
#        preds.index.name = None
        seq_df.index = seq_df["Res_name"]
        seq_df.index.name = None

        self.seq_df = seq_df.copy()

        #### Add the chemical shift info back in

        # Make columns for the i-1 predicted shifts of C, CA and CB
        preds_m1 = preds[list({"C","CA","CB","Res_type","Res_name"}.
                              intersection(preds.columns))].copy()
        preds_m1.index = preds_m1.index+1
        preds_m1.columns = preds_m1.columns + "_m1"
        preds = pd.merge(preds, preds_m1, how="left",
                         left_index=True, right_index=True)

        # Make column for the i+1 Res_name
        preds_p1 = preds[["Res_name"]].copy()
        preds_p1.index = preds_p1.index-1
        preds_p1.columns = ["Res_name_p1"]
        preds = pd.merge(preds, preds_p1, how="left",
                         left_index=True, right_index=True)

        # Set index to Res_name
        preds.index = preds["Res_name"]
        preds.index.name = None

        # Restrict to only certain atom types
        atom_set = {"H","N","C","CA","CB","C_m1","CA_m1","CB_m1","HA"}
        preds = preds[["Res_name","Res_N","Res_type","Res_name_m1",
                       "Res_name_p1","Res_type_m1"]+
                      list(atom_set.intersection(preds.columns))]

        self.logger.info("Finished reading in %d predicted residues from %s"
                         % (len(preds.index), filename))

        self.preds = preds
        return(self.preds)

    def import_pred_shifts_2(self, filename, filetype, offset=0):
        """ Import predicted chemical shifts from a ShiftX2 results file.
        Note that this function does not transfer predictions for i-1 atoms -
        that is done by map_preds_to_sequence()

        Returns
        A DataFrame containing the predicted shifts, or None if the import failed.

        Parameters
        filename: path to file containing predicted shifts
        filetype: either "shiftx2" or "sparta+"
        offset: an optional integer to add to the residue number.
        """
        
        #### Import the raw data
        if filetype == "shiftx2":
            preds_long = pd.read_csv(filename)
            if any(preds_long.columns == "CHAIN"):
                if len(preds_long["CHAIN"].unique())>1:
                    self.logger.warning(
                            """Chain identifier dropped - if multiple chains are
                            present in the predictions, they will be merged.""")
                preds_long = preds_long.drop("CHAIN", axis=1)
            preds_long = preds_long.reindex(columns=["NUM","RES","ATOMNAME",
                                                     "SHIFT"])
            preds_long.columns = ["Res_N","Res_type","Atom_type","Shift"]
        elif filetype == "sparta+":
            # Work out where the column names and data are
            with open(filename, 'r') as f:
                for num, line in enumerate(f, 1):
                    if line.find("VARS")>-1:
                        colnames_line = num
                        colnames = line.split()[1:]
                        break

            preds_long = pd.read_table(filename, sep="\s+", names=colnames,
                                       skiprows=colnames_line+1)
            preds_long = preds_long.reindex(columns=["RESID","RESNAME",
                                                     "ATOMNAME","SHIFT"])
            preds_long.columns = ["Res_N","Res_type","Atom_type","Shift"]

            # Sparta+ uses HN for backbone amide proton - convert to H
            preds_long.loc[preds_long["Atom_type"]=="HN", "Atom_type"] = "H"
        else:
            self.logger.error("Invalid predicted shift type: '%s'. Allowed "
                              "options are 'shiftx2' or 'sparta+'" % (filetype))
            return(None)

        self.logger.info("Imported %d predicted chemical shifts from %s"
                         % (len(preds_long.index), filename))
        
        #### Initial processing and conversion from long to wide
        # Convert from long to wide format
        preds = preds_long.pivot(index="Res_N", columns="Atom_type",
                                 values="Shift")
        preds.index.name = None

        # Add residue type back into wide table
        tmp = preds_long[["Res_N","Res_type"]]
        tmp = tmp.drop_duplicates(subset="Res_N")
        tmp.index = tmp["Res_N"]
        tmp.index.name = None
        preds = pd.concat([tmp, preds], axis=1)

        # Apply sequence offset and create Res_name column
        preds["Res_N"] = preds["Res_N"] + offset
        preds.insert(1, "Res_name", preds["Res_N"].astype(str) + preds["Res_type"])
        preds["Res_name"] = preds["Res_name"].str.rjust(5)

        self.logger.info("Finished reading in %d predicted residues from %s"
                         % (len(preds.index), filename))

        self.preds = preds
        return(self.preds)

    def simulate_pred_shifts(self, filename, sd, seed=None):
        """Generate a 'simulated' predicted shift DataFrame by importing some
        observed chemical shifts (in 'test' format), and adding Gaussian
        errors.

        sd: dictionary of standard deviation for each atom type
            eg. {"H":0.1,"N":0.5}
        """
        from SNAPS_importer import SNAPS_importer
        import numpy.random as rand

        if seed is not None:
            rand.seed(seed)


        importer = SNAPS_importer()
        importer.import_testset_shifts(filename, remove_Pro=False)

        preds = importer.obs

        # Limit columns
        preds = preds[["Res_N","Res_type"]+
                      list({"C","CA","CB","H","N","HA"}.intersection(preds.columns))]

        # Add the random shifts
        for atom in self.pars["atom_set"].intersection(preds.columns):
            preds.loc[:,atom] += rand.normal(0, sd[atom], size=len(preds.index))

        # Add other columns back in
        preds.insert(1, "Res_name", (preds["Res_N"].astype(str) +
                  preds["Res_type"]))
        preds.loc[:,"Res_name"] = preds["Res_name"].str.rjust(5)

        preds.index = preds["Res_N"]
        #preds.index.name=None

        # Make columns for the i-1 predicted shifts of C, CA and CB
        preds_m1 = preds[list({"C","CA","CB","Res_type","Res_name"}.
                              intersection(preds.columns))].copy()
        preds_m1.index = preds_m1.index+1
        preds_m1.columns = preds_m1.columns + "_m1"
        preds = pd.merge(preds, preds_m1, how="left",
                         left_index=True, right_index=True)

        # Make column for the i+1 Res_name
        preds_p1 = preds[["Res_name"]].copy()
        preds_p1.index = preds_p1.index-1
        preds_p1.columns = ["Res_name_p1"]
        preds = pd.merge(preds, preds_p1, how="left",
                         left_index=True, right_index=True)

        # Set index to Res_name
        preds.index = preds["Res_name"]
        preds.index.name = None

        # Restrict to only certain atom types
        atom_set = {"H","N","C","CA","CB","C_m1","CA_m1","CB_m1","HA"}
        preds = preds[["Res_name","Res_N","Res_type","Res_name_m1",
                       "Res_name_p1","Res_type_m1"]+
                      list(atom_set.intersection(preds.columns))]

        self.logger.info("Finished simulating %d predicted residues from %s"
                         % (len(preds.index), filename))

        self.preds = preds
        return(self.preds)

    def map_preds_to_sequence(self, seq_df, preds):
        """Align the predictions to the sequence, and deal with any 
        inconsistencies. Also transfer predictions for i-1 atoms.
        
        Returns a dataframe with the residues defined in seq_df, with 
        all aligned predictions from preds"""

        # Do the alignment
        sequence = seq_df.Res_type.sum()
        pred_sequence = preds.Res_type.dropna().sum()
        aligner = Align.PairwiseAligner()
        aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
        alignment = aligner.align(sequence, pred_sequence)[0]
        self.alignment = alignment

        # Create a dataframe mapping preds onto sequence
        # alignment.aligned gives nested lists describing the aligned regions
        # For example, it might be [[[0,5], [10,15]], [[0,5], [15,20]]]
        # This means residues 0-4 of sequence 1 align with 0-4 of sequence 2, and 
        # residues 10-14 of sequence 1 align to 15-19 of sequence 2
        seq_df_aligned_regions = alignment.aligned[0]
        preds_aligned_regions = alignment.aligned[1]

        seq_df_aligned_residues = []
        for region in seq_df_aligned_regions: 
            seq_df_aligned_residues += list(range(region[0], region[1]))
        preds_aligned_residues = []
        for region in preds_aligned_regions: 
            preds_aligned_residues += list(range(region[0], region[1]))

        preds = preds.iloc[preds_aligned_residues, :]       # Discard predictions that are not in the alignment
        preds.index = seq_df.index[seq_df_aligned_residues] # Index the predictions with their position in sequence
        preds = preds.rename(columns={"Res_N":"Pred_N", "Res_name":"Pred_name", "Res_type":"Pred_type"})
        aligned_preds = pd.concat((seq_df, preds), axis=1)
        
        # If glycine HA is missing, replace it with average of HA2 and HA3
        mask = aligned_preds.HA.isna()
        aligned_preds.loc[mask, "HA"] = 0.5 * (aligned_preds.loc[mask, "HA2"] + aligned_preds.loc[mask, "HA3"])

        # Remove atoms that are not used?
        cols_to_keep = ["Seq_N", "Res_N", "Res_type", "Res_name", "Res_name_m1", 
                        "Res_name_p1", "Pred_N", "Pred_type", "Pred_name"] 
        atom_list = list(self.pars["atom_set"].intersection(aligned_preds.columns))
        aligned_preds = aligned_preds.loc[:, cols_to_keep + atom_list]

        # Remove predictions where residue type is inconsistent
        inconsistent_res_type = (aligned_preds.Res_type != aligned_preds.Pred_type)
        aligned_preds.loc[inconsistent_res_type, atom_list] = np.nan
       
        # Add columns for i-1 and i+1 atoms
        aligned_preds.index = aligned_preds.Res_name
        aligned_preds.index.name = None
        preds_m1 = aligned_preds[list({"C","CA","CB", "N","Res_type","Res_name", "Res_name_p1"}.
                              intersection(aligned_preds.columns))].copy()
        preds_m1.index = preds_m1.Res_name_p1   # Note _p1, because we need to transfer the chemical shift to the i+1 residue!
        aligned_preds = pd.merge(aligned_preds, preds_m1[["C","CA","CB", "N"]], how="left",
                         left_index=True, right_index=True, suffixes=["", "_m1"])
        
        preds_p1 = aligned_preds[list({"N","Res_type","Res_name", "Res_name_m1"}.
                              intersection(aligned_preds.columns))].copy()
        preds_p1.index = preds_p1.Res_name_m1   # Note _m1, because we need to transfer the chemical shift to the i-1 residue!
        aligned_preds = pd.merge(aligned_preds, preds_p1[["N"]], how="left",
                         left_index=True, right_index=True, suffixes=["", "_p1"])

        # Output info to log
        self.logger.info("Aligned %d predicted residues to sequence" % ((~aligned_preds.Pred_name.isna()).sum()))
        self.logger.info("\n"+str(alignment))
        
        self.aligned_preds = aligned_preds
        return(aligned_preds)
    
    def create_generic_predictions(self):
        """Create a set of predicted shifts based only on residue type. 
        These can be used where there are missing predictions from the structure.

        Returns a dataframe with the generic shift predictions for each residue.
        """
        generic_preds = self.aligned_preds.copy()
        generic_preds = generic_preds.loc[:,self.seq_df.columns]

        # Import the average shifts for each residue type
        residue_shifts = pd.read_csv(self.pars["generic_shift_file"], index_col=0)
        
        # Create the generic shift predictions
        atom_list = ["H", "N", "C", "CA", "CB", "HA"]
        for atom in atom_list: generic_preds[atom] = np.nan

        for res in residue_shifts.index:
            for atom in atom_list:
                generic_preds.loc[generic_preds.Res_type==res, atom] = residue_shifts.loc[res, atom]

        # Create the i-1 and i+1 columns
        generic_preds.index = generic_preds.Res_name
        generic_preds.index.name = None
        preds_m1 = generic_preds[list({"C","CA","CB", "N","Res_type","Res_name", "Res_name_p1"}.
                              intersection(generic_preds.columns))].copy()
        preds_m1.index = preds_m1.Res_name_p1   # Note _p1, because we need to transfer the chemical shift to the i+1 residue!
        generic_preds = pd.merge(generic_preds, preds_m1[["C","CA","CB", "N"]], how="left",
                         left_index=True, right_index=True, suffixes=["", "_m1"])
        
        preds_p1 = generic_preds[list({"N","Res_type","Res_name", "Res_name_m1"}.
                              intersection(generic_preds.columns))].copy()
        preds_p1.index = preds_p1.Res_name_m1   # Note _m1, because we need to transfer the chemical shift to the i-1 residue!
        generic_preds = pd.merge(generic_preds, preds_p1[["N"]], how="left",
                         left_index=True, right_index=True, suffixes=["", "_p1"])
        
        # Restrict to atom types in preds 
        col_mask = generic_preds.columns.isin(self.aligned_preds.columns) 
        generic_preds = generic_preds.loc[:,col_mask]

        self.generic_preds = generic_preds
        return(generic_preds)

    def prepare_obs_preds(self):
        """Perform preprocessing on the observed and predicted shifts to prepare
        them for calculation of the log probability matrix.

        1) Discard atom types that are not used or aren't present in both
        2) Create dummy spin systems for prolines
        3) Add dummy rows to obs or preds to bring them to the same length

        Returns the modified obs and preds DataFrames.
        """

        obs = self.obs.copy()
        preds = self.aligned_preds.copy()

        #### Restrict atom types
        # self.pars["atom_set"] is the set of atoms to be used in the analysis
        # Get all non-atom columns
        all_atoms = {"H","N","C","CA","CB","C_m1","CA_m1","CB_m1","HA", "N_m1", "N_p1"}
        obs_metadata = list(set(obs.columns).difference(all_atoms))
        preds_metadata = list(set(preds.columns).
                              difference(all_atoms))
        # Get atoms that are in atom_set *and* present in both obs and preds
        shared_atoms = list(self.pars["atom_set"].intersection(obs.columns).
                            intersection(preds.columns))
        obs = obs.loc[:,obs_metadata+shared_atoms]
        preds = preds.loc[:,preds_metadata+shared_atoms]

        self.logger.info("Restricted atom types to %s" % ",".join(shared_atoms))

        #### Create dummy rows
        # Create columns to keep track of dummy status
        preds["Dummy_res"] = False
        obs["Dummy_SS"] = False

        # Create dummy observations for each proline residue
        # When probabilities are assigned, each proline observation will only match to its own proline
        prolines = preds[preds.Res_type=="P"].Res_name
        proline_obs = pd.DataFrame(index=prolines, columns=obs.columns)
        proline_obs["SS_name"] = prolines
        proline_obs["Dummy_SS"] = True
        obs = pd.concat([obs, proline_obs])

        # Add extra dummy residues or spin systems, so that there are equal numbers
        N = len(obs.index)
        M = len(preds.index)

        if N>M:     # If there are more spin systems than predictions
            dummies = pd.DataFrame(np.nan, columns = preds.columns,
                        index=["DR_"+str(i) for i in 1+np.arange(N-M)])
            dummies["Res_name"] = dummies.index
            dummies["Dummy_res"] = True
            #preds = preds.append(dummies)
            preds = pd.concat([preds, dummies])
            self.logger.info("Added %d dummy predicted residues" % len(dummies.index))
        elif M>N:
            dummies = pd.DataFrame(np.nan, columns = obs.columns,
                        index=["DSS_"+str(i) for i in 1+np.arange(M-N)])
            dummies["SS_name"] = dummies.index
            dummies["Dummy_SS"] = True
            # obs = obs.append(dummies)
            obs = pd.concat([obs, dummies])
            self.logger.info("Added %d dummy observed residues" % len(dummies.index))

        obs = obs.sort_values("SS_name")
        obs.index = obs.SS_name
        obs.index.name = None

        self.obs = obs.copy()
        self.aligned_preds = preds.copy()

        return(self.obs, self.preds)

    def calc_prob_matrix(self, preds=None, atom_sd=None, atom_res_sd=None, 
                         glycine_CB_penalty=0.01, delta_correlation=None, 
                         normalise_by=None):
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

        obs = self.obs.copy()
        if preds is None:
            preds = self.aligned_preds.copy()  
        atoms = list(self.pars["atom_set"].intersection(obs.columns))
        prob_matrix = pd.DataFrame(1.0, index=obs.index, columns=preds.index)
        prob_matrix.index.name = "SS_name"
        prob_matrix.columns.name = "Res_name"
        if delta_correlation is None:
            delta_correlation = self.pars["delta_correlation"]

        # Use default atom_sd values if not defined
        if atom_sd==None:
            atom_sd = self.pars["atom_sd"]

        if delta_correlation:
            # Import parameters describing the delta correlations
            # Note: this also sorts the atom types in d_mean and c_cov into the
            # same order as the 'atoms' list defined above.
            d_mean = pd.read_csv(self.pars["delta_correlation_mean_file"],
                                    header=None, index_col=0).loc[atoms,1]
            d_cov = (pd.read_csv(self.pars["delta_correlation_cov_file"],
                                    index_col=0).loc[atoms,atoms])
            self.logger.info("Imported delta_correlation info from %s and %s"
                            % (self.pars["delta_correlation_mean_file"],
                            self.pars["delta_correlation_cov_file"]))
            
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
                                reshape([len(obs.index),-1]),
                                index=obs.index, columns=preds.index)
            preds_atom = pd.DataFrame(preds[atom].repeat(len(preds.index)).values.
                                reshape([len(preds.index),-1]).transpose(),
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
                    prob_atom = pd.DataFrame(norm.pdf(delta_atom, scale=atom_sd[atom]), 
                                        index=obs.index, columns=preds.index)
                else:
                    # If atom is from a neighbouring residue, choose the appropriate atom type.
                    atom2 =  atom
                    if atom2=="C_m1": atom2 = "C"
                    if atom2=="CA_m1": atom2 = "CA"
                    if atom2=="CB_m1": atom2 = "CB"
                    if atom2 in ["N_m1","N_p1"]: atom2 = "N"

                    prob_atom = pd.DataFrame(1.0, index=obs.index, columns=preds.index)
                    for res in atom_res_sd.index:
                        mask = preds.Res_type==res
                        tmp = norm.pdf(delta_atom.loc[:,mask], scale=atom_res_sd.loc[res, atom2])
                        prob_atom.loc[:,mask] = pd.DataFrame(tmp, 
                                        index=obs.index, columns=preds.index[mask])

                prob_matrix = prob_matrix * prob_atom

        if delta_correlation:
            self.logger.info("Accounting for correlated prediction errors")

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
            glycines = preds[preds.Res_type=="G"].Res_name
            obs_CB = obs[~obs.CB.isna()].SS_name
            for g in glycines:
                prob_matrix.loc[obs_CB, g] = prob_matrix.loc[obs_CB, g] * glycine_CB_penalty
            
            glycines_m1 = preds[preds.Res_type=="G"].Res_name
            obs_CB_m1 = obs[~obs.CB_m1.isna()].SS_name
            for g in glycines_m1:
                prob_matrix.loc[obs_CB_m1, g] = prob_matrix.loc[obs_CB_m1, g] * glycine_CB_penalty

        # Pair up the proline dummy spin systems with the appropriate residue
        prolines = preds[preds.Res_type=="P"].Res_name
        prob_matrix.loc[:, prolines] = 1e-10
        prob_matrix.loc[prolines, :] = 1e-10
        for x in prolines:
            prob_matrix.loc[x, x] = 1.0

        # Replace any zero probabilities which a very small number
        # (Zero probabilities can happen due to rounding error, I think)
        prob_matrix[prob_matrix==0] = 1e-100

        # Do the normalisation, if needed
        if normalise_by=="Res":
            prob_matrix = prob_matrix.div(prob_matrix.sum(axis=0), axis="columns")
        elif normalise_by=="SS":
            prob_matrix = prob_matrix.div(prob_matrix.sum(axis=1), axis="rows")

        self.logger.info("Calculated probability matrix (%dx%d)",
                         prob_matrix.shape[0], prob_matrix.shape[1])
        
        # self.prob_matrix = prob_matrix
        return(prob_matrix)

    def calc_generic_prob_matrix(self, only_missing_preds=True, normalise_by=None):
        """Calculate a probability matrix using generic shift predictions.
        This can be used in place of the probability matrix from predictions, 
        or used to compensate for missing predictions.
        
        Parameters:
        only_missing_preds: if True, only generic predictions from atoms with 
        missing predicted shifts are used.

        Returns
        The probability matrix"""
        generic_preds = self.generic_preds.copy()
        preds = self.aligned_preds

        # Remove generic predictions for atoms with a real prediction
        if only_missing_preds:
            cols = generic_preds.columns[generic_preds.columns.isin(self.pars["atom_set"])]
            generic_preds[~preds.loc[:,cols].isna()] = np.nan
            # breakpoint()

        # Import average generic shifts and standard deviations
        generic_stdev = pd.read_csv(self.pars["generic_shift_stdev_file"], index_col=0)

        # Calculate the probability matrix
        generic_prob_matrix = self.calc_prob_matrix(preds=generic_preds, atom_res_sd=generic_stdev, glycine_CB_penalty=1.0, delta_correlation=False)

        # Do the normalisation, if needed
        if normalise_by=="Res":
            generic_prob_matrix = (generic_prob_matrix / generic_prob_matrix.sum(axis=0))
        elif normalise_by=="SS":
            generic_prob_matrix = (generic_prob_matrix / generic_prob_matrix.sum(axis=1))

        return(generic_prob_matrix)

    def calc_log_prob_matrix(self, prob_matrix=None):
        """Calculate a log probability matrix from a probability matrix
        
        By default uses the prob_matrix stored in the SNAPS_assigner object, 
        but can also be passed a different prob_matrix.

        Returns
        A DataFrame containing the log probabilities
        """
        if prob_matrix is None:
            prob_matrix = self.prob_matrix

        # Remove any zeroes, by setting them to the minimum value in the matrix
        replacement_value = min(prob_matrix[prob_matrix>0].min().min(), 1e-200)
        prob_matrix[prob_matrix==0] = replacement_value

        log_prob_matrix = np.log10(prob_matrix)

        # self.log_prob_matrix = log_prob_matrix
        return(log_prob_matrix)

    def calc_log_prob_matrix_old(self, atom_sd=None, sf=1, default_prob=0.01):
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

        obs = self.obs.copy()
        preds = self.aligned_preds.copy()
        atoms = list(self.pars["atom_set"].intersection(obs.columns))

        if self.pars["pred_correction"]:
            # Import parameters for correcting the shifts
            lm_pars = pd.read_csv(self.pars["pred_correction_file"], index_col=0)
            self.logger.info("Imported pred_correction info from %s"
                             % self.pars["pred_correction_file"])

        if self.pars["delta_correlation"]:
            # Import parameters describing the delta correlations
            # Note: this also sorts the atom types in d_mean and c_cov into the
            # same order as the 'atoms' list defined above.
            if self.pars["pred_correction"]:
                d_mean = pd.read_csv(self.pars["delta_correlation_mean_corrected_file"],
                                     header=None, index_col=0).loc[atoms,1]
                d_cov = (pd.read_csv(self.pars["delta_correlation_cov_corrected_file"],
                                     index_col=0).loc[atoms,atoms])
                self.logger.info("Imported delta_correlation info from %s and %s"
                             % (self.pars["delta_correlation_mean_corrected_file"],
                                self.pars["delta_correlation_cov_corrected_file"]))
            else:
                d_mean = pd.read_csv(self.pars["delta_correlation_mean_file"],
                                     header=None, index_col=0).loc[atoms,1]
                d_cov = (pd.read_csv(self.pars["delta_correlation_cov_file"],
                                     index_col=0).loc[atoms,atoms])
                self.logger.info("Imported delta_correlation info from %s and %s"
                             % (self.pars["delta_correlation_mean_file"],
                                self.pars["delta_correlation_cov_file"]))

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
            obs_atom = pd.DataFrame(obs[atom].repeat(len(obs.index)).values.
                                reshape([len(obs.index),-1]),
                                index=obs.index, columns=preds.index)
            preds_atom = pd.DataFrame(preds[atom].repeat(len(preds.index)).values.
                                reshape([len(preds.index),-1]).transpose(),
                                index=obs.index, columns=preds.index)


            # If correcting the predictions, subtract a linear function of the
            # observed shift from each predicted shift
            if self.pars["pred_correction"]:
                self.logger.info("Calculating corrections to predicted shifts")
                preds_corr_atom = preds_atom
                for res in preds["Res_type"].dropna().unique():
                    if (atom+"_"+res) in lm_pars.index:
                        # Find shifts of the current atom/residue combination
                        mask = ((lm_pars["Atom_type"]==atom) &
                                (lm_pars["Res_type"]==res))

                        # Look up model parameters
                        grad = lm_pars.loc[mask,"Grad"].tolist()[0]
                        offset = lm_pars.loc[mask,"Offset"].tolist()[0]

                        # Make the correction
                        if atom in ("C_m1","CA_m1","CB_m1"):
                            mask = (preds["Res_type_m1"]==res)
                            preds_corr_atom.loc[:, mask] = (
                                    preds_atom.loc[:, mask]
                                    - grad * obs_atom.loc[:, mask]
                                    - offset)
                        else:
                            mask = (preds["Res_type"]==res)
                            preds_corr_atom.loc[:, mask] = (
                                    preds_atom.loc[:, mask]
                                    - grad * obs_atom.loc[:, mask]
                                    - offset)

                delta_atom = preds_corr_atom - obs_atom
            else:
                delta_atom = preds_atom - obs_atom

            if self.pars["delta_correlation"]:
                # Store delta matrix for this atom type for later analysis
                delta_atom = delta_atom.fillna(0.0)
                delta_list = delta_list + [delta_atom.values]
            else:
                # Make a note of NA positions in delta, and set them to zero
                # (this avoids warnings when using norm.logpdf)
                na_mask = delta_atom.isna()
                delta_atom[na_mask] = 0.0
                delta_atom = delta_atom.astype(float)

                # Calculate the log probability density
                prob_atom = pd.DataFrame(norm.logpdf(delta_atom,
                                                     scale=atom_sd[atom]),
                                         index=obs.index, columns=preds.index)
                # norm.logpdf(delta_atom, scale=atom_sd[atom])

                # Replace former NA values with a default value
                prob_atom[na_mask] = log10(default_prob)

                # Add to the log prob matrix
                log_prob_matrix = log_prob_matrix + prob_atom

        if self.pars["delta_correlation"]:
            self.logger.info("Accounting for correlated prediction errors")

            # Combine the delta matrixes from each atom type into a single 3D matrix
            delta_mat = np.array(delta_list)
            # Move the axis containing the atom category to the end
            # (necessary for multivariate_normal function)
            delta_mat = np.moveaxis(delta_mat, 0, -1)

            # Make a note of NA positions in delta, and set them to zero
            # na_mask = delta_mat.isna()
            # delta_mat[na_mask] = 0.0
            

            # Initialise multivariate model and calculate log_pdf
            mvn = multivariate_normal(d_mean, d_cov)
            log_prob_matrix = pd.DataFrame(mvn.logpdf(delta_mat),
                                           index=obs.index, columns=preds.index)


            # Apply a penalty for missing data
            # na_matrix = na_mask.sum(axis=-1)    # Count how many NA values for
            #                                     # each Res/SS pair
            # log_prob_matrix = log_prob_matrix + log10(default_prob) * na_matrix

        if self.pars["use_SS_class_info"]:
            # For each type of residue type information that's available, make a
            # matrix showing the probability modifications due to type mismatch,
            # then add it to log_prob_matrix
            # Maybe make SS_class mismatch a parameter in config file?
            for ss_class in {"SS_class","SS_class_m1"}.intersection(obs.columns):
                #print(ss_class)
                SS_class_matrix = pd.DataFrame(0.0, index=log_prob_matrix.index,
                                                columns=log_prob_matrix.columns)

                # For each amino acid type in turn:
                for res in preds["Res_type"].dropna().unique():
                    # Work out which observations could be that aa type
                    allowed = obs[ss_class].str.contains(res).fillna(True)
                    # Select the predictions which are that aa type
                    pred_list = preds.loc[preds["Res_type_m1"]==res,"Res_name"].to_list()
                    # For the selected predictions, penalise any observations
                    # where the current aa type is not allowed
                    for p in pred_list:
                        SS_class_matrix.loc[:,p] = (1 - allowed.astype(float))*-2 #log10(0.01)

                log_prob_matrix = log_prob_matrix + SS_class_matrix

        # Sort out NAs and dummy residues/spin systems
        log_prob_matrix[log_prob_matrix.isna()] = 2*np.nanmin(
                                                        log_prob_matrix.values)
        log_prob_matrix.loc[obs["Dummy_SS"], :] = 0
        log_prob_matrix.loc[:, preds["Dummy_res"]] = 0

        log_prob_matrix.index.name = "SS_name"
        log_prob_matrix.columns.name = "Res_name"

        # Pair up the proline dummy spin systems with the appropriate residue
        prolines = preds[preds.Res_type=="P"].Res_name
        log_prob_matrix.loc[:, prolines] = -10
        log_prob_matrix.loc[prolines, :] = -10
        for x in prolines:
            log_prob_matrix.loc[x, x] = 0


        self.logger.info("Calculated log probability matrix (%dx%d)",
                         log_prob_matrix.shape[0], log_prob_matrix.shape[1])

        self.log_prob_matrix = log_prob_matrix
        return(self.log_prob_matrix)

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
                mismatch_atom = mismatch_atom.fillna(0.0)
                consistent_links_atom = consistent_links_atom.fillna(0.0)

                # Update mismatch and consistent links matrixes
                mismatch_matrix = mismatch_matrix.combine(abs(mismatch_atom), np.maximum)
                consistent_links_matrix = consistent_links_matrix + consistent_links_atom

            allowed_links_matrix = (mismatch_matrix < 0.2).astype(int)
            for i in allowed_links_matrix.index: allowed_links_matrix.loc[i, i] = 0     # Set diagonal to zero

            self.logger.info("Calculated mismatch and consistent_links matrixes (%dx%d)",
                         mismatch_matrix.shape[0], mismatch_matrix.shape[1])

            self.mismatch_matrix = mismatch_matrix
            self.consistent_links_matrix = consistent_links_matrix
            self.allowed_links_matrix = allowed_links_matrix
            return(self.mismatch_matrix, consistent_links_matrix)

    def calc_triplet_prob_matrix(self, prob_matrix=None, allowed_links_matrix=None, normalise_by=None):
        """ Calculate the probability that each spin system is assigned to a given residue, 
        summed over all possible triplets, accounting for allowed sequential links.

        """
        if prob_matrix is None:
            prob_matrix = self.prob_matrix

        if allowed_links_matrix is None:
            allowed_links_matrix = self.allowed_links_matrix

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

        self.triplet_prob_matrix = triplet_prob_matrix
        return(triplet_prob_matrix)

    def find_best_assignment(self, score_matrix, maximise=True, inc=None, exc=None, exc_mask=None,
                             dummy_rows=[], dummy_cols=[], return_none_all_dummy=False):
        """ Use the Hungarian algorithm to find the highest scoring assignment,
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

        self.logger.debug("Started linear assignment")

        if maximise:
            penalty = -2*score_matrix.abs().max().max()
        else:
            penalty = 2*score_matrix.abs().max().max()
        if exc_mask is not None:    
            score_matrix = score_matrix.mask(exc_mask, other=penalty)

        if inc is not None:
            # Check for conflicting entries in inc
            conflicts = inc[row_name].duplicated(keep=False) | inc[col_name].duplicated(keep=False)
            if any(conflicts):
                self.logger.warning("Warning: entries in inc conflict with one another - dropping conflicts")
                inc = inc[~conflicts]

            if exc is not None:
                # Check constraints are consistent
                # Get rid of any entries in exc which share a Res or SS with inc
                exc_in_inc = exc[row_name].isin(inc[row_name]) | exc[col_name].isin(inc[col_name])
                if any(exc_in_inc):
                    self.logger.warning("Some values in exc are also found in inc, so are redundant.")
                    exc = exc.loc[~exc_in_inc, :]

            # Removed fixed assignments from probability matrix and obs, preds
            # dataframes. Latter is needed if inc includes any dummy SS/res,
            # and to detect if the reduced data is entirely dummies
            score_matrix_reduced = score_matrix.drop(index=inc[row_name]).drop(columns=inc[col_name])
            self.logger.debug("%d assignments were fixed, %d remain to be assigned"
                             % (len(inc), len(score_matrix_reduced.index)))
        else:
            score_matrix_reduced = score_matrix

        # If the matrix consists entirely of dummy rows or columns, the
        # assignment will not be meaningful. In this case, may want to return None
        if return_none_all_dummy:
            if dummy_rows is not None:
                if set(score_matrix_reduced.index).issubset(dummy_rows):
                    self.logger.debug("Score matrix includes only dummy rows")
                    return(None)
            if dummy_cols is not None:
                if set(score_matrix_reduced.columns).issubset(dummy_cols):
                    self.logger.debug("Score matrix includes only dummy columns")
                    return(None)   
            
        if exc is not None:
            # Penalise excluded (row, col) pairs
            if maximise:
                penalty = -2*score_matrix.abs().max().max()
            else:
                penalty = 2*score_matrix.abs().max().max()
            
            for i, r in exc.iterrows():     # iterates over (index, row as pd.Series) tuples
                # If one side of an exclude pair is a dummy row or column,
                # exclude *all* dummy rows and columns
                if r[col_name] in dummy_cols:
                    score_matrix_reduced.loc[r[row_name],
                                             dummy_cols] = penalty
                elif r[row_name] in dummy_rows:
                    score_matrix_reduced.loc[dummy_rows,
                                             r[col_name]] = penalty
                else:
                    score_matrix_reduced.loc[r[row_name], r[col_name]] = penalty

            self.logger.debug("Penalised %d excluded row,column pairs" % len(exc.index))

        if maximise:
            row_ind, col_ind = linear_sum_assignment(-1*score_matrix_reduced)
            # -1 because linear_sum_assignment() minimises sum, but we want to
            # maximise it.
        else:
            row_ind, col_ind = linear_sum_assignment(score_matrix_reduced)

        # Construct results dataframe
        matching_reduced = pd.DataFrame({row_name:score_matrix_reduced.index[row_ind],
                                         col_name:score_matrix_reduced.columns[col_ind]})

        if inc is not None:
            matching = pd.concat([inc, matching_reduced]).sort_values("Res_name")
            return(matching)
        else:
            matching_reduced = matching_reduced.sort_values("Res_name")
            return(matching_reduced)

    def make_assign_df(self, matching, set_assign_df=False):
        """Make a dataframe with full assignment information, given a dataframe
        of SS_name and Res_name.

        Matching may have additional columns, which will also be kept.
        """
        obs = self.obs.copy()
        obs.index.name = "name" # Needed to avoid error when merging dataframes.
        preds = self.aligned_preds.copy()
        log_prob_matrix = self.log_prob_matrix.copy()
        valid_atoms = list(self.pars["atom_set"])
        extra_cols = set(matching.columns).difference({"SS_name","Res_name"})

        if not {"SS_name","Res_name"}.issubset(matching.columns):
            self.logger.warning("Cannot make assignment dataframe, as matching"
                                +" dataframe does not have the correct columns")

        assign_df = pd.merge(matching,
                             preds.loc[:,["Res_N","Res_type", "Res_name",
                                    "Pred_N", "Pred_name", "Dummy_res"]],
                             on="Res_name", how="left")
        assign_df = assign_df[["Res_name","Res_N","Res_type","SS_name",
                               "Pred_N", "Pred_name", "Dummy_res"]+list(extra_cols)]
        assign_df = pd.merge(assign_df,
                             obs.loc[:, obs.columns.isin(
                                     ["SS_name","Dummy_SS"]+valid_atoms)],
                             on="SS_name", how="left")
        assign_df = pd.merge(assign_df,
                             preds.loc[:, preds.columns.isin(
                                     valid_atoms+["Res_name"])],
                             on="Res_name", suffixes=("","_pred"), how="left")
        
        assign_df.index = assign_df.SS_name     # Needed to match Log_prob to correct row.
        assign_df.index.names = ["index"]
        assign_df["Log_prob"] = df_lookup(log_prob_matrix,
                                          assign_df["SS_name"],
                                          assign_df["Res_name"])
        # Careful above not to get rows/columns confused

        assign_df = assign_df.sort_values(by="Res_N")

        if set_assign_df:
            self.assign_df = assign_df

        return(assign_df)

    def assign_from_preds(self, log_prob_matrix=None, set_assign_df=False, inc=None, exc=None):
        """Assign the observed spin systems using predicted shifts only

        This function essentially wraps around find_best_assignment() and
        make_assign_df()
        inc: a DataFrame of (row, col) pairs which must be part of the assignment.
            First column has the index names, second has the column names.
        exc: a DataFrame of (row, col) pairs which may not be part of the assignment.
        """
        if log_prob_matrix is None:
            log_prob_matrix = self.log_prob_matrix

        matching = self.find_best_assignment(log_prob_matrix, maximise=True, inc=inc, exc=exc)
        assign_df = self.make_assign_df(matching, set_assign_df)

        self.logger.info("Finished calculating best assignment based on predictions")
        return(assign_df)

    def calc_overall_matching_prob(self, matching):
        "Calculate the sum log probability of a particular matching"
        # return(sum(self.log_prob_matrix.lookup(matching["SS_name"],
        #                                            matching["Res_name"])))
        return(sum(df_lookup(self.log_prob_matrix, matching["SS_name"],
                                                    matching["Res_name"])))

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

        tmp = pd.concat([matching, self.aligned_preds[["Res_name_m1","Res_name_p1"]]], axis=1)

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
        tmp["Num_good_links_m1"] = tmp["Num_good_links_m1"].fillna(0)
        tmp["Num_good_links_p1"] = tmp["Num_good_links_p1"].fillna(0)
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
            assign_df["Max_mismatch_m1"] = np.nan
            assign_df["Max_mismatch_p1"] = np.nan
            assign_df["Num_good_links_m1"] = 0
            assign_df["Num_good_links_p1"] = 0
            assign_df["Confidence"] = "Undefined"

            self.logger.warning("Couldn't calculate assignment confidence: "+
                                "need sequential atom information")

        if set_assign_df:
            self.assign_df = assign_df
        return(assign_df)

    def find_alt_assignments(self, N=1, by_ss=True, verbose=False):
        """ Find the next-best assignment(s) for each residue or spin system

        This works by setting the log probability to a very high value for each
        residue in turn, and rerunning the assignment. Assumes that the best assignment
        has already been found, and is stored in self.assign_df.

        Arguments:
        N: number of alternative assignments to generate
        by_ss: if true, calculate next best assignment for each spin system.
            Otherwise, calculate it for each residue.

        Output:
        A Dataframe containing the original assignments, and the
        alt_assignments by rank
        """

        log_prob_matrix = self.log_prob_matrix.copy()
        preds = self.aligned_preds.copy()
        best_matching = self.assign_df.loc[:,["SS_name","Res_name"]]
        best_matching.index = best_matching["SS_name"]
        best_matching.index.name = None
        alt_matching = None
        dummy_rows = self.obs.loc[self.obs.Dummy_SS, "SS_name"]
        dummy_cols = preds.loc[preds.Dummy_res, "Res_name"]

        # Calculate sum probability for the best matching
        best_sum_prob = self.calc_overall_matching_prob(best_matching)

        # Calculate the value used to penalise the best match for each residue
        penalty = 2*log_prob_matrix.min().min()
        self.logger.debug("Penalty value: %f", penalty)

        # Initialise DataFrame for storing alt_assignments
        alt_matching_all = best_matching.copy()
        alt_matching_all["Rank"] = 1
        alt_matching_all["Rel_prob"] = 0

        for i in best_matching.index:   # Consider each spin system in turn
            ss = best_matching.loc[i, "SS_name"]
            res = best_matching.loc[i, "Res_name"]
            self.logger.debug("Finding alt assignments for original match %s - %s", ss, res)
            if verbose: print(ss, res)

            excluded = best_matching.loc[[i], :]

            for j in range(N):
                alt_matching = self.find_best_assignment(self.log_prob_matrix, exc=excluded, 
                                                         dummy_rows=dummy_rows, dummy_cols=dummy_cols)

                alt_matching["Rank"] = j+2
                alt_sum_prob = self.calc_overall_matching_prob(alt_matching)
                alt_matching["Rel_prob"] = alt_sum_prob - best_sum_prob

                # Add the alt match for this ss or res to the results dataframe
                # and also the excluded dataframe.
                if by_ss:
                    alt_matching_all = pd.concat([alt_matching_all, 
                                                  alt_matching.loc[alt_matching["SS_name"]==ss, :]], 
                                                  ignore_index=True)
                    res = alt_matching.loc[alt_matching["SS_name"]==ss,
                                           "Res_name"].tolist()[0]
                    # The .tolist()[0] is to convert a single-item series into a string.
                else:
                    alt_matching_all = pd.concat([alt_matching_all, 
                                                  alt_matching.loc[alt_matching["Res_name"]==res, :]], 
                                                  ignore_index=True)

                    ss = alt_matching.loc[alt_matching["Res_name"]==res,
                                          "SS_name"].tolist()[0]
                # excluded = excluded.append(pd.DataFrame({"SS_name":[ss],"Res_name":[res]}),
                #                            ignore_index=True)
                excluded = pd.concat([excluded, pd.DataFrame({"SS_name":[ss],"Res_name":[res]})], 
                                     ignore_index=True)

        self.alt_assign_df = self.make_assign_df(alt_matching_all)
        
        if by_ss:
            self.alt_assign_df = self.alt_assign_df.sort_values(
                                                by=["SS_name", "Rank"])
        else:
            self.alt_assign_df = self.alt_assign_df.sort_values(
                                                by=["Res_name", "Rank"])

        return(self.alt_assign_df)

    def find_kbest_assignments(self, k, init_inc=None, init_exc=None, verbose=False):
        """ Find the k best overall assignments using the Murty algorithm.

        k: the number of assignments to find
        verbose: print information about progress

        This algorithm works by defining nodes. A node is a particular set of
        constraints on the assignment, consisting of pairs that muct be included,
        and pairs that must be excluded. These constraints define a set of
        potential assignments, one of which will be optimal. The clever part is
        that this set of assignments can be further subdivided into the optimal
        assignment plus a set of child nodes, each with additional constraints.
        The optimal solution can be found for each child node, allowing them to
        be ranked. The highest ranking child can then be split up further,
        allowing computational effort to be focused on the sets containing the
        best solutions.
        For more details, see:
        Murty, K. (1968). An Algorithm for Ranking all the Assignments in
        Order of Increasing Cost. Operations Research, 16(3), 682-687
        """
        Node = namedtuple("Node", ["sum_log_prob","matching","inc","exc"])

        # Initial best matching (subject to initial constraints)
        best_matching = self.find_best_assignment(inc=init_inc, exc=init_exc)
        best_matching.index = best_matching["SS_name"]
        best_matching.index.name = None

        # Define lists to keep track of nodes
        ranked_nodes = SortedListWithKey(key=lambda n: n.sum_log_prob)
        unranked_nodes = SortedListWithKey(
                            [Node(self.calc_overall_matching_prob(best_matching),
                            best_matching, inc=init_inc, exc=init_exc)],
                            key=lambda n: n.sum_log_prob)

        while len(ranked_nodes)<k:
            # Set highest scoring unranked node as current_node
            current_node = unranked_nodes.pop()

            if verbose:
                s = str(len(ranked_nodes))+"\t"
                if current_node.inc is not None:
                    s = s + "inc:" +str(len(current_node.inc))+ "\t"
                if current_node.exc is not None:
                    s = s + "exc:"+ str(len(current_node.exc))
                print(s)

            # If the current node has forced included pairings, get a list of
            # all parts of the matching that can vary.
            if current_node.inc is not None:
                matching_reduced = current_node.matching[
                                            ~current_node.matching["SS_name"].
                                            isin(current_node.inc["SS_name"])]
            else:
                matching_reduced = current_node.matching

            # Create child nodes and add them to the unranked_nodes list
            # -1 in range(matching_reduced.shape[0]-1) is because forcing
            # inclusion of all but one residue actually results in getting back
            # the original matching, and also results in an infinite loop of
            # identical child nodes...
            for i in range(matching_reduced.shape[0]-1):
                #print("\t", i)
                #Construct dataframes of excluded and included pairs
                exc_i = matching_reduced.iloc[[i],:]
                if current_node.exc is not None:
                    # exc_i = exc_i.append(current_node.exc, ignore_index=True)
                    exc_i = pd.concat([exc_i, current_node.exc], ignore_index=True)
                inc_i = matching_reduced.iloc[0:i,:]
                if current_node.inc is not None:
                    # inc_i = inc_i.append(current_node.inc, ignore_index=True)
                    inc_i = pd.concat([inc_i, current_node.inc], ignore_index=True)
                inc_i = inc_i.drop_duplicates()
                # If there are no included pairs, it's better for inc_i to be None than an empty df
                if inc_i.shape[0]==0:
                    inc_i = None

                matching_i = self.find_best_assignment(inc=inc_i, exc=exc_i,
                                                     return_none_if_all_dummy=True,
                                                     verbose=False)
                if matching_i is None:
                    # If the non-constrained residues or spin systems are all
                    # dummies, find_best_assignment will return None, and this
                    # node can be discarded
                    pass
                else:
                    # Create a new child node and add to unranked_nodes
                    matching_i.index = matching_i["SS_name"]
                    matching_i.index.name = None
                    node_i = Node(self.calc_overall_matching_prob(matching_i),
                                  matching_i, inc_i, exc_i)
                    unranked_nodes.add(node_i)
            ranked_nodes.add(current_node)

        return(ranked_nodes, unranked_nodes)

    def find_consistent_assignments(self, threshold=0.2, set_assign_df=False):
        """Try to find a consistent set of assignments by optimising both match
        to predictions and mismatches between adjacent residues

        Returns an assign_df DataFrame, but does not modify the class

        Parameters
        threshold: the maximum allowed mismatch for a good sequential link
        """

        self.logger.info("Started assigning based on predictions and sequential links")
        preds = self.aligned_preds.copy()
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
                res_m1 = preds.loc[res,"Res_name_m1"]
                res_p1 = preds.loc[res,"Res_name_p1"]

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
    
    def find_consistent_assignments_2(self, threshold=0.2, set_assign_df=False, max_iterations=10):
        """Try to find a consistent set of assignments by optimising both match
        to predictions and mismatches between adjacent residues.
        In this version, only exclude the single incorrect match at the end of each
        connected fragment, instead of excluding all mismatches.

        Returns an assign_df DataFrame, but does not modify the class

        Parameters
        threshold: the maximum allowed mismatch for a good sequential link
        """

        self.logger.info("Started assigning based on predictions and sequential links")
        preds = self.aligned_preds.copy()
        assign_df0 = self.assign_from_preds()
        assign_df0 = self.add_consistency_info(assign_df0, threshold)
        best_assign_df = assign_df0
        N_HM_conf0 = assign_df0["Confidence"].isin(["High","Medium"]).sum()
        self.logger.info("At the current stage, there are "+str(N_HM_conf0)+
                         " high or medium confidence assignments")

        iterations = 0
        unsuccessful_iterations = 0

        a = self.copy()
        penalty = -1 # a.log_prob_matrix.min().min()

        while unsuccessful_iterations < max_iterations:
            # Find all residues with an adjacent confident residue
            HM_conf_res = assign_df0.loc[assign_df0["Confidence"].isin(["High","Medium"]),
                                         "Res_name"].tolist()

            #Make a dataframe for looking up which spin system is assigned to each residue
            tmp = assign_df0[["Res_name","SS_name"]]
            tmp.index = tmp["Res_name"]

            # Penalise any spin system assignments that don't match a neighbouring confident assignment
            
            breakpoint()
            for res in HM_conf_res:
                # Get the neighbouring residues
                res_m1 = preds.loc[res,"Res_name_m1"]
                res_p1 = preds.loc[res,"Res_name_p1"]
                
                ss = tmp.SS_name[res]

                # (Need to be careful with NaN values at this point)    
                if not pd.isna(res_m1) and res_m1 not in HM_conf_res:
                    ss_m1 = tmp.loc[res_m1,"SS_name"]
                    # Set the inconsistent spin system to have a high log_prob
                    # penalty is added so in case no spin systems are allowed -
                    # this way, the predictions do still have an influence.
                    if ~a.obs.Dummy_SS[ss_m1]:          # Don't penalise dummy residues
                        a.log_prob_matrix.loc[ss_m1, res_m1] += penalty
                        print(res_m1, ss_m1)
                if not pd.isna(res_p1) and res_p1 not in HM_conf_res:
                    ss_p1 = tmp.loc[res_p1,"SS_name"]
                    #Set the inconsistent spins system to have a high log_prob
                    if ~a.obs.Dummy_SS[ss_p1]:          # Don't penalise dummy residues
                        a.log_prob_matrix.loc[ss_p1, res_p1] += penalty
                        print(res_p1, ss_p1)

            # Make a list of high confidence assignments to enforce 
            high_conf_assignments = assign_df0.loc[assign_df0["Confidence"].isin(["High"]),["SS_name","Res_name"]]

            # Rerun assignment and see how many are consistent now
            assign_df1 = a.assign_from_preds(inc=high_conf_assignments)
            assign_df1 = a.add_consistency_info(assign_df1, threshold)
            N_HM_conf1 = assign_df1["Confidence"].isin(["High","Medium"]).sum()
            print()

            # Check if the number of high/medium confidence assignments has increased
            # If yes, try another round of consrtained assignment
            # If no, stick with the previous best assignment.
            if N_HM_conf1 > N_HM_conf0:
                assign_df0 = assign_df1
                best_assign_df = assign_df1
                N_HM_conf0 = N_HM_conf1
                unsuccessful_iterations = 0
            else:
                assign_df0 = assign_df1
                unsuccessful_iterations += 1
            iterations += 1

            self.logger.info("After iteration "+str(iterations)+", there are "+str(N_HM_conf1)+
                            " high or medium confidence assignments")
            print("After iteration "+str(iterations)+", there are "+str(N_HM_conf1)+
                            " high or medium confidence assignments")

        if set_assign_df:
            self.assign_df = best_assign_df

        return(best_assign_df)
    
    def find_consistent_assignments_3(self, threshold=0.2, set_assign_df=False, 
                                      init_inc=None, init_exc=None, verbose=False,
                                      max_iterations = 50):
        """Try to find a consistent set of assignments by optimising both match
        to predictions and mismatches between adjacent residues.
        In this version, perform a tree search by finding the worst mismatch AB, creating three new nodes 
        (A&!B, B&!A, !A&!B), then following the one with the best total probability. Inspired by Murty 
        algorithm for k-best assignments (see: Murty, K. (1968). An Algorithm for Ranking all the Assignments 
        in Order of Increasing Cost. Operations Research, 16(3), 682-687)

        Returns an assign_df DataFrame, but does not modify the class

        Parameters
        threshold: the maximum allowed mismatch for a good sequential link
        """
        self.logger.info("Started assigning based on predictions and sequential links")
        
        # Node = namedtuple("Node", ["sum_log_prob","matching","inc","exc"])

        # Initial best matching (subject to initial constraints)
        best_matching = self.find_best_assignment(self.log_prob_matrix, maximise=True, 
                                                  inc=init_inc, exc=init_exc)
        best_matching.index = best_matching["Res_name"]     # Note that we'll index matching by Res_name
        best_matching.index.name = None

        # Define lists to keep track of nodes
        # ranked_nodes = SortedListWithKey(key=lambda n: n.sum_log_prob)
        # unranked_nodes = SortedListWithKey(
        #                     [Node(self.calc_overall_matching_prob(best_matching),
        #                     best_matching, inc=init_inc, exc=init_exc)],
        #                     key=lambda n: n.sum_log_prob)
        
        tmp = {"ID": 0, 
               "sum_log_prob": self.calc_overall_matching_prob(best_matching), 
               "worst_mismatch": -1.0,
               "matching": best_matching,
               "inc": init_inc, "exc": init_exc,
               "Ranked": False,
               "parent": None}    
                                                                                        
        node_df = pd.DataFrame([tmp])
        # breakpoint()
        iterations = 0
        while True:
            breakpoint()
            # Set highest scoring unranked node as current_node
            # current_node = unranked_nodes.pop()
            next_best_node_index = node_df.loc[~node_df.Ranked, "sum_log_prob"].idxmax()
            current_node = node_df.loc[next_best_node_index,:]

            if verbose:
                s = str(node_df.Ranked.sum())+"\t" + str(current_node.sum_log_prob) + "\t"
                if current_node.inc is not None:
                    s = s + "inc:" +str(len(current_node.inc))+ "\t"
                if current_node.exc is not None:
                    s = s + "exc:"+ str(len(current_node.exc))
                print(s)
                self.logger.info(s)

            # If the current node has forced included pairings, get a list of
            # all parts of the matching that can vary.
            if False:   # current_node.inc is not None:     # Currently disabled
                matching_reduced = current_node.matching[
                                            ~current_node.matching["SS_name"].
                                            isin(current_node.inc["SS_name"])]
            else:
                matching_reduced = current_node.matching
            matching_reduced = matching_reduced.sort_values("Res_name")
            
            # Find the worst mismatch in the current node
            consistency_df = self.check_matching_consistency(current_node.matching, threshold=threshold).sort_values("Res_name")
            worst_mismatch = consistency_df.Max_mismatch.max()
            node_df.loc[next_best_node_index, "worst_mismatch"] = worst_mismatch

            # I don't think this selects the correct rows (outdated?)
            tmp = consistency_df.Max_mismatch.idxmax()
            first_row = consistency_df.index.get_loc(tmp)
            pair_A = matching_reduced.iloc[[first_row],:]   # [] around first_row needed to return DataFrame instead of Series
            pair_B = matching_reduced.iloc[[first_row+1],:]
            pairs_AB = matching_reduced.iloc[[first_row,first_row+1],:]

            if verbose:
                s = "Worst mismatch at current node (%f): %s-%s, %s-%s" % (worst_mismatch, 
                                                                           pair_A.Res_name.values[0], pair_A.SS_name.values[0], 
                                                                           pair_B.Res_name.values[0], pair_B.SS_name.values[0])
                self.logger.info(s)
                print(s)

            # Code below doesn't account for worst pair overlapping with forced-included pairs

            # Create the 3 child nodes and add them to the unranked_nodes list
            for inc_exc in [(pair_A, pair_B),(pair_B, pair_A),(pairs_AB.loc[[],:], pairs_AB)]:
                # Set up included residues
                inc_i = inc_exc[0]
                exc_i = inc_exc[1]

                # breakpoint()
                if current_node.inc is not None: 
                    # Check if any of the excluded residues for this new node are also force-included
                    if set(exc_i.Res_name) & set(current_node.inc.Res_name):
                        print("Residue in both inc and exc")    # If there are any residues in both exc_i and curent_node.inc, do not create a new node
                        continue
                    # inc_i = inc_i.append(current_node.inc, ignore_index=True)
                    inc_i = pd.concat([inc_i, current_node.inc], ignore_index=True)
                    inc_i = inc_i.drop_duplicates()
                if inc_i.shape[0]==0:   # If there are no included residues, faster to replace with None
                    inc_i = None

                # Set up excluded resiudes
                
                if current_node.exc is not None:
                #     # Check if any of the force-included residues for this new node are also excluded
                #     # if set(inc_i.Res_name) & set(current_node.exc.Res_name):
                #     #     continue
                #     # exc_i = exc_i.append(current_node.exc, ignore_index=True)
                    exc_i = pd.concat([exc_i, current_node.exc], ignore_index=True)
                
                # breakpoint()

                # Find the best matching for the given included and excluded residues
                matching_i = self.find_best_assignment(self.log_prob_matrix, maximise=True,
                                                        inc=inc_i, exc=exc_i,
                                                        return_none_all_dummy=True)
                if matching_i is None:
                    # If the non-constrained residues or spin systems are all
                    # dummies, find_best_assignment will return None, and this
                    # node can be discarded
                    print("No matching found")
                    pass
                else:
                    # Create a new child node and add to unranked_nodes
                    matching_i.index = matching_i["Res_name"]
                    matching_i.index.name = None
                    # node_i = Node(self.calc_overall_matching_prob(matching_i),
                    #               matching_i, inc_i, exc_i)
                    # unranked_nodes.add(node_i)
                    tmp = {"ID": node_df.ID.max()+1, 
                        "sum_log_prob": self.calc_overall_matching_prob(matching_i), 
                        "worst_mismatch": -1.0,
                        "matching": matching_i,
                        "inc": inc_i, "exc": exc_i,
                        "Ranked": False,
                        "parent": current_node.ID}
                    # breakpoint()
                    node_df = pd.concat([node_df, pd.DataFrame([tmp])], ignore_index=True)  
                    node_df.index = node_df.ID
                    node_df.index.name = None
                    
                
            # ranked_nodes.add(current_node)
            node_df.loc[next_best_node_index, "Ranked"] = True
            node_df = node_df.sort_values("sum_log_prob", ascending=False)

            if worst_mismatch < threshold:
                break   # If the worst mismatch in current assignment is less than the threshold, stop searching.

            iterations += 1
            if iterations >= max_iterations:
                break
            # breakpoint()
        breakpoint()
        return(node_df)

        # assign_df0 = self.assign_from_preds()
        # best_assign_df = assign_df0
        # assign_df0 = self.add_consistency_info(assign_df0, threshold)
        # # self.logger.info("At the current stage, there are "+str(N_HM_conf0)+
        # #                  " high or medium confidence assignments")

        # # Find the worst mismatch in the initial assignment
        # tmp = assign_df0.Max_mismatch.idxmax()
        # worst_pair = ((assign_df0.loc[tmp, "Res_name"],assign_df0.loc[tmp, "SS_name"]),
        #               (assign_df0.loc[tmp+1, "Res_name"],assign_df0.loc[tmp+1, "SS_name"]))

        # # Create a dataframe to keep track of the nodes
        # tmp = {"ID": "", 
        #        "Total_log_prob": assign_df0.Log_prob.sum(), 
        #        "N_mismatches": (assign_df0.Max_mismatch>threshold).sum(),
        #        "Worst_mismatch": assign_df0.Max_mismatch.max(),
        #        "Worst_pair": worst_pair,
        #        "Assignment": (assign_df0.sort_values("Res_name").SS_name+";").sum()}    # List of SS_names, separated by ;, in order of ascending Res_name
        #                                                                                 # Should probably have a separator that is definitely not going to appear in names...
        # node_df = pd.DataFrame(tmp)

        # breakpoint()
        

        # if set_assign_df:
        #     self.assign_df = best_assign_df

        # return(best_assign_df)

    def find_consistent_assignments_4(self, threshold=0.2, 
                                      init_inc=None, init_exc=None,
                                      max_iterations=50, verbose=True):
        """Try to find a consistent set of assignments by optimising both match
        to predictions and mismatches between adjacent residues.
        In this version, perform a tree search by finding the worst mismatch AB, creating three new nodes 
        (A&!B, B&!A, !A&!B), then following the one with the best total probability. Inspired by Murty 
        algorithm for k-best assignments (see: Murty, K. (1968). An Algorithm for Ranking all the Assignments 
        in Order of Increasing Cost. Operations Research, 16(3), 682-687)

        In this version, all inconsistent assignments next to force-included residues
        are excluded at once, to try and speed up search.

        Returns an assign_df DataFrame, but does not modify the class

        Parameters
        threshold: the maximum allowed mismatch for a good sequential link
        """
        self.logger.info("Started assigning based on predictions and sequential links")

        # Initial best matching (subject to initial constraints)
        best_matching = self.find_best_assignment(self.log_prob_matrix, maximise=True, 
                                                  inc=init_inc, exc=init_exc).sort_values("Res_name")
        best_matching.index = best_matching["Res_name"]     # Note that we'll index matching by Res_name
        best_matching.index.name = None
        consistency_df = self.check_matching_consistency(best_matching, threshold=threshold).sort_values("Res_name")

        # Define dataframe to keep track of nodes
        node_df = pd.DataFrame([{"ID": 0, 
                "Parent": None, "Depth": 0, "ID2": 0.0,
                "Ranked": False, "Iteration": 0,
                "Sum_log_prob": self.calc_overall_matching_prob(best_matching), 
                "Worst_mismatch": consistency_df.Max_mismatch.max(),
                "N_high": (consistency_df.Confidence=="High").sum(), 
                "N_med": (consistency_df.Confidence=="Medium").sum(),
                "N_mismatch": (consistency_df.Max_mismatch_p1>=threshold).sum(),
                "Total_mismatch": consistency_df.Max_mismatch_p1.sum(),
                "Delta_mismatch": 0,
                "Chosen_mismatch": 0,
                "Matching": best_matching,
                "Inc": init_inc, "Exc": init_exc,
                "N_inc": 0, "N_exc": 0 }])
        node_df.set_index("ID")

        # Main loop: create and check new nodes until a consistent assignment is found, or iteration limit is reached
        iterations = 0
        while True:
            # if iterations > 100: breakpoint()
            # breakpoint()

            # Choose the highest-scoring unranked node
            # next_node_index = node_df.loc[~node_df.Ranked, "Sum_log_prob"].idxmax()
            # next_node_index = (node_df.loc[~node_df.Ranked, "Sum_log_prob"] - node_df.loc[~node_df.Ranked, "N_mismatch"]).idxmax()
            # next_node_index = (node_df.loc[~node_df.Ranked, "N_high"] + node_df.loc[~node_df.Ranked, "N_med"]).idxmax()
            next_node_index = node_df.loc[~node_df.Ranked, :].sort_values(["Delta_mismatch","N_mismatch", "N_high","N_med"],ascending=[True, True, False, False]).index[0]
            next_node_index = node_df.loc[~node_df.Ranked, :].sort_values(["Delta_mismatch","N_mismatch", "N_high","N_med","Sum_log_prob"],ascending=[True, True, False, False, False]).index[0]
            current_node = node_df.loc[next_node_index,:].copy()
            node_df.loc[next_node_index, "Iteration"] = iterations

            # Choose which mismatch will be used to create child nodes
            consistency_df = self.check_matching_consistency(current_node.Matching, threshold=threshold).sort_values("Res_name")
            
 
            # Choose the largest mismatch
            # res_A = consistency_df.Max_mismatch_p1.idxmax()
            # res_B = consistency_df.Max_mismatch_m1.idxmax()

            # Prioritising mismatches where at least one pair has good links
            consistency_df["Res_name_m1"] = consistency_df.Res_name.shift(1)
            consistency_df["Res_name_p1"] = consistency_df.Res_name.shift(-1)
            consistency_df["Priority"] = 0.0
            consistency_df.loc[(consistency_df.Num_good_links_m1>0) &
                               (consistency_df.Max_mismatch_m1<threshold) &
                               (consistency_df.Max_mismatch_p1>threshold),"Priority"] = 1.0
            consistency_df.loc[(consistency_df.Num_good_links_p1>0) &
                               (consistency_df.Max_mismatch_p1<threshold) &
                               (consistency_df.Max_mismatch_m1>threshold),"Priority"] = 1.0
            
            next_res = consistency_df.sort_values(["Priority", "Max_mismatch"], ascending=False).index[0]
            if consistency_df.loc[next_res,"Max_mismatch_m1"] > consistency_df.loc[next_res,"Max_mismatch_p1"]:
                res_A = consistency_df.Res_name_m1[next_res]
                res_B = consistency_df.Res_name[next_res]
            else:
                res_A = consistency_df.Res_name[next_res]
                res_B = consistency_df.Res_name_p1[next_res]

            # prepare dataframes for including/excluding assignments
            # breakpoint()
            assn_df_A = current_node.Matching.loc[[res_A], :]
            assn_df_B = current_node.Matching.loc[[res_B], :]
            assn_df_AB = current_node.Matching.loc[[res_A,res_B], :]
            assn_df_empty = current_node.Matching.loc[[],:]

            # Create child nodes and add to node_df
            for child in ["A","B","!AB"]:
                # Set up included residues
                inc_assn = assn_df_empty
                if child=="A":
                    inc_assn = assn_df_A
                elif child=="B":
                    inc_assn = assn_df_B                   
                if current_node.Inc is not None:
                    inc_assn = pd.concat([inc_assn, current_node.Inc], ignore_index=True)
                    inc_assn = inc_assn.drop_duplicates()
                
                # Set up excluded residues
                exc_assn = assn_df_empty
                if child == "!AB":
                    exc_assn = assn_df_AB
                if current_node.Exc is not None:
                    exc_assn = pd.concat([exc_assn, current_node.Exc], ignore_index=True)

                # Remove duplicate constraints
                inc_assn["Res_SS"] = inc_assn.Res_name + inc_assn.SS_name
                exc_assn["Res_SS"] = exc_assn.Res_name + exc_assn.SS_name
                inc_assn = inc_assn.drop_duplicates("Res_SS").reset_index(drop=True)
                exc_assn = exc_assn.drop_duplicates("Res_SS").reset_index(drop=True)
                # inc_assn.index = inc_assn.Res_name
                # inc_assn.index = inc_assn.Res_name

                # Exclude any assignments that are inconsistent with the included residues
                exc_mask = pd.DataFrame(data=False, index=self.log_prob_matrix.index, columns=self.log_prob_matrix.columns)

                res_name = inc_assn.loc[:,"Res_name"]
                ss_name = inc_assn.loc[:, "SS_name"]
                res_name_m1 = self.aligned_preds.loc[res_name, "Res_name_m1"].reset_index(drop=True)
                res_name_p1 = self.aligned_preds.loc[res_name, "Res_name_p1"].reset_index(drop=True)
                # # Work out which spin systems should be excluded at the i-1 position
                ss_name_m1 = ss_name[~res_name_m1.isna()]
                res_name_m1 = res_name_m1[~res_name_m1.isna()]
                exc_mask_m1 = self.mismatch_matrix.loc[:, ss_name_m1] > threshold
                exc_mask_m1.columns = res_name_m1
                exc_mask.loc[:,res_name_m1] = exc_mask.loc[:,res_name_m1] | exc_mask_m1
                # # Work out which spin systems should be excluded at the i+1 position
                ss_name_p1 = ss_name[~res_name_p1.isna()]
                res_name_p1 = res_name_p1[~res_name_p1.isna()]
                exc_mask_p1 = (self.mismatch_matrix.loc[ss_name_p1, :] > threshold).transpose()
                exc_mask_p1.columns = res_name_p1
                exc_mask.loc[:,res_name_p1] = exc_mask.loc[:,res_name_p1] | exc_mask_p1

                
                # Add exc_assn pairs into exc_mask
                for i in exc_assn.index: 
                    exc_mask.loc[exc_assn.SS_name[i], exc_assn.Res_name[i]] = True

                # Discard node if the constraints are inconsistent
                # (ie. if any assignments are both included and excluded)
                
                inconsistent_constraints = False
                for i in inc_assn.index:
                    if exc_mask.loc[inc_assn.SS_name[i], inc_assn.Res_name[i]]:
                        inconsistent_constraints = True
                        break
                # if iterations > 100: breakpoint()
                if inconsistent_constraints:
                    self.logger.info("Inconsistent contraints found for child %s of node %d, so it was discarded" % (child, current_node.ID))
                    # breakpoint()
                    continue


                # Find the best matching
                # if inc_assn.shape[0]==0:    # If there are no included residues, faster to replace with None
                #     inc_assn = None
                matching = self.find_best_assignment(self.log_prob_matrix, maximise=True,
                                                        inc=inc_assn[["Res_name","SS_name"]], 
                                                        exc_mask=exc_mask,
                                                        return_none_all_dummy=True)
                if matching is None:
                    # If the non-constrained residues or spin systems are all
                    # dummies, find_best_assignment will return None, and this
                    # node can be discarded
                    self.logger.info("No matching found for child %s of node %d, so it was discarded" % (child, current_node.ID))
                    # breakpoint()
                    continue
                
                # Check whether any excluded assignments hae made it into the matching.
                tmp = df_lookup(exc_mask, matching.SS_name, matching.Res_name)
                if tmp.sum() >0:
                    self.logger.info("Excluded assignments present in matching for child %s of node %d, so it was discarded" % (child, current_node.ID))
                    # breakpoint()
                    continue

                # Add node to node_df
                matching.index = matching["Res_name"]
                matching.index.name = None
                new_consistency_df = self.check_matching_consistency(matching, threshold=threshold).sort_values("Res_name")

                if child == "A":
                    tmp = 0.0
                elif child=="B":
                    tmp=1.0/3
                else:
                    tmp=2.0/3

                new_node = pd.DataFrame([{"ID": node_df.ID.max() + 1, 
                            "Parent": current_node.ID, 
                            "Depth": current_node.Depth + 1,
                            "Ranked": False,
                            "ID2": current_node.ID2+tmp*3.0**-current_node.Depth,
                            "Iteration": np.nan,
                            "Sum_log_prob": self.calc_overall_matching_prob(matching), 
                            "Worst_mismatch": new_consistency_df.Max_mismatch.max(),
                            "N_high": (new_consistency_df.Confidence=="High").sum(), 
                            "N_med": (new_consistency_df.Confidence=="Medium").sum(),
                            "N_mismatch": (new_consistency_df.Max_mismatch_p1>=threshold).sum(),
                            "Total_mismatch": new_consistency_df.Max_mismatch_p1.sum(),
                            "Delta_mismatch": min(0, (new_consistency_df.Max_mismatch_p1>=threshold).sum()- current_node.N_mismatch),   
                                            # Change in mismatches relative to parent, but capped at +0. So fewer mismatches results in a -ve value, increased mismatches gives 0.
                            "Chosen_mismatch": self.aligned_preds.loc[res_A, "Res_N"],
                            "Matching": matching,
                            "Inc": inc_assn, "Exc": exc_assn,
                            "N_inc": inc_assn.index.size, "N_exc": exc_mask.sum().sum() }])
                
                node_df = pd.concat([node_df, new_node], ignore_index=True)  
                node_df.index = node_df.ID
                node_df.index.name = None
            
            node_df.loc[next_node_index, "Ranked"] = True
            node_df = node_df.sort_values("Sum_log_prob", ascending=False)

            # If a consistent assignment has been found, exit the loop
            if current_node.Worst_mismatch < threshold:
                break
            
            if verbose: print(iterations, current_node.ID, current_node.Parent, current_node.Depth,
                    current_node.N_mismatch, current_node.N_high, current_node.N_med,
                    current_node.N_inc, "%.2f" % current_node.Sum_log_prob)
            
            iterations += 1
            # Exit the loop if max iterations reached
            if iterations >= max_iterations:
                break
        
        # breakpoint()
        return(node_df)

    def find_seq_assignment(self):
        """Find the ordering that maximises the number of good sequential links"""

        row_ind, col_ind = linear_sum_assignment(self.mismatch_matrix-1*self.consistent_links_matrix)
        matching = pd.DataFrame({"i_m1":self.consistent_links_matrix.index[row_ind],
                                 "i":self.consistent_links_matrix.columns[col_ind]})


        return(matching)

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

            output_df = df.loc[:, ["Group","Atom","Nuc","Shift"]]
            output_df["Sdev"] = 0.0
            output_df["Assignments"] = int(1)

            output_df = output_df.dropna()

            if filepath is not None:
                output_df.to_csv(filepath, sep="\t", float_format="%.3f",
                                 index=False)
        elif format=="xeasy":
            df.loc[df["Atom_type"]=="H","Atom_type"] = "HN"

            output_df = df.loc[:,["Shift","Atom_type","Res_N"]]
            output_df.insert(1, "Sdev", 0)

            output_df["Shift"] = output_df["Shift"].fillna(999.0)
            output_df = output_df.dropna()
            output_df = output_df.reset_index(drop=True)

            if filepath is not None:
                output_df.to_csv(filepath, sep="\t", float_format="%.3f",
                                 index=True, header=False)
        elif format=="nmrpipe":
            df.loc[df["Atom_type"]=="H","Atom_type"] = "HN"

            output_df = df.loc[:, ["Res_N", "Res_type", "Atom_type", "Shift"]]

            output_df["Shift"] = output_df["Shift"].fillna(9999.0)
            output_df = output_df.dropna()

            # NmrPipe requres a sequence, but we don't necessarily have a complete one
            # Piece togther what we can from all_preds, and put X at other positions
            # Have updated to use seq_df, which should be complete.
            tmp = pd.DataFrame({"Res_N":np.arange(self.seq_df["Res_N"].min(),
                                                  self.seq_df["Res_N"].max()+1)})
            tmp = tmp.merge(self.seq_df[["Res_N","Res_type"]], how="left", on="Res_N")
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

    def output_peaklists(self, filepath, format="sparky",
                         spectra=["hsqc","hnco","hncaco","hncacb", "hncocacb"]):
        """ Output assigned peaklists
        """
        return(0)


    def plot_strips(self, outfile=None, format="html", return_json=True, plot_width=1000, assign_df=None):
        """Make a strip plot of the assignment.

        Uses bokeh module for plotting. Returns the bokeh plot object.

        outfile: if defined, the plot will be saved to this location
        format: type of output. Either "html" or "png"
        return_json: if tue, return the plot as a json object
        plot_width: The width of the output plot in pixels
        """
        if assign_df is None:
            df = self.assign_df
        else:
            df = assign_df
            
        plotlist = []

        if not self.sequential_atoms_present(df.columns):
            # You can't draw a strip plot
            self.logger.warning("No sequential links in data - strip plot not drawn.")
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
                vlines[atom+"_z"] = np.nan
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
                hlines[atom+"_a"] = np.nan
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

        if not pd.Series(["H","N"]).isin(assign_df.columns).all():
            # You can't draw an HSQC plot
            self.logger.warning("Observed data does not contain both H and N - HSQC cannot be plotted.")
            return(None)

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
            plt.circle(tmp["H"], tmp["N"], radius=0.02, color=colourmap[k], legend_label=k)

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

    def plot_prob_matrix(self, prob_matrix, matching=None, lower_limit=None, upper_limit=None,
                         outfile=None, format="html", return_json=True, plot_width=750):
        """Plot a heatmap showing the probability matrix."""
        # Get names of Residues and spin systems
        if matching is not None:
            Res_names = matching.Res_name.to_list()
            SS_names = matching.SS_name.to_list()
        else:
            Res_names = prob_matrix.columns.to_list()
            SS_names = prob_matrix.index.to_list()

        # Convert probability matrix from wide to long
        prob_matrix = prob_matrix.reset_index()
        prob_long = prob_matrix.melt(id_vars="SS_name", value_name="Probability")

        plt = heatmap(x=prob_long.Res_name, y=prob_long.SS_name, z=prob_long.Probability,
                    x_range=Res_names, y_range=SS_names, z_range=[lower_limit,upper_limit],
                    frame_width=plot_width, frame_height=plot_width, flip_y_axis=True)
        
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