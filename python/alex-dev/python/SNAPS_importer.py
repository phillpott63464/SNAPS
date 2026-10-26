# -*- coding: utf-8 -*-
"""
Defines a class with functions related to importing peak lists and shift lists.

@author: Alex
"""

import numpy as np
import pandas as pd
import itertools
from pathlib import Path
from Bio.SeqUtils import seq1
from math import sqrt

from NEF_reader import read_nef_shifts_from_file_to_pandas


#import nmrstarlib

class SNAPS_importer:
    # Attributes
#    peaklists = {}
#    assignments = {}
#    roots = None
#    peaks = None
#    obs = None
    
    def __init__(self):
        self.peaklists = {}
        #assignments = {}
        self.roots = None
        self.obs = None
    
    # Functions
    def import_hsqc_peaks(self, filename, filetype):
        """ Import a peak list for an HSQC spectrum. 
        
        filename: Path to file containing an HSQC peak list.
        filetype: Allowed values are "ccpn", "sparky", "xeasy" or "nmrpipe"
        """
        # Import from file
        if filetype=="ccpn":
            hsqc = pd.read_table(filename) 
            hsqc = hsqc[["Assign F1","Position F1","Position F2", "Height"]]
            hsqc.columns = ["SS_name","H","N","Height"]      
            #hsqc.index = hsqc["SS_name"]
        elif filetype=="sparky":
            hsqc = pd.read_table(filename, sep="\s+")
            hsqc.columns = ["SS_name","H","N","Height"]
            # If assigned, Name has format "A123HN-A123N"
            # If unassigned, Name column contains "?-?"
            # Get just the first part before the hyphen
            tmp = list(zip(*hsqc["SS_name"].str.split("-")))[0]
            hsqc["SS_name"] = tmp
            
            # Give arbitrary names to the unassigned peaks
            N_unassigned = sum(hsqc["SS_name"]=="?")
            hsqc.loc[hsqc["SS_name"]=="?", "SS_name"] = list("x"+
                    pd.Series(range(N_unassigned)).astype(str))
            
        elif filetype=="xeasy":
            hsqc = pd.read_table(filename, sep="\s+", comment="#", header=None,
                                 usecols=[0,1,2,5], 
                                 names=["SS_name","H","N","Height"])
            hsqc["SS_name"] = "x" + hsqc["SS_name"].astype(str)
        elif filetype=="nmrpipe":
            # Work out where the column names and data are
            with open(filename, 'r') as f:
                for num, line in enumerate(f, 1):
                    if line.find("VARS")>-1:
                        colnames_line = num
                        colnames = line.split()[1:]
            
            hsqc = pd.read_table(filename, sep="\s+", skiprows=colnames_line+1,
                                names=colnames)
            hsqc = hsqc[["INDEX", "ASS", "X_PPM", "Y_PPM", "HEIGHT"]]
            hsqc.columns = ["ID", "SS_name", "H", "N", "Height"]
            # Use ASS as SS_name if available, otherwise use ID
            hsqc.loc[hsqc["SS_name"]!=None,"SS_name"] = \
                hsqc.loc[hsqc["SS_name"]!=None,"ID"].astype(str)
            hsqc = hsqc[["SS_name", "H", "N", "Height"]]
        else:
            print("import_hsqc_peaks: invalid filetype '%s'." % (filetype))
            return(None)
        
        # Check whether columns go mixed up somehow, and correct if so
        if hsqc["H"].mean() > hsqc["N"].mean():
            tmp = hsqc["N"]
            hsqc["N"] = hsqc["H"]
            hsqc["H"] = tmp
        
        hsqc.index = hsqc["SS_name"]
        
        self.peaklists["hsqc"] = hsqc
        self.roots = hsqc
        
        return(self.roots)
        
    def import_3d_peaks(self, filename, filetype, spectrum, 
                        assign_nearest_root=False):
        """Import a 3D peak list in various formats
        
        filetype: one of "ccpn", "sparky", "xeasy" or "nmrpipe"
        spectrum: one of "hnco", "hncaco", "hnca", "hncoca", "hncacb",
            "hncocacb" or "hnha"
        assign_nearest_root: If True, this will assign each peak to the spin 
            system with the closest root (hsqc) peak. If False, peaks will be 
            assigned to spin systems based on information in the original file. 
            In this case, the proton assignment is used for CCPN and Sparky, 
            while the ASS column is used for nmrPipe. Xeasy peaklists alone 
            do not seem to contain assignment information.
        """
        if filetype == "ccpn":
            peaks = pd.read_table(filename,
                                  usecols=["Position F1","Position F2",
                                           "Position F3","Assign F1",
                                           "Assign F2","Assign F3",
                                           "Height"])
            peaks.columns = ["F1","F2","F3","A1","A2","A3","Height"]
        elif filetype == "sparky":
            peaks = pd.read_table(filename, sep="\s+")
            peaks.columns=["Name","F1","F2","F3","Height"]
            peaks["A1"], peaks["A2"], peaks["A3"] = list(zip(
                                                *peaks["Name"].str.split("-")))
            #return(peaks)
        elif filetype == "xeasy":
            peaks = pd.read_table(filename, sep="\s+", comment="#", header=None,
                                 usecols=[1,2,3,6], 
                                 names=["F1","F2","F3","Height"])
            peaks["SS_name"] = None
        elif filetype == "nmrpipe":
            # Work out where the column names and data are
            with open(filename, 'r') as f:
                for num, line in enumerate(f, 1):
                    if line.find("VARS")>-1:
                        colnames_line = num
                        colnames = line.split()[1:]
                        
            peaks = pd.read_table(filename, sep="\s+", skiprows=colnames_line+1,
                                names=colnames)
            peaks = peaks[["ASS", "X_PPM", "Y_PPM", "Z_PPM", "HEIGHT"]]
            peaks.columns = ["SS_name", "F1", "F2", "F3", "Height"]
            
        else:
            print("import_3d_peaks: invalid filetype '%s'." % (filetype))
            return(None)
            
        # Work out which column contains which dimension
        dim = {}
        for i in ["1","2","3"]:
            if peaks["F"+i].mean() > 150:
                dim["C"] = i
            elif peaks["F"+i].mean() > 100:
                dim["N"] = i
            elif peaks["F"+i].mean() > 15:
                dim["C"] = i
            elif peaks["F"+i].mean() > 6:
                dim["H"] = i
            elif peaks["F"+i].mean() > 0:
                dim["HA"] = i
            else: 
                dim["Unknown"] = i
        
        # Check that the correct dimensions were identified
        # Also check that spectrum argument is valid
        if spectrum in ["hnco", "hncaco", "hnca","hncoca","hncacb","hncocacb","hncb","hncocb"]:
            if set(dim.keys()) != set(["H","N","C"]):
                print("Error: couldn't identify "+spectrum+" columns.") 
        elif spectrum == "hnha":
            if set(dim.keys()) != set(["H","N","HA"]):
                print("Error: couldn't identify "+spectrum+" columns.")
        else:
            print("Invalid value of argument: spectrum.")
        
        # Populate SS_name column
        if filetype in ["ccpn", "sparky"]:
            peaks["SS_name"] = peaks["A"+dim["H"]]
        elif filetype == "nmrpipe":
            peaks["SS_name"] = peaks["ASS"]

        # Choose and rename columns. 
        peaks = peaks[["SS_name", "F"+dim["H"], "F"+dim["N"], "F"+dim["C"], 
                       "Height"]]
        peaks.columns = ["SS_name", "H", "N", "C", "Height"]
        peaks["Spectrum"] = spectrum
        
        # If assign_nearest_root, find closest root resonance for each peak 
        # and set that as SS_name.
        if assign_nearest_root or spectrum=="xeasy":
            peaks["SS_name"] == None
            
            roots = self.roots
            for i in peaks.index:
                delta = ((roots["H"]-peaks.loc[i,"H"])**2 
                         + (0.2*(roots["N"]-peaks.loc[i,"N"]))**2).apply(sqrt)
                peaks.loc[i, "SS_name"] = roots.loc[delta.idxmin(), "SS_name"]

        # Also, only keep spin systems that are in self.roots
        #peaks = peaks.loc[peaks["SS_name"].isin(self.roots["SS_name"])]
        
        # Add peaks to the peaklist dictionary
        self.peaklists[spectrum] = peaks
            
        return(self.peaklists[spectrum])
        
    def find_shifts_from_peaks(self):
        """ Work out chemical shifts for each spin system from peak lists
        
        Will use all spectra in the self.peaklists dictionary. The following
        spectra are supported: hnco, hncaco, hnca, hncoca, hncacb, hncocacb.
        """
        # Possible extension: could put a parameter to control hncacb sign interpretation
        obs = pd.DataFrame({"SS_name": self.roots["SS_name"], 
                            "H": self.roots["H"],
                            "N": self.roots["N"]})
        obs.index = obs["SS_name"]
        obs.index.name = None
        for spec in self.peaklists.keys():
            peaks = self.peaklists[spec]
            for ss in obs.loc[:, "SS_name"]:
                if ss in peaks["SS_name"].values:
                    ss_peaks = peaks.loc[peaks["SS_name"]==ss,:]
                    if spec=="hnco":
                        # Set C_m1 shift to strongest peak in this spin system.         
                        i = ss_peaks.loc[:,"Height"].idxmax()
                        obs.loc[ss, "C_m1"] = ss_peaks.loc[i, "C"]
                    elif spec=="hncaco":
                        # Set C shift to strongest peak in this spin system.         
                        i = ss_peaks.loc[:,"Height"].idxmax()
                        obs.loc[ss, "C"] = ss_peaks.loc[i, "C"]
                    elif spec=="hncoca":
                        # Set CA_m1 shift to strongest peak in this spin system.         
                        i = ss_peaks.loc[:,"Height"].idxmax()
                        obs.loc[ss, "CA_m1"] = ss_peaks.loc[i, "C"]
                    elif spec=="hnca":
                        # Set CA shift to strongest peak in this spin system.         
                        i = ss_peaks.loc[:,"Height"].idxmax()
                        obs.loc[ss, "CA"] = ss_peaks.loc[i, "C"]
                    elif spec=="hncocb":    # For HNcoCACB experiments tuned to only show CB peaks
                                            # Note this does not account for glycine peaks in any way!
                        # Set CB_m1 shift to strongest peak in this spin system.         
                        i = ss_peaks.loc[:,"Height"].idxmax()
                        obs.loc[ss, "CB_m1"] = ss_peaks.loc[i, "C"]
                    elif spec=="hncb":      # For HNCACB experiments tuned to only show CB peaks
                                            # Note this does not account for glycine peaks in any way! May lead to accidentally picking i-1 CB if i is glycine
                        # Set CB shift to strongest peak in this spin system.         
                        i = ss_peaks.loc[:,"Height"].idxmax()
                        obs.loc[ss, "CB"] = ss_peaks.loc[i, "C"]
                    elif spec=="hncocacb":
                        # Use a simple heuristic to guess if peak is CA or CB:
                        # - If only 1 peak, CA if shift >41 ppm, otherwise CB
                        # - If >1 peak, only keep the two with highest 
                        #   (absolute) intensity. 
                        # - If both >48 ppm, the largest shift is CB. 
                        #   Otherwise, the smallest shift is CB
                        if sum(peaks["SS_name"]==ss)==1:
                            if ss_peaks["C"].item()>41:
                                obs.loc[ss, "CA_m1"] = ss_peaks["C"].item()
                            else:
                                obs.loc[ss, "CB_m1"] = ss_peaks["C"].item()
                        else:
                            ss_peaks.loc[:,"Abs_height"] = ss_peaks["Height"].abs()
                            # Above line throws a SettingWithCopy warning, 
                            # but I can't seem to fix it
                            
                            ss_peaks = ss_peaks.sort_values(by="Abs_height",
                                                            ascending=False)
                            ss_peaks = ss_peaks.iloc[0:2,:]
                            C_max = ss_peaks["C"].max()
                            C_min = ss_peaks["C"].min()
                            if C_max>48 and C_min>48:
                                obs.loc[ss,"CA_m1"] = C_min
                                obs.loc[ss,"CB_m1"] = C_max
                            else:
                                obs.loc[ss,"CA_m1"] = C_max
                                obs.loc[ss,"CB_m1"] = C_min
                    elif spec=="hncacb":
                        # Use a simple heuristic to guess if peak is CA or CB:
                        # - If only 1 peak, CA if shift >41 ppm, otherwise CB
                        # - If >1 peak, only keep the two with highest 
                        #   (absolute) intensity. 
                        # - If strongest peak is 41-48 ppm and >twice height of
                        #   next highest, then it's glycine CA
                        # - Else, if both >48 ppm, the largest shift is CB. 
                        #   Otherwise, the smallest shift is CB
                        if sum(peaks["SS_name"]==ss)==1:
#                            print(ss_peaks)
#                            print(ss_peaks["C"])
#                            print(ss_peaks["C"].item())
                            if ss_peaks["C"].item()>41:
                                obs.loc[ss, "CA"] = ss_peaks["C"].item()
                            else:
                                obs.loc[ss, "CB"] = ss_peaks["C"].item()
                        else:
                            ss_peaks.loc[:,"Abs_height"] = ss_peaks["Height"].abs()
                            # Above line throws a SettingWithCopy warning, 
                            # but I can't seem to fix it
                            ss_peaks = ss_peaks.sort_values(by="Abs_height",
                                                            ascending=False)
                            ss_peaks = ss_peaks.iloc[0:2,:]
                            C_max = ss_peaks["C"].max()
                            C_min = ss_peaks["C"].min()
                            if (ss_peaks.iloc[0,:]["C"]>41 and 
                                ss_peaks.iloc[0,:]["C"]<48 and 
                                ss_peaks.iloc[0,:]["Abs_height"] >
                                2*ss_peaks.iloc[1,:]["Abs_height"]):
                                
                                obs.loc[ss,"CA"] = ss_peaks.iloc[0,:]["C"]
                            elif C_max>48 and C_min>48:
                                obs.loc[ss,"CA"] = C_min
                                obs.loc[ss,"CB"] = C_max
                            else:
                                obs.loc[ss,"CA"] = C_max
                                obs.loc[ss,"CB"] = C_min
                    elif spec=="hsqc":
                        break       # H and N shifts should already have been imported, so can skip.
                    else:
                        print("Spectrum type %s not recognised" % spec)
                        break
            
        self.obs = obs
        return(self.obs)
            
    def import_obs_shifts(self, filename, filetype, SS_num=False):
        """ Import a chemical shift list
        
        filename: Path to text file containing chemical shifts.
        filetype: Allowed values are "snaps", "ccpn", "sparky", "xeasy", 
            "nmrpipe", "mars" or "nef"
            The "ccpn" option is for importing a Resonance table exported from 
            Analysis v2.x. The "snaps" option is for importing an unassigned 
            shift table previously exported from SNAPS, while "snaps_obs" is for
            importing an observed shift table produced by 
        SS_num: If true, will extract the longest number from the SS_name and 
        treat it as the residue number. Without this, it is not possible to get
        the i-1 shifts for each spin system.
            
        """
        # Import from file
        if filetype=="snaps":
            obs = pd.read_table(filename)
        elif filetype=="snaps_obs":
            obs = pd.read_table(filename)
            obs.index.name = None
            self.obs = obs
            return self.obs
        elif filetype=="ccpn":
            obs = pd.read_table(filename)
            obs = obs.loc[:,["Residue", "Assign Name", "Shift"]]
            obs.columns = ["SS_name", "Atom_type", "Shift"]
            obs["Atom_type"] = obs["Atom_type"].str.upper()
        elif filetype=="sparky":
            obs = pd.read_table(filename, sep="\s+")
            obs = obs.loc[:,["Group", "Atom", "Shift"]]
            obs.columns = ["SS_name", "Atom_type", "Shift"]
            obs.loc[obs["Atom_type"]=="HN", "Atom_type"] = "H"
        elif filetype=="xeasy":
            obs = pd.read_table(filename, sep="\s+", 
                                header=None, na_values="999.000",
                                names=["i","Shift","SD","Atom_type","SS_name"])
            obs = obs.loc[:, ["SS_name", "Atom_type", "Shift"]]
            obs["SS_name"] = obs["SS_name"].astype(str)
            obs = obs.dropna(subset=["Shift"])
            obs.loc[obs["Atom_type"]=="HN", "Atom_type"] = "H"
        elif filetype=="nmrpipe":
            # Work out where the column names and data are
            with open(filename, 'r') as f:
                for num, line in enumerate(f, 1):
                    if line.find("VARS")>-1:
                        colnames_line = num
            
            obs = pd.read_table(filename, sep="\s+", skiprows=colnames_line+1, 
                                names=["SS_name","Res_type","Atom_type","Shift"])
            obs = obs.loc[:, ["SS_name", "Atom_type", "Shift"]]
            obs["SS_name"] = obs["SS_name"].astype(str)
            obs.loc[obs["Atom_type"]=="HN", "Atom_type"] = "H"
        elif filetype=="mars":
            obs_wide = pd.read_table(filename, sep="\s+", na_values="-")
            obs_wide = obs_wide.rename(columns={"CO":"C","CO-1":"C_m1",
                                                "CA-1":"CA_m1","CB-1":"CB_m1"})
            obs_wide = obs_wide.drop(columns="HA-1")
            obs_wide["SS_name"] = obs_wide.index.astype(str)
            obs = obs_wide.melt(id_vars="SS_name", 
                                value_vars=["H","HA","N","C","CA",
                                            "CB","C_m1","CA_m1","CB_m1"], 
                                var_name="Atom_type", value_name="Shift")
        elif filetype=='nef':
            obs = read_nef_shifts_from_file_to_pandas(filename)

#        elif filetype == "nmrstar":
#            tmp = nmrstarlib.read_files(filename)
#            return(tmp)
        else:
            print("import_obs_shifts: invalid filetype '%s'." % (filetype))
            return(None)
        
        # Restrict to backbone atom types
        obs = obs.loc[obs["Atom_type"].isin(["H","HA","N","C","CA","CB",
                                             "C_m1","CA_m1","CB_m1"]),:]
        
        # Convert from long to wide
        obs = obs.pivot(index="SS_name", columns="Atom_type", values="Shift")
        obs.insert(0, "SS_name", obs.index.values)
        
        # Extract residue number from SS_name (if present), and get m1 shifts
        if SS_num:
            obs["Res_N"] = obs["SS_name"].str.extract(r"(\d+)").astype(int)
            
            obs.index = obs["Res_N"]
            obs_m1 = obs[list({"C","CA","CB"}.intersection(obs.columns))]
            obs_m1.index = obs_m1.index+1
            obs_m1.columns = obs_m1.columns + "_m1"
            obs = pd.merge(obs, obs_m1, how="left", 
                           left_index=True, right_index=True)
            obs = obs.drop(columns="Res_N")
            
            # Set index back to SS_name
            obs.index = obs["SS_name"]
            obs.index.name = None

        # the index name shouldn't match a column name!
        obs.index.name = None
        self.obs = obs
        return self.obs
    
    def import_aa_type_info(self, filename, offset="i-1"):
        """ Add amino acid type information to previously-imported observed 
        shifts
        
        filename: a file with amino acid information
            This should have the format:
            SS_name_1   AVI   in   # to set AVI as the only allowed aa types
            SS_name_2   T     ex   # to exclude T from the allowed aa types
        offset: either "i" or "i_minus_1". Whether the aa type restriction 
            applies to the i spin system or to the preceeding i-1 spin system.
        """
        AA_str = "ACDEFGHIKLMNPQRSTVWY"
        
        if offset=="i":
            col = "SS_class"
        elif offset=="i-1":
            col = "SS_class_m1"
        else:
            print("invalid value of offset: must be 'i' or 'i-1'.")
            return(None)
        
        # Import file
        df = pd.read_table(filename, sep="\s+", comment="#", 
                                    header=None, names=["SS_name","AA","Type"])
        
        # For rows with Type=="in", the SS_class is the same as AA
        # For rows with Type=="ex", the SS_class is all aminos *except* AA
        df[col] = df["AA"]
        mask = (df["Type"]=="ex")
        for i in df.index[mask]:
            tmp = AA_str
            for c in df.loc[i, "AA"]:
                tmp = tmp.replace(c, "")
            df.loc[i, col] = tmp
        
        df.index = df["SS_name"]
        
        # Create SS_class column in obs DataFrame if it doesn't already exist.
        if col not in self.obs.columns:
            self.obs[col] = AA_str
        
        # Write SS_class info into obs data frame. Overwrite any previous info 
        # for these spin systems, but keep SS_class info for any spin systems 
        # not in df
        self.obs.loc[df.index, col] = df.loc[:, col]
       
        # Nan's can be any amino acid
        self.obs[col] = self.obs[col].fillna(AA_str)
        
        return(self.obs)
    
    def import_testset_shifts(self, filename, remove_Pro=True, 
                          short_aa_names=True, SS_class=None, SS_class_m1=None):
        """ Import observed chemical shifts from testset data
        
        This function is intended for use with test data only, and is unlikely 
        to work well on 'real' data.
        
        filename: The simplified BMRB file containing observed shift info.
        remove_Pro: If True, remove proline residues from output
        short_aa_names: If True, single letter aa codes are used, otherwise 3 
            letter codes are used
        SS_class: Either None or a list of strings, each of which is a list of 
            amino acids (eg. ["VIA","G","S","T","DN","FHYWC","REKPQML"] would 
            give the HADAMAC classes). If not None, a column SS_class will be 
            created which gives the class containing the residue type.
        SS_class_m1: as above, but for the i-1 residue.
        
        """
        #### Import the observed chemical shifts
        obs_long = pd.read_table(filename)
        obs_long = obs_long[["Residue_seq_code","Residue_label",
                             "Atom_name","Chem_shift_value"]]
            # Note that input file has another column "Residue_PDB_seq_code".
            # If you use that, you get weird slips in sequence numbering.
        obs_long.columns = ["Res_N","Res_type","Atom_type","Shift"]
        # Convert residue type to single-letter code
        if short_aa_names: 
            obs_long["Res_type"] = obs_long["Res_type"].apply(seq1)
            obs_long["SS_name"] = (obs_long["Res_N"].astype(str) + 
                    obs_long["Res_type"])
            obs_long["SS_name"] = obs_long["SS_name"].str.rjust(5)
        else:
            obs_long["SS_name"] = (obs_long["Res_N"].astype(str) + 
                                    obs_long["Res_type"])
            obs_long["SS_name"] = obs_long["SS_name"].str.rjust(7)
            obs_long["Res_type"] = obs_long["Res_type"].apply(seq1)
        obs_long = obs_long.reindex(columns=["Res_N","Res_type","SS_name",
                                             "Atom_type","Shift"])
        
        # Convert from long to wide
        obs = obs_long.pivot(index="Res_N", columns="Atom_type", 
                             values="Shift")
        
        # Add the other columns back in
        tmp = obs_long[["Res_N","Res_type","SS_name"]]
        tmp = tmp.drop_duplicates(subset="SS_name")
        tmp.index = tmp["Res_N"]
        obs = pd.concat([tmp, obs], axis=1)

        
        
        # Make columns for the i-1 observed shifts of C, CA and CB
        obs_m1 = obs[list({"C","CA","CB","Res_type"}.intersection(obs.columns))]
        obs_m1.index = obs_m1.index+1
        obs_m1.columns = obs_m1.columns + "_m1"
        obs = pd.merge(obs, obs_m1, how="left", left_index=True, 
                       right_index=True)
        
        # Restrict to specific atom types
        atom_set = {"H","N","C","CA","CB","C_m1","CA_m1","CB_m1","HA"}
        obs = obs[["Res_N","Res_type","Res_type_m1","SS_name"]+
                  list(atom_set.intersection(obs.columns))]
        
        # Add SS_class information
        if SS_class is not None:
            obs["SS_class"]=obs["Res_type"]
            for g in SS_class:
                obs["SS_class"] = obs["SS_class"].str.replace("["+g+"]", g)
        if SS_class_m1 is not None:
            obs["SS_class_m1"]=obs["Res_type_m1"]
            for g in SS_class_m1:
                obs["SS_class_m1"] = obs["SS_class_m1"].str.replace("["+g+"]", g)
        
        obs.index = obs["SS_name"]
        obs.index.name = None
        
        if remove_Pro:
            # Remove prolines, as they wouldn't be observed in a real spectrum
            obs = obs.drop(obs.index[obs["Res_type"].isin(["PRO","P"])]) 
        
        self.obs = obs
        return(self.obs)
    
    def export_obs_shifts(self, filename):
        """ Export a chemical shift list.
        
        The purpose of this function is to make a shift list which the user can 
        manually modify/correct, and then reimport.
        """
        
        df = self.obs.melt(id_vars="SS_name", 
                      value_vars=set(self.obs.columns).intersection({"H","N",
                                    "HA","C","CA","CB","C_m1","CA_m1","CB_m1"}), 
                      var_name="Atom_type", value_name="Shift")
        df = df.sort_values(by=["SS_name", "Atom_type"])
            
        df.to_csv(filename, sep="\t", float_format="%.3f", index=False)
        
        return(df)
        


