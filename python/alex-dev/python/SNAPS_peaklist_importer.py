# This script reads a list describing a set of peak lists, imports the peak lists 
# then converts them into a list of observed chemical shifts for use with SNAPS

import numpy as np
import pandas as pd
from SNAPS_importer import SNAPS_importer
from SNAPS_assigner import SNAPS_assigner, df_lookup
import logging
import argparse
import sys

# Read the list of spectra to imoprt
parser = argparse.ArgumentParser(description="SNAPS peaklist importer")
parser.add_argument("peaklist_csv", default=None, 
                    help="""A csv file describing the spectrum type, format and location of peaklists.
                    The columns should be Spectrum, File_type, File_name.
                    The HSQC should be on the first row.""")
parser.add_argument("out_file", default="obs_shifts.csv",
                    help="""The file where chemical shifts will be output.""")

args = parser.parse_args(sys.argv[1:])

peaklists = pd.read_csv(args.peaklist_csv,)

# Read in the actual peaklists, then generate the chemical shift data frame
importer = SNAPS_importer()

for i in peaklists.index:
    if peaklists.Spectrum[i]=="hsqc":
        importer.import_hsqc_peaks(peaklists.File_name[i], peaklists.File_type[i])
    else:
        importer.import_3d_peaks(peaklists.File_name[i], peaklists.File_type[i], peaklists.Spectrum[i])

importer.find_shifts_from_peaks()

importer.obs.to_csv(args.out_file, sep="\t", float_format="%.3f", index=False)

tmp = importer.import_obs_shifts(args.out_file, "snaps_obs")