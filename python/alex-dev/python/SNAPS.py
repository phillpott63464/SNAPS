#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main SNAPS script for assigning an observed shift list based on predicted shifts

@author: aph516
"""
from tabulate import tabulate

from SNAPS_importer import SNAPS_importer
from SNAPS_assigner import SNAPS_assigner
import logging
from pathlib import Path

import pdb

# For testing
from plotnine import *

def get_arguments(system_args):
    import argparse

    parser = argparse.ArgumentParser(
            description="SNAPS (Simple NMR Assignments from Predicted Shifts)")

    # Mandatory arguments
    parser.add_argument("shift_file",
                        help="A table of observed chemical shifts.")
    parser.add_argument("pred_file",
                        help="A table of predicted chemical shifts.")
    parser.add_argument("output_dir",
                        help="The directory results will be written to.")

    # Information on input files and configuration options
    parser.add_argument("--shift_type",
                        choices=["snaps", "ccpn", "sparky", "mars",
                                 "xeasy", "nmrpipe", "nef", "test", "snaps_obs"],
                        default="snaps", 
                        help="The format of the observed shift file.")
    parser.add_argument("--pred_type",
                        choices=["shiftx2", "sparta+"],
                        default="shiftx2",
                        help="The format of the predicted shifts")
    parser.add_argument("--seq_file", default=None,
                        help="""A fasta file containing the protein sequence corresponding to the observed data.
                        Note that it does not have to exactly match the sequence used for shift prediction.""")
    parser.add_argument("--seq_numbering", default=1, 
                        help="""Information on how the sequence is numbered. Several options:
                        If a single number is given, this is used for the first residue in the sequence.
                        If a comma separated list of ranges is given (eg "-5:10,15:100"), the numbering will be discontinuous""")
    parser.add_argument("--pred_seq_offset", type=int, default=0,
                        help="""An offset to apply to the residue numbering in
                        the predicted shifts.""")
    parser.add_argument("-c", "--config_file",
                        default="../config/config.txt",
                        help="A file containing parameters for the analysis.")
    parser.add_argument("--test_aa_classes", default=None,
                        help="""For test data only.
                        A string containing a comma-separated list of the amino acid
                        classes for the i residue, a semicolon, then a list of AA
                        classes for the i-1 residue. No spaces.
                        eg. "ACDEFGHIKLMNPQRSTVWY;G,S,T,AVI,DN,FHYWC,REKPQML" for
                        a sequential HADAMAC """)
    parser.add_argument("--simulate_pred_shifts", action="store_true",
                        help="""If present and if shift_type="test", then use simulated 
                        predicted shifts instead of the ones given in pred_file""")
    parser.add_argument("--sim_pred_multiplier", type=float, default=0.0,
                        help="""If simulate_pred_shifts is True, multiply the atom_95_quartile
                        given in the config file by this value to get the standard deviation 
                        of the simulated shifts for each atom type.""")
    parser.add_argument("--sim_pred_seed", type=int, default=0,
                        help="Random seed used if simulate_pred_shifts is True")
    parser.add_argument("--test", default=None,
                        help="Test SNAPS using example data from the specified test protein")
    #TODO: Need to rethink how SS_class info is imported.

    # Options controlling output files
    parser.add_argument("--shift_output_type", default="sparky",
                        choices=["sparky", "xeasy", "nmrpipe"],
                        help="One or more output formats for chemical shift export")
    parser.add_argument("--shift_output_confidence", nargs="*",
                        choices=["High","Medium","Low","Unreliable","Undefined"],
                        default=["High","Medium","Low","Unreliable","Undefined"],
                        help="""Limits the shiftlist output to assignments with
                        particular confidence levels. More than one level is allowed""")
    parser.add_argument("--strip_plot",
                        action="store_true",
                        help="Output a strip plot to assess assignment quality.")
    parser.add_argument("--hsqc_plot",
                        action="store_true",
                        help="Output an HSQC plot of the assignments.")
    parser.add_argument("--probability_plot",
                        action="store_true",
                        help="Output a heatmap of the residue assignment probabilities.")



    args = parser.parse_args(system_args)
    if args.test is not None:   # For convenience when testing
        # args = parser.parse_args(("data/P3a_L273R/naps_shifts.txt",
        #                           "data/P3a_L273R/shiftx2.cs",
        #                           "output/test.txt",
        #                           "--shift_type","snaps",
        #                           "--pred_type","shiftx2",
        #                           "-c","config/config_yaml_2.txt",
        #                           "-l","output/test.log",
        #                           "--strip_plot_file", "output/strip_plot.htm",
        #                           "--hsqc_plot_file", "output/hsqc_plot.htm",
        #                           "--test"))
        import pandas as pd

        testset_df = pd.read_table("data/testset/testset.txt", header=None,
                                names=["ID","PDB","BMRB","Resolution","Length","Seq_offset"])
        testset_df["obs_file"] = [x for x in "data/testset/simplified_BMRB/"+testset_df["BMRB"].astype(str)+".txt"]
        # breakpoint()
        testset_df["preds_file"] = [x for x in "data/testset/shiftx2_results/"+testset_df["ID"]+"_"+testset_df["PDB"]+".cs"]
        testset_df["out_name"] = testset_df["ID"]+"_"+testset_df["BMRB"].astype(str)
        testset_df.index = testset_df["ID"]

        args = parser.parse_args((testset_df.loc[args.test, "obs_file"],
                                  testset_df.loc[args.test, "preds_file"],
                                  "output/test",
                                  "--seq_file", "data/testset/BMRB_seqs/"+testset_df.loc[args.test, "BMRB"].astype(str)+".txt",
                                  "--seq_numbering", str(testset_df.loc[args.test, "Seq_offset"]+1),
                                  "--shift_type","test",
                                  "--pred_type","shiftx2",
                                  "-c","config/config_yaml_2.txt",  # "-c","config/test/config_consistent_2.yaml",
                                  "--strip_plot",
                                  "--hsqc_plot",
                                  "--probability_plot",
                                  "--test", args.test))
    return(args)

def runSNAPS(system_args):

    #### Command line arguments
    args = get_arguments(system_args)

    # Create output directory, if it doesn't already exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    #### Set up logging
    # Create a logger
    logger = logging.getLogger("SNAPS")
    logger.setLevel(logging.DEBUG)
    # Create a log handler that writes to a specific file.
    # In principle you could have multiple handlers, but here I just have one.
    # Need to explicitly define a handler so it can be explicitly closed
    # once the analysis is complete.
    log_handler = logging.FileHandler(output_dir/"log.txt", mode='w')
    log_handler.setLevel(logging.DEBUG)
    #log_handler.setFormatter(logging.Formatter("%(levelname)s %(asctime)s %(module)s %(funcName)s %(message)s"))
    log_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(log_handler)


    #### Set up the SNAPS_assigner object
    a = SNAPS_assigner()

    # Import config file
    a.read_config_file(args.config_file)

    # Import observed and predicted shifts
    importer = SNAPS_importer()

    # Import sequence if available, and align predicted shifts
    a.import_sequence(args.seq_file, args.seq_numbering)

    if args.simulate_pred_shifts:
        # Calculate the errors for each atom as the 95% percentile interval multipled by 
        # the sim_pred_multiplier argument
        atom_errors = a.pars["atom_95_percentile"]
        for k in atom_errors.keys():
             atom_errors[k] = atom_errors[k]*args.sim_pred_multiplier

        a.simulate_pred_shifts(args.shift_file, atom_errors, args.sim_pred_seed)
    else:
        preds_2 = a.import_pred_shifts_2(args.pred_file, args.pred_type)
        a.map_preds_to_sequence(a.seq_df, preds_2)
        a.preds = a.aligned_preds

    

    # Import observed shifts
    if args.shift_type=="test":
        if args.test_aa_classes is None:
            importer.import_testset_shifts(args.shift_file)
        else:
            AA_class, AA_class_m1 = args.test_aa_classes.split(";")
            importer.import_testset_shifts(args.shift_file,
                                           SS_class=AA_class.split(","),
                                           SS_class_m1=AA_class_m1.split(","))
    else:
        importer.import_obs_shifts(args.shift_file, args.shift_type, SS_num=False)

    a.obs = importer.obs
    logger.info("Finished reading in %d spin systems from %s",
                 len(a.obs["SS_name"]), args.shift_file)

    #### Do the analysis
    a.prepare_obs_preds()
    a.prob_matrix = a.calc_prob_matrix(normalise_by="SS")

    if a.pars["use_generic_predictions"]:
        a.create_generic_predictions()
        a.generic_prob_matrix = a.calc_generic_prob_matrix()
        # breakpoint()
        a.prob_matrix *= a.generic_prob_matrix
        a.prob_matrix = a.prob_matrix.div(a.prob_matrix.sum(axis=1), axis="rows")
    
    # Test of probability normalisation
    for i in range(20):
        a.prob_matrix = a.prob_matrix.div(a.prob_matrix.sum(axis=0), axis="columns")
        a.prob_matrix = a.prob_matrix.div(a.prob_matrix.sum(axis=1), axis="rows")

    a.log_prob_matrix = a.calc_log_prob_matrix()
    # a.calc_log_prob_matrix_old()
    a.calc_mismatch_matrix()
    


    if a.pars["iterate_until_consistent"] == 1:
        a.assign_df = a.find_consistent_assignments(set_assign_df=True)
    else:
        a.assign_from_preds(set_assign_df=True)
        a.add_consistency_info(threshold=a.pars["seq_link_threshold"])
        if (a.pars["alt_assignments"] > 0):
            a.find_alt_assignments(N=a.pars["alt_assignments"])

        if a.pars["triplet_prob_iterations"] > 0:
            triplet_prob_list = [a.prob_matrix]
            triplet_log_prob_list = [a.log_prob_matrix]
            triplet_assign_df_list = [a.assign_df]
            for i in range(a.pars["triplet_prob_iterations"]):
                triplet_prob_matrix = a.calc_triplet_prob_matrix(prob_matrix=triplet_prob_list[i], normalise_by="SS")
                # Test of probability normalisation
                for i in range(20):
                    triplet_prob_matrix = triplet_prob_matrix.div(triplet_prob_matrix.sum(axis=0), axis="columns")
                    triplet_prob_matrix = triplet_prob_matrix.div(triplet_prob_matrix.sum(axis=1), axis="rows")
                
                triplet_log_prob_matrix = a.calc_log_prob_matrix(triplet_prob_matrix)
                triplet_assign_df = a.assign_from_preds(log_prob_matrix=triplet_log_prob_matrix, set_assign_df=False)
                triplet_assign_df = a.add_consistency_info(input_assign_df=triplet_assign_df, threshold=a.pars["seq_link_threshold"])
                
                triplet_prob_list += [triplet_prob_matrix]
                triplet_log_prob_list += [triplet_log_prob_matrix]
                triplet_assign_df_list += [triplet_assign_df]
       
            
    if a.pars["iterate_until_consistent"] == 2:
        high_conf_assn = a.assign_df.loc[a.assign_df.Confidence=="High", ["Res_name", "SS_name"]]
        a.assign_df = a.assign_df.sort_values("Res_N")
        a.assign_df["Confidence_m1"] = a.assign_df.Confidence.shift(1)
        a.assign_df["Confidence_p1"] = a.assign_df.Confidence.shift(-1)
        very_high_conf_assn = a.assign_df.loc[(a.assign_df.Confidence=="High") &
                                              (a.assign_df.Confidence_m1=="High") &
                                              (a.assign_df.Confidence_p1=="High"), ["Res_name", "SS_name"]]
        node_df = a.find_consistent_assignments_4(threshold=a.pars["seq_link_threshold"], max_iterations=a.pars["max_iterations"], verbose=True, init_inc=very_high_conf_assn)
        best_node = (node_df.N_high + node_df.N_med - node_df.N_mismatch).idxmax()
        best_matching = node_df.loc[best_node, "Matching"]
        b = a.copy()
        b.make_assign_df(best_matching, set_assign_df=True)
        b.add_consistency_info(threshold=a.pars["seq_link_threshold"])
        b.assign_df.to_csv(output_dir/"consistent_assign_df.tsv", sep="\t", float_format="%.3f",
                           index=False)
        node_df.to_csv(output_dir/"node_df.tsv", sep="\t", float_format="%.3f", index=False)
        b.plot_strips(output_dir/"strip_plot_consistent.htm", "html")
        
        # breakpoint()

        plt1 = ggplot(node_df[node_df.Ranked]) + geom_point(aes(x="Iteration",y="ID2", color="N_high+N_med"))
        plt2 = ggplot(node_df[node_df.Ranked]) + geom_point(aes(x="Iteration",y="Sum_log_prob", color="N_high+N_med"))
        plt3 = ggplot(node_df[node_df.Ranked]) + geom_point(aes(x="Iteration",y="N_mismatch", color="N_high+N_med"))
        plt4 = ggplot(node_df[node_df.Ranked]) + geom_point(aes(x="Iteration",y="Total_mismatch", color="N_high+N_med"))
        plt5 = ggplot(node_df[node_df.Ranked]) + geom_point(aes(x="Iteration",y="Chosen_mismatch", color="N_high+N_med"))
        plt1.save(output_dir/"test_history.pdf", verbose=False)
        plt2.save(output_dir/"test_history_probability.pdf", verbose=False)
        plt3.save(output_dir/"test_history_mismatches.pdf", verbose=False)
        plt4.save(output_dir/"test_history_total_mismatch.pdf", verbose=False)
        plt5.save(output_dir/"test_history_chosen_mismatch.pdf", verbose=False)
        
        # breakpoint()


    
    #### Output the results
    
    # Tabulate doesn't account for if some atom types are missing.
    # headings = '''
    #     Res_name Res_N Res_type SS_name Dummy_res Dummy_SS CA CA_pred HA HA_pred H H_pred CB CB_pred
    #      C C_pred N N_pred Log_prob Max_mismatch_m1 Max_mismatch_p1 Num_good_links_m1 
    # '''.split()
    # table = []
    # for df_index, df_row in a.assign_df.iterrows():
    #     table_row = []
    #     table.append(table_row)
    #     for heading in headings:
    #         table_row.append(df_row[heading])

    # with open(args.output_file, 'w') as fp:
    #     print(tabulate(table, tablefmt='plain', headers=headings), file=fp)
    
    a.assign_df.to_csv(output_dir/"assign_df.tsv", sep="\t", float_format="%.3f",
                           index=False)
    logger.info("Finished writing results to assign_df.tsv")

    if (a.pars["alt_assignments"] > 0):
        a.alt_assign_df.to_csv(output_dir/"alt_assign_df.tsv", sep="\t", float_format="%.3f",
                                index=False)
        logger.info("Finished writing alternative assignment results to alt_assign_df.tsv")

    #### Write chemical shift lists
    a.output_shiftlist(output_dir/"assigned_shifts.txt", args.shift_output_type,
                        confidence_list=args.shift_output_confidence)

    #### Make some plots
    plots = []
    if args.hsqc_plot:
        hsqc_plot = a.plot_hsqc(output_dir/"hsqc_plot.htm", "html")
        logger.info("Finished writing HSQC plot to hsqc_plot.htm")
        plots += [hsqc_plot]

    if args.strip_plot:
        strip_plot = a.plot_strips(output_dir/"strip_plot.htm", "html")
        logger.info("Finished writing strip plot to strip_plot.htm")
        plots += [strip_plot]

    if args.probability_plot:
        matching = a.assign_df.loc[:,["Res_name","SS_name"]]
        prob_plot = a.plot_prob_matrix(a.log_prob_matrix, matching, -10, 0,
                                    output_dir/"probability_plot.htm", "html",
                                    plot_width=750)
        plots += [prob_plot]
   
    if a.pars["triplet_prob_iterations"] > 0:
        for i in range(a.pars["triplet_prob_iterations"]):
            triplet_assign_df_list[i+1].to_csv(output_dir/("triplet_assign_df_iteration_%d.tsv" % (i+1)), sep="\t", float_format="%.3f",
                           index=False)
            if args.strip_plot:
                a.plot_strips(output_dir/("triplet_strip_plot_iteration_%d.htm" % (i+1)), "html", assign_df=triplet_assign_df_list[i+1])
            if args.probability_plot:
                matching = a.assign_df.loc[:,["Res_name","SS_name"]]
                a.plot_prob_matrix(triplet_log_prob_list[i+1], matching, -10, 0,
                                            output_dir/("triplet_probability_plot_iteration_%d.htm" % (i+1)), "html",
                                            plot_width=750)

    # Close the log file
    logger.handlers[0].close()
    logger.removeHandler(logger.handlers[0])

    return(plots)


#%% Run the actual script
if __name__ == '__main__':
    import sys

    runSNAPS(sys.argv[1:])
