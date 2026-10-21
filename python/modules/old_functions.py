"""
Functions that aren't actually used in SNAPS like... at all.
Removed to make codebase cleaner and easier to read. But kept just in case they become relevant.
"""

# Not sure what this does, only ever used in test4.py?
def import_sequence(self, filename, filetype="snaps"):
    """ Imports the protein sequence

    The sequence information is used to fill in any gaps if prediction
    information is missing. It can also be used if the residue numbering is
    discontinuous. The sequence file should be in FASTA format, and the
    sequence id should be either:
        1) the residue number of the first amino acid, or
        2) a comma separated list of residue ranges eg -5:10,15:100 (this is
        for proteins with discontinuous sequence numbering)
    If the file contains multiple sequences, only the first will be used.

    Parameters
    filename: path to file containing sequence information
    """

    fasta_records = SeqIO.parse(open(filename), "fasta")
    record1 = next(fasta_records)

    # Parse the sequence
    seq_list = list(str(record1.seq))

    # Parse the fasta id and make a list of residue numbers
    tmp = record1.id.split(",")
    if len(tmp) == 1:  # If id was "123" or "123-456"
        tmp2 = tmp[0].split(":")
        if len(tmp2) == 1:  # If id was "123"
            res_N_start = int(tmp2[0])
            res_N_list = list(range(res_N_start, res_N_start + len(seq_list)))
        else:  # If id was "123-456"
            start, end = tmp2
            res_N_list = list(range(int(start), int(end) + 1))
    else:  # If id was "123-200,300-456"
        res_N_list = []
        for x in tmp:
            start, end = x.split(":")
            res_N_list += list(range(int(start), int(end) + 1))

    # Check res_N_list and seq_list are the same length, and correct if not
    if len(res_N_list) > len(seq_list):
        res_N_list = res_N_list[0:len(seq_list)]
        self.logger.warning("The residue range provided is longer than the" +
                            " length of the sequence, so has been truncated")
    elif len(res_N_list) < len(seq_list):
        extra_needed = len(seq_list) - len(res_N_list)
        last_N = res_N_list[-1]
        res_N_list += list(range(last_N, last_N + extra_needed))
        self.logger.warning("The residue range provided is shorter than the" +
                            " length of the sequence, so has been extended")

    # Make a dataframe
    seq_df = pd.DataFrame({"Res_N": res_N_list, "Res_type": seq_list})
    seq_df["Res_name"] = seq_df["Res_N"].astype(str) + seq_df["Res_type"]
    seq_df["Res_name"] = seq_df["Res_name"].str.rjust(5)
    # Pad Res_name to constant length

    seq_df.index = seq_df["Res_name"]
    seq_df.index.name = None

    self.seq_df = seq_df
    self.logger.info("Finished importing a sequence of length %d" %
                     len(seq_df.index))

    return (self.seq_df)

"""
The following are never used anywhere, even in the test scripts. 
"""


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
    preds = preds[["Res_N", "Res_type"] +
                  list({"C", "CA", "CB", "H", "N", "HA"}.intersection(preds.columns))]

    # Add the random shifts
    for atom in self.pars["atom_set"].intersection(preds.columns):
        preds.loc[:, atom] += rand.normal(0, sd[atom], size=len(preds.index))

    # Add other columns back in
    preds.insert(1, "Res_name", (preds["Res_N"].astype(str) +
                                 preds["Res_type"]))
    preds.loc[:, "Res_name"] = preds["Res_name"].str.rjust(5)

    preds.index = preds["Res_N"]
    # preds.index.name=None

    # Make columns for the i-1 predicted shifts of C, CA and CB
    preds_m1 = preds[list({"C", "CA", "CB", "Res_type", "Res_name"}.
                          intersection(preds.columns))].copy()
    preds_m1.index = preds_m1.index + 1
    preds_m1.columns = preds_m1.columns + "_m1"
    preds = pd.merge(preds, preds_m1, how="left",
                     left_index=True, right_index=True)

    # Make column for the i+1 Res_name
    preds_p1 = preds[["Res_name"]].copy()
    preds_p1.index = preds_p1.index - 1
    preds_p1.columns = ["Res_name_p1"]
    preds = pd.merge(preds, preds_p1, how="left",
                     left_index=True, right_index=True)

    # Set index to Res_name
    preds.index = preds["Res_name"]
    preds.index.name = None

    # Restrict to only certain atom types
    atom_set = {"H", "N", "C", "CA", "CB", "C_m1", "CA_m1", "CB_m1", "HA"}
    preds = preds[["Res_name", "Res_N", "Res_type", "Res_name_m1",
                   "Res_name_p1", "Res_type_m1"] +
                  sorted(atom_set.intersection(preds.columns))]

    self.preds = preds
    return (self.preds)


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
    best_matching = self.find_best_assignments(inc=init_inc, exc=init_exc)
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
                exc_i = exc_i.append(current_node.exc, ignore_index=True)
            inc_i = matching_reduced.iloc[0:i,:]
            if current_node.inc is not None:
                inc_i = inc_i.append(current_node.inc, ignore_index=True)
            inc_i = inc_i.drop_duplicates()
            # If there are no included pairs, it's better for inc_i to be None than an empty df
            if inc_i.shape[0]==0:
                inc_i = None

            matching_i = self.find_best_assignments(inc=inc_i, exc=exc_i,
                                                 return_none_if_all_dummy=True,
                                                 verbose=False)
            if matching_i is None:
                # If the non-constrained residues or spin systems are all
                # dummies, find_best_assignments will return None, and this
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

def find_alt_assignments(self, N=1, by_ss=True, verbose=False):
        """ Find the next-best assignment(s) for each residue or spin system

        This works by setting the log probability to a very high value for each
        residue in turn, and rerunning the assignment

        Arguments:
        best_match_indexes: [row_ind, col_ind] output from find_best_assignment()
        N: number of alternative assignments to generate
        by_ss: if true, calculate next best assignment for each spin system.
            Otherwise, calculate it for each residue.

        Output:
        A Dataframe containing the original assignments, and the
        alt_assignments by rank
        """

        log_prob_matrix = self.log_prob_matrix
        best_matching = self.assign_df.loc[:,["SS_name","Res_name"]]
        best_matching.index = best_matching["SS_name"]
        best_matching.index.name = None
        alt_matching = None

        # Calculate sum probability for the best matching
        best_sum_prob = self.calc_overall_matching_prob(best_matching)

        # Calculate the value used to penalise the best match for each residue
        penalty = 2*log_prob_matrix.min().min()
        logging.debug("Penalty value: %f", penalty)

        # Initialise DataFrame for storing alt_assignments
        alt_matching_all = best_matching.copy()
        alt_matching_all["Rank"] = 1
        alt_matching_all["Rel_prob"] = 0

        for i in best_matching.index:   # Consider each spin system in turn
            ss = best_matching.loc[i, "SS_name"]
            res = best_matching.loc[i, "Res_name"]
            logging.debug("Finding alt assignments for original match %s - %s", ss, res)
            if verbose: print(ss, res)

            excluded = best_matching.loc[[i], :]

            for j in range(N):
                alt_matching = self.find_best_assignments(exc=excluded)

                alt_matching["Rank"] = j+2
                alt_sum_prob = self.calc_overall_matching_prob(alt_matching)
                alt_matching["Rel_prob"] = alt_sum_prob - best_sum_prob

                # Add the alt match for this ss or res to the results dataframe
                # and also the excluded dataframe.
                if by_ss:
                    # alt_matching_all = alt_matching_all.append(
                    #         alt_matching.loc[alt_matching["SS_name"]==ss, :],
                    #         ignore_index=True)
                    alt_matching_all = pd.concat([alt_matching_all, alt_matching.loc[alt_matching["SS_name"]==ss, :]], ignore_index=True)
                    res = alt_matching.loc[alt_matching["SS_name"]==ss,
                                           "Res_name"].tolist()[0]
                    # The .tolist()[0] is to convert a single-item series into a string.
                else:
                    # alt_matching_all = alt_matching_all.append(
                    #         alt_matching.loc[alt_matching["Res_name"]==res, :],
                    #         ignore_index=True)
                    alt_matching_all = pd.concat([alt_matching_all, alt_matching.loc[alt_matching["Res_name"]==res, :]], ignore_index=True)

                    ss = alt_matching.loc[alt_matching["Res_name"]==res,
                                          "SS_name"].tolist()[0]
                # excluded = excluded.append(pd.DataFrame({"SS_name":[ss],"Res_name":[res]}),
                #                            ignore_index=True)
                excluded = pd.concat([excluded, pd.DataFrame({"SS_name":[ss],"Res_name":[res]})], ignore_index=True)

        self.alt_assign_df = self.make_assign_df(alt_matching_all)
        if by_ss:
            self.alt_assign_df = self.alt_assign_df.sort_values(
                                                by=["SS_name", "Rank"])
        else:
            self.alt_assign_df = self.alt_assign_df.sort_values(
                                                by=["Res_name", "Rank"])

        return(self.alt_assign_df)

def output_peaklists(self, filepath, format="sparky",
                     spectra=["hsqc","hnco","hncaco","hncacb", "hncocacb"]):
    """ Output assigned peaklists
    """
    return(0)


def find_seq_assignment(self):
    """Find the ordering that maximises the number of good sequential links"""

    row_ind, col_ind = linear_sum_assignment(self.mismatch_matrix-1*self.consistent_links_matrix)
    matching = pd.DataFrame({"i_m1":self.consistent_links_matrix.index[row_ind],
                             "i":self.consistent_links_matrix.columns[col_ind]})


    return(matching)

# Technically referenced but only by functions in here
def calc_overall_matching_prob(self, matching):
    "Calculate the sum log probability of a particular matching"
    # return(sum(self.log_prob_matrix.lookup(matching["SS_name"],
    #                                            matching["Res_name"])))
    return(sum(df_lookup(self.log_prob_matrix, matching["SS_name"],
                                                matching["Res_name"])))
