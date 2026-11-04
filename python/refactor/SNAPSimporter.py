from pathlib import Path
from Bio import SeqIO, Align
import numpy as np
import pandas as pd
from refactor.dataclasses import ShiftData
from refactor.format_readers import (
    CCPNReader,
    FormatReader,
    MarsReader,
    NEFReader,
    NMRPipeReader,
    ShiftX2Reader,
    SnapsReader,
    SparkyReader,
    SpartaReader,
    XeasyReader,
)
from refactor.SNAPSconfig import SNAPSConfig, build_config


class SNAPSImporter:
    """Orchestrates format readers to import all required data"""

    def __init__(
        self,
        readers: list[FormatReader] | None = None,
        config: SNAPSConfig | None = None,
    ):
        self.readers = readers if readers is not None else self._default_readers()
        self._reader_by_format = {r.format_name: r for r in self.readers}
        self._reader_by_ext = {}
        for reader in self.readers:
            for ext in reader.file_extensions:
                self._reader_by_ext.setdefault(ext, []).append(reader)

        if config is None:
            self.config = build_config()
        else:
            self.config = config

    def _default_readers(self) -> list[FormatReader]:
        """Load all available format readers"""
        return [
            NEFReader(),
            ShiftX2Reader(),
            SparkyReader(),
            MarsReader(),
            XeasyReader(),
            NMRPipeReader(),
            SpartaReader(),
            CCPNReader(),
            SnapsReader(),
        ]

    def _get_reader(
        self, file_path: Path, format_hint: str | None = None
    ) -> FormatReader:
        """Find appropriate reader for file"""
        # Try format hint first
        if format_hint and format_hint in self._reader_by_format:
            return self._reader_by_format[format_hint]

        # # Try extension mapping
        ext = file_path.suffix.lower()
        if ext in self._reader_by_ext and len(self._reader_by_ext[ext]) == 1:
            return self._reader_by_ext[ext][0]

        # Try each reader's can_read method
        for reader in self.readers:
            result = reader.can_read(file_path=file_path)
            if result:
                return reader

        raise ValueError(f"No reader found for {file_path}")

    def clean_obs(self, obs_shift):
        obs_shift = obs_shift.loc[
            obs_shift["Atom_type"].isin(
                ["H", "HA", "N", "C", "CA", "CB", "C_m1", "CA_m1", "CB_m1"]
            ),
            :,
        ]

        # Convert from long to wide
        obs_shift = obs_shift.pivot(
            index="SS_name", columns="Atom_type", values="Shift"
        )
        obs_shift.insert(0, "SS_name", obs_shift.index.values)

        # the index name shouldn't match a column name!
        obs_shift.index.name = None

        return obs_shift

    def clean_preds_no_sequence(self, preds_long: pd.DataFrame, offset: int = 0):
        #### Initial processing and conversion from long to wide
        # Add sequence number offset and create residue names
        preds_long["Res_N"] = preds_long["Res_N"] + offset
        preds_long.insert(
            1,
            "Res_name",
            (preds_long["Res_N"].astype(str) + preds_long["Res_type"]),
        )
        # Left pad with spaces to a constant length (helps with sorting)
        preds_long["Res_name"] = preds_long["Res_name"].str.rjust(5)

        # Convert from long to wide format
        preds = preds_long.pivot(index="Res_N", columns="Atom_type", values="Shift")
        preds.index.name = None

        # Add the other residue name and type back in
        tmp = preds_long[["Res_N", "Res_type", "Res_name"]]
        tmp = tmp.drop_duplicates(subset="Res_name")
        tmp.index = tmp["Res_N"]
        tmp.index.name = None
        preds = pd.concat([tmp, preds], axis=1)

        #### Make consistent with seq_df (and create if it doesn't already exist)
        # TODO: Maybe this should be split off into a separate function?
        # If seq_df is missing, create it based on preds
        seq_df = preds.copy()[["Res_N", "Res_type", "Res_name"]]

        # If there are any missing residue numbers, create them
        min_N = seq_df["Res_N"].min()
        max_N = seq_df["Res_N"].max()
        missing_residue_numbers = set(range(min_N, max_N + 1)).difference(
            seq_df["Res_N"]
        )
        if len(missing_residue_numbers) > 0:
            tmp = pd.DataFrame(
                {
                    "Res_N": sorted(missing_residue_numbers),
                    "Res_type": "X",
                    "Res_name": np.nan,
                }
            )
            tmp["Res_name"] = tmp["Res_N"].astype(str) + tmp["Res_type"]
            tmp["Res_name"] = tmp["Res_name"].str.rjust(5)
            tmp.index = tmp["Res_N"]
            tmp.index.name = None
            # seq_df = seq_df.append(tmp).sort_index()
            seq_df = pd.concat([seq_df, tmp]).sort_index()

        # Add/delete residues from preds so it matches seq_df
        # Only keep predictions that are in seq_df
        tmp = len(preds.index)
        preds = preds[preds["Res_N"].isin(seq_df["Res_N"])]
        tmp2 = tmp - len(preds.index)

        # Add predictions for any residue number that is in seq_df but not preds
        tmp = pd.DataFrame(
            data=seq_df[~seq_df["Res_N"].isin(preds["Res_N"])],
            columns=preds.columns,
        )
        # preds = preds.append(tmp)
        preds = pd.concat([preds, tmp])

        # If Res_name is inconsistent between seq_df and preds, make a compromise
        preds.index = preds["Res_N"]
        preds = preds.sort_index()
        seq_df.index = seq_df["Res_N"]
        seq_df = seq_df.sort_index()
        mask = preds["Res_name"] != seq_df["Res_name"]

        if any(mask):
            ambiguous_res_names = (
                seq_df.loc[mask, "Res_name"] + "(" + preds.loc[mask, "Res_type"] + "?)"
            )
            preds.loc[mask, "Res_name"] = ambiguous_res_names
            seq_df.loc[mask, "Res_name"] = ambiguous_res_names
            # Note we don't update the Res_type in preds, because this is potentially
            # used for correcting the chemical shift

            # This isn't tested yet because I don't have sequence imports working yet
            # TODO: test this
            self.config.logger.warning(
                (
                    "There were inconsistencies between the provided sequence "
                    "file and the predicted shifts. The following residues had "
                    "inconsistent amino acid types: %s"
                )
                % ", ".join(ambiguous_res_names)
            )

        #        preds.index = preds["Res_name"]
        #        preds.index.name = None
        seq_df.index = seq_df["Res_name"]
        seq_df.index.name = None

        #### Add the chemical shift info back in

        # Make columns for the i-1 predicted shifts of C, CA and CB
        preds_m1 = preds[
            sorted(
                {"C", "CA", "CB", "Res_type", "Res_name"}.intersection(preds.columns)
            )
        ].copy()
        preds_m1.index = preds_m1.index + 1
        preds_m1.columns = preds_m1.columns + "_m1"
        preds = pd.merge(preds, preds_m1, how="left", left_index=True, right_index=True)

        # Make column for the i+1 Res_name
        preds_p1 = preds[["Res_name"]].copy()
        preds_p1.index = preds_p1.index - 1
        preds_p1.columns = ["Res_name_p1"]
        preds = pd.merge(preds, preds_p1, how="left", left_index=True, right_index=True)

        # Set index to Res_name
        preds.index = preds["Res_name"]
        preds.index.name = None

        # Restrict to only certain atom types
        atom_set = {"H", "N", "C", "CA", "CB", "C_m1", "CA_m1", "CB_m1", "HA"}
        preds = preds[
            [
                "Res_name",
                "Res_N",
                "Res_type",
                "Res_name_m1",
                "Res_name_p1",
                "Res_type_m1",
            ]
            + sorted(atom_set.intersection(preds.columns))
        ]

        return preds

    def import_data(
        self,
        obs_file: Path,
        pred_file: Path,
        obs_format: str | None = None,
        pred_format: str | None = None,
        obs_frame: str | None = None,
        pred_frame: str | None = None,
        obs_chain: str = "A",
        pred_chain: str = "A",
        aa_restraints_file: Path | None = None,
        aa_restraints_frame: str | None = None,
    ) -> ShiftData:
        """Import all data and return structured container"""

        # Get appropriate readers
        obs_reader = self._get_reader(obs_file, obs_format)
        pred_reader = self._get_reader(pred_file, pred_format)

        self.config.logger.info(f"Observed shifts reader={obs_reader.format_name}")
        self.config.logger.info(f"Predicted shifts reader={pred_reader.format_name}")

        # Import observed shifts
        if isinstance(obs_reader, NEFReader):
            obs_shifts = obs_reader.read_observed_shifts(obs_file, chain=obs_chain)
        else:
            obs_shifts = obs_reader.read_observed_shifts(obs_file)

        # Import predicted shifts
        if isinstance(pred_reader, NEFReader):
            pred_shifts = pred_reader.read_predicted_shifts(pred_file, chain=pred_chain)
        else:
            pred_shifts = pred_reader.read_predicted_shifts(pred_file)

        obs_shifts = self.clean_obs(obs_shifts)

        # Import sequence
        if self.config.sequence_file is not None:
            sequence = self.import_sequence(self.config.sequence_file)
            pred_shifts = self.clean_preds_with_sequence(pred_shifts, self.config.pred_offset)
            pred_shifts = self.map_preds_to_sequence(sequence, pred_shifts)
        else:
            sequence = None
            pred_shifts = self.clean_preds_no_sequence(pred_shifts, offset=self.config.pred_offset)

        # # Import AA restraints if provided
        # aa_restraints = None
        # if aa_restraints_file:
        #     restraint_reader = self._get_reader(aa_restraints_file)
        #     aa_restraints = restraint_reader.read_aa_restraints(
        #         aa_restraints_file,
        #         frame=aa_restraints_frame
        #     )

        return ShiftData(
            obs_shifts=obs_shifts,
            pred_shifts=pred_shifts,
            sequence=sequence,
            # aa_restraints=aa_restraints,
            metadata={
                "obs_file": str(obs_file),
                "pred_file": str(pred_file),
                "obs_format": obs_reader.format_name,
                "pred_format": pred_reader.format_name,
                "obs_frame": obs_frame,
                "pred_frame": pred_frame,
            },
        )

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
            self.config.logger.warning("The residue range provided is longer than the"+
                                " length of the sequence, so has been truncated")
        elif len(res_N_list) < len(seq_list):
            extra_needed = len(seq_list) - len(res_N_list)
            last_N = res_N_list[-1]
            res_N_list += list(range(last_N, last_N + extra_needed))
            self.config.logger.warning("The residue range provided is shorter than the"+
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

        self.config.logger.info("Finished importing a sequence of length %d" %
                         len(seq_df.index))

        return(seq_df)

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

        preds = preds.iloc[preds_aligned_residues, :]  # Discard predictions that are not in the alignment
        preds.index = seq_df.index[seq_df_aligned_residues]  # Index the predictions with their position in sequence
        preds = preds.rename(columns={"Res_N": "Pred_N", "Res_name": "Pred_name", "Res_type": "Pred_type"})
        aligned_preds = pd.concat((seq_df, preds), axis=1)

        # If glycine HA is missing, replace it with average of HA2 and HA3
        mask = aligned_preds.HA.isna()
        aligned_preds.loc[mask, "HA"] = 0.5 * (aligned_preds.loc[mask, "HA2"] + aligned_preds.loc[mask, "HA3"])

        # Remove atoms that are not used?
        cols_to_keep = ["Seq_N", "Res_N", "Res_type", "Res_name", "Res_name_m1",
                        "Res_name_p1", "Pred_N", "Pred_type", "Pred_name"]
        atom_list = list(self.config.atom_set.intersection(aligned_preds.columns))
        aligned_preds = aligned_preds.loc[:, cols_to_keep + atom_list]

        # Remove predictions where residue type is inconsistent
        inconsistent_res_type = (aligned_preds.Res_type != aligned_preds.Pred_type)
        aligned_preds.loc[inconsistent_res_type, atom_list] = np.nan

        # Add columns for i-1 and i+1 atoms
        aligned_preds.index = aligned_preds.Res_name
        aligned_preds.index.name = None
        preds_m1 = aligned_preds[list({"C", "CA", "CB", "N", "Res_type", "Res_name", "Res_name_p1"}.
                                      intersection(aligned_preds.columns))].copy()
        preds_m1.index = preds_m1.Res_name_p1  # Note _p1, because we need to transfer the chemical shift to the i+1 residue!
        aligned_preds = pd.merge(aligned_preds, preds_m1[["C", "CA", "CB", "N"]], how="left",
                                 left_index=True, right_index=True, suffixes=["", "_m1"])

        preds_p1 = aligned_preds[list({"N", "Res_type", "Res_name", "Res_name_m1"}.
                                      intersection(aligned_preds.columns))].copy()
        preds_p1.index = preds_p1.Res_name_m1  # Note _m1, because we need to transfer the chemical shift to the i-1 residue!
        aligned_preds = pd.merge(aligned_preds, preds_p1[["N"]], how="left",
                                 left_index=True, right_index=True, suffixes=["", "_p1"])

        # Output info to log
        self.config.logger.info("Aligned %d predicted residues to sequence" % ((~aligned_preds.Pred_name.isna()).sum()))
        self.config.logger.info("\n" + str(alignment))

        self.aligned_preds = aligned_preds
        return (aligned_preds)

    def clean_preds_with_sequence(self, preds_long: pd.DataFrame, offset: int = 0):
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
        #### Initial processing and conversion from long to wide
        # Convert from long to wide format
        preds = preds_long.pivot(index="Res_N", columns="Atom_type",
                                 values="Shift")
        preds.index.name = None

        # Add residue type back into wide table
        tmp = preds_long[["Res_N", "Res_type"]]
        tmp = tmp.drop_duplicates(subset="Res_N")
        tmp.index = tmp["Res_N"]
        tmp.index.name = None
        preds = pd.concat([tmp, preds], axis=1)

        # Apply sequence offset and create Res_name column
        preds["Res_N"] = preds["Res_N"] + offset
        preds.insert(1, "Res_name", preds["Res_N"].astype(str) + preds["Res_type"])
        preds["Res_name"] = preds["Res_name"].str.rjust(5)

        return (preds)