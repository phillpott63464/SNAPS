from typing import List, Optional
from pathlib import Path
from modules.format_readers import (
    ShiftX2Reader,
    SpartaReader,
    NEFReader,
    NMRPipeReader,
    CCPNReader,
    SparkyReader,
    SnapsReader,
    XeasyReader,
    MarsReader,
    FormatReader
)
from modules.dataclasses import ShiftData
import pandas as pd
import numpy as np

class SNAPSImporter:
    """Orchestrates format readers to import all required data"""

    def __init__(self, readers: Optional[List[FormatReader]] = None):
        self.readers = self._default_readers()
        self._reader_by_format = {r.format_name: r for r in self.readers}
        self._reader_by_ext = {}
        for reader in self.readers:
            for ext in reader.file_extensions:
                self._reader_by_ext.setdefault(ext, []).append(reader)

    def _default_readers(self) -> List[FormatReader]:
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
            self,
            file_path: Path,
            format_hint: Optional[str] = None
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
            print(
                f"Reader: {type(reader).__name__}, "
                f"File: {file_path.name}, "
                f"Can read: {result}"
            )
            if result:
                return reader

        raise ValueError(f"No reader found for {file_path}")

    def clean_obs(self, obs_shift):
        obs_shift = obs_shift.loc[obs_shift["Atom_type"].isin(["H", "HA", "N", "C", "CA", "CB",
                                                               "C_m1", "CA_m1", "CB_m1"]), :]

        # Convert from long to wide
        obs_shift = obs_shift.pivot(index="SS_name", columns="Atom_type", values="Shift")
        obs_shift.insert(0, "SS_name", obs_shift.index.values)

        # the index name shouldn't match a column name!
        obs_shift.index.name = None

        return obs_shift

    def clean_preds(self, preds_long: pd.DataFrame, offset: int = 0):
        #### Initial processing and conversion from long to wide
        # Add sequence number offset and create residue names
        preds_long['Res_N'] = preds_long['Res_N'] + offset
        preds_long.insert(
            1,
            'Res_name',
            (preds_long['Res_N'].astype(str) + preds_long['Res_type']),
        )
        # Left pad with spaces to a constant length (helps with sorting)
        preds_long['Res_name'] = preds_long['Res_name'].str.rjust(5)

        # Convert from long to wide format
        preds = preds_long.pivot(
            index='Res_N', columns='Atom_type', values='Shift'
        )
        preds.index.name = None

        # Add the other residue name and type back in
        tmp = preds_long[['Res_N', 'Res_type', 'Res_name']]
        tmp = tmp.drop_duplicates(subset='Res_name')
        tmp.index = tmp['Res_N']
        tmp.index.name = None
        preds = pd.concat([tmp, preds], axis=1)

        #### Make consistent with seq_df (and create if it doesn't already exist)
        # TODO: Maybe this should be split off into a separate function?
        # If seq_df is missing, create it based on preds
        seq_df = preds.copy()[['Res_N', 'Res_type', 'Res_name']]

        # If there are any missing residue numbers, create them
        min_N = seq_df['Res_N'].min()
        max_N = seq_df['Res_N'].max()
        missing_residue_numbers = set(range(min_N, max_N + 1)).difference(
            seq_df['Res_N']
        )
        if len(missing_residue_numbers) > 0:
            tmp = pd.DataFrame(
                {
                    'Res_N': list(missing_residue_numbers),
                    'Res_type': 'X',
                    'Res_name': np.NaN,
                }
            )
            tmp['Res_name'] = tmp['Res_N'].astype(str) + tmp['Res_type']
            tmp['Res_name'] = tmp['Res_name'].str.rjust(5)
            tmp.index = tmp['Res_N']
            tmp.index.name = None
            # seq_df = seq_df.append(tmp).sort_index()
            seq_df = pd.concat([seq_df, tmp]).sort_index()

        # Add/delete residues from preds so it matches seq_df
        # Only keep predictions that are in seq_df
        tmp = len(preds.index)
        preds = preds[preds['Res_N'].isin(seq_df['Res_N'])]
        tmp2 = tmp - len(preds.index)

        # Add predictions for any residue number that is in seq_df but not preds
        tmp = pd.DataFrame(
            data=seq_df[~seq_df['Res_N'].isin(preds['Res_N'])],
            columns=preds.columns,
        )
        # preds = preds.append(tmp)
        preds = pd.concat([preds, tmp])

        # If Res_name is inconsistent between seq_df and preds, make a compromise
        preds.index = preds['Res_N']
        preds = preds.sort_index()
        seq_df.index = seq_df['Res_N']
        seq_df = seq_df.sort_index()
        mask = preds['Res_name'] != seq_df['Res_name']

        if any(mask):
            ambiguous_res_names = (
                    seq_df.loc[mask, 'Res_name']
                    + '('
                    + preds.loc[mask, 'Res_type']
                    + '?)'
            )
            preds.loc[mask, 'Res_name'] = ambiguous_res_names
            seq_df.loc[mask, 'Res_name'] = ambiguous_res_names
            # Note we don't update the Res_type in preds, because this is potentially
            # used for correcting the chemical shift
            logger.warning(
                (
                    'There were inconsistencies between the provided sequence '
                    'file and the predicted shifts. The following residues had '
                    'inconsistent amino acid types: %s'
                )
                % ', '.join(ambiguous_res_names)
            )

        #        preds.index = preds["Res_name"]
        #        preds.index.name = None
        seq_df.index = seq_df['Res_name']
        seq_df.index.name = None

        #### Add the chemical shift info back in

        # Make columns for the i-1 predicted shifts of C, CA and CB
        preds_m1 = preds[
            list(
                {'C', 'CA', 'CB', 'Res_type', 'Res_name'}.intersection(
                    preds.columns
                )
            )
        ].copy()
        preds_m1.index = preds_m1.index + 1
        preds_m1.columns = preds_m1.columns + '_m1'
        preds = pd.merge(
            preds, preds_m1, how='left', left_index=True, right_index=True
        )

        # Make column for the i+1 Res_name
        preds_p1 = preds[['Res_name']].copy()
        preds_p1.index = preds_p1.index - 1
        preds_p1.columns = ['Res_name_p1']
        preds = pd.merge(
            preds, preds_p1, how='left', left_index=True, right_index=True
        )

        # Set index to Res_name
        preds.index = preds['Res_name']
        preds.index.name = None

        # Restrict to only certain atom types
        atom_set = {'H', 'N', 'C', 'CA', 'CB', 'C_m1', 'CA_m1', 'CB_m1', 'HA'}
        preds = preds[
            [
                'Res_name',
                'Res_N',
                'Res_type',
                'Res_name_m1',
                'Res_name_p1',
                'Res_type_m1',
            ]
            + list(atom_set.intersection(preds.columns))
            ]

        return preds

    def import_data(
            self,
            obs_file: Path,
            pred_file: Path,
            obs_format: Optional[str] = None,
            pred_format: Optional[str] = None,
            obs_frame: Optional[str] = None,
            pred_frame: Optional[str] = None,
            obs_chain: str = 'A',
            pred_chain: str = 'A',
            pred_offset: int = 0,
            aa_restraints_file: Optional[Path] = None,
            aa_restraints_frame: Optional[str] = None,
    ) -> ShiftData:
        """Import all data and return structured container"""

        # Get appropriate readers
        obs_reader = self._get_reader(obs_file, obs_format)
        print(obs_reader)
        pred_reader = self._get_reader(pred_file, pred_format)
        print(pred_reader)

        # Import observed shifts
        obs_shifts = obs_reader.read_observed_shifts(
            obs_file,
            # frame=obs_frame, # What's this mean to be?
            # chain=obs_chain
        )

        # Import predicted shifts
        pred_shifts = pred_reader.read_predicted_shifts(
            pred_file,
            # frame=pred_frame,
            # chain=pred_chain,
            # offset=pred_offset
        )

        pred_shifts = self.clean_preds(pred_shifts)
        obs_shifts = self.clean_obs(obs_shifts)

        # Import sequence
        # sequence = pred_reader.read_sequence(pred_file)

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
            # sequence=sequence,
            # aa_restraints=aa_restraints,
            metadata={
                'obs_file': str(obs_file),
                'pred_file': str(pred_file),
                'obs_format': obs_reader.format_name,
                'pred_format': pred_reader.format_name,
                'obs_frame': obs_frame,
                'pred_frame': pred_frame,
            }
        )
