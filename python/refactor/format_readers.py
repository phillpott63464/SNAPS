from symbol import continue_stmt
from typing import Protocol, Optional
from pathlib import Path
from NEF_reader import (
    read_nef_obs_shifts_from_file_to_pandas,
    read_nef_pred_shifts_from_file_to_pandas,
)
import pandas as pd
from typing import List


class FormatReader:
    ...


class ShiftX2Reader(FormatReader):
    """ShiftX2 format reader"""

    @property
    def format_name(self) -> str:
        return 'shiftx2'

    @property
    def file_extensions(self) -> List[str]:
        return ['.cs', '.txt']

    def can_read(self, file_path: Path) -> bool:
        try:
            with open(file_path) as f:
                header = next((line.strip() for line in f if line.strip()), '')

            columns = set(column.strip() for column in header.split(','))

            required = {
                'NUM',
                'RES',
                'ATOMNAME',
                'SHIFT',
            }

            return required.issubset(columns)

        except (OSError, UnicodeError):
            return False

    def read_observed_shifts(self, file_path: Path) -> Optional[pd.DataFrame]:
        return None   # There are no observed shifts in ShiftX2

    def read_predicted_shifts(self, file_path: Path) -> Optional[pd.DataFrame]:
        preds_long = pd.read_csv(file_path)
        if any(preds_long.columns == 'CHAIN'):
            if len(preds_long['CHAIN'].unique()) > 1:
                print(
                    """Chain identifier dropped - if multiple chains are
                        present in the predictions, they will be merged."""
                )
            preds_long = preds_long.drop('CHAIN', axis=1)
        preds_long = preds_long.reindex(
            columns=['NUM', 'RES', 'ATOMNAME', 'SHIFT']
        )
        preds_long.columns = ['Res_N', 'Res_type', 'Atom_type', 'Shift']

        return preds_long


class SpartaReader(FormatReader):
    """Sparta+ format reader"""

    @property
    def format_name(self) -> str:
        return 'sparta+'

    @property
    def file_extensions(self) -> List[str]:
        return ['.txt']

    def can_read(self, file_path: Path) -> bool:
        try:
            with open(file_path) as f:
                lines = f.readlines()

            vars_lines = [
                line.split()[1:]
                for line in lines
                if line.strip().startswith('VARS ')
            ]

            if len(vars_lines) != 1:
                return False

            columns = set(vars_lines[0])

            required = {
                'RESID',
                'RESNAME',
                'ATOMNAME',
                'SS_SHIFT',
                'SHIFT',
                'RC_SHIFT',
                'HM_SHIFT',
                'EF_SHIFT',
                'SIGMA',
            }

            return required.issubset(columns) and any(
                line.strip().startswith('FORMAT ') for line in lines
            )

        except (OSError, UnicodeError):
            return False

    def read_observed_shifts(self, file_path: Path) -> Optional[pd.DataFrame]:
        return None   # There are no observed shifts in sparta

    def read_predicted_shifts(self, file_path: Path) -> Optional[pd.DataFrame]:
        # Work out where the column names and data are
        with open(file_path, 'r') as f:
            for num, line in enumerate(f, 1):
                if line.find('VARS') > -1:
                    colnames_line = num
                    colnames = line.split()[1:]
                    break

        preds_long = pd.read_table(
            file_path, sep='\s+', names=colnames, skiprows=colnames_line + 1
        )
        preds_long = preds_long.reindex(
            columns=['RESID', 'RESNAME', 'ATOMNAME', 'SHIFT']
        )
        preds_long.columns = ['Res_N', 'Res_type', 'Atom_type', 'Shift']

        # Sparta+ uses HN for backbone amide proton - convert to H
        preds_long.loc[preds_long['Atom_type'] == 'HN', 'Atom_type'] = 'H'

        return preds_long


class NMRPipeReader(FormatReader):
    """nmrpipe format reader"""

    @property
    def format_name(self) -> str:
        return 'nmrpipe'

    @property
    def file_extensions(self) -> List[str]:
        return ['.tab']

    def can_read(self, file_path: Path) -> bool:
        try:
            with open(file_path) as f:
                lines = f.readlines()

            vars_lines = [
                line.split()[1:]
                for line in lines
                if line.strip().startswith('VARS ')
            ]

            if len(vars_lines) != 1:
                return False

            columns = vars_lines[0]

            return set(columns) == {
                'RESID',
                'RESNAME',
                'ATOMNAME',
                'SHIFT',
            } and any(line.strip().startswith('FORMAT ') for line in lines)

        except (OSError, UnicodeError):
            return False

    def read_observed_shifts(self, file_path: Path) -> Optional[pd.DataFrame]:
        # Work out where the column names and data are
        with open(file_path, 'r') as f:
            for num, line in enumerate(f, 1):
                if line.find('VARS') > -1:
                    colnames_line = num

        obs = pd.read_table(
            file_path,
            sep=r'\s+',
            skiprows=colnames_line + 1,
            names=['SS_name', 'Res_type', 'Atom_type', 'Shift'],
        )
        obs = obs.loc[:, ['SS_name', 'Atom_type', 'Shift']]
        obs['SS_name'] = obs['SS_name'].astype(str)
        obs.loc[obs['Atom_type'] == 'HN', 'Atom_type'] = 'H'

        return obs

    def read_predicted_shifts(self, file_path: Path) -> Optional[pd.DataFrame]:
        return None


class CCPNReader(FormatReader):
    """ccpn format reader"""

    @property
    def format_name(self) -> str:
        return 'ccpn'

    @property
    def file_extensions(self) -> List[str]:
        return ['.txt']

    def can_read(self, file_path: Path) -> bool:
        try:
            with open(file_path) as f:
                header = next((line.strip() for line in f if line.strip()), '')

            columns = set(header.split('\t'))

            return {
                'Assign Name',
                'Residue',
                'Shift',
            }.issubset(columns)

        except (OSError, UnicodeError):
            return False

    def read_observed_shifts(self, file_path: Path) -> Optional[pd.DataFrame]:
        obs = pd.read_table(file_path)
        obs = obs.loc[:, ['Residue', 'Assign Name', 'Shift']]
        obs.columns = ['SS_name', 'Atom_type', 'Shift']
        obs['Atom_type'] = obs['Atom_type'].str.upper()

        return obs

    def read_predicted_shifts(self, file_path) -> Optional[pd.DataFrame]:
        return None


class SparkyReader(FormatReader):
    """sparky format reader"""

    @property
    def format_name(self) -> str:
        return 'sparky'

    @property
    def file_extensions(self) -> List[str]:
        return ['.txt']

    def can_read(self, file_path: Path) -> bool:
        try:
            with open(file_path) as f:
                header = next((line.strip() for line in f if line.strip()), '')

            columns = header.split()

            return {
                'Group',
                'Atom',
                'Nuc',
                'Shift',
                'SDev',
                'Assignments',
            }.issubset(columns)

        except (OSError, UnicodeError):
            return False

    def read_observed_shifts(self, file_path: Path) -> Optional[pd.DataFrame]:
        obs = pd.read_table(file_path, sep=r'\s+')
        obs = obs.loc[:, ['Group', 'Atom', 'Shift']]
        obs.columns = ['SS_name', 'Atom_type', 'Shift']
        obs.loc[obs['Atom_type'] == 'HN', 'Atom_type'] = 'H'

        return obs

    def read_predicted_shifts(self, file_path) -> Optional[pd.DataFrame]:
        return None


class SnapsReader(FormatReader):
    """SNAPS format reader"""

    @property
    def format_name(self) -> str:
        return 'snaps'

    @property
    def file_extensions(self) -> List[str]:
        return ['.txt', '.tab']

    def can_read(self, file_path: Path) -> bool:
        try:
            with open(file_path) as f:
                header = next((line.strip() for line in f if line.strip()), '')

            return header.split() == [
                'SS_name',
                'Atom_type',
                'Shift',
            ]

        except (OSError, UnicodeError):
            return False

    def read_observed_shifts(self, file_path: Path) -> Optional[pd.DataFrame]:
        obs = pd.read_table(file_path)
        return obs

    def read_predicted_shifts(self, file_path: Path) -> Optional[pd.DataFrame]:
        return None


class XeasyReader(FormatReader):
    """XEASY format reader"""

    @property
    def format_name(self) -> str:
        return 'xeasy'

    @property
    def file_extensions(self) -> List[str]:
        return ['.prot', '.txt']

    def can_read(self, file_path: Path) -> bool:
        try:
            found_data = False

            with open(file_path) as f:
                for line in f:
                    line = line.strip()

                    if not line or line.startswith('#'):
                        continue

                    fields = line.split()

                    if len(fields) != 5:
                        return False

                    try:
                        index = int(fields[0])
                        shift = float(fields[1])
                        uncertainty = float(fields[2])
                        residue = int(fields[4])
                    except ValueError:
                        return False

                    atom = fields[3]

                    if index < 1 or residue < 1:
                        return False

                    if uncertainty < 0:
                        return False

                    if not atom or not atom[0].isalpha():
                        return False

                    found_data = True

            return found_data

        except (OSError, UnicodeError):
            return False

    def read_observed_shifts(self, file_path: Path) -> Optional[pd.DataFrame]:
        obs = pd.read_table(
            file_path,
            sep=r'\s+',
            header=None,
            na_values='999.000',
            names=['i', 'Shift', 'SD', 'Atom_type', 'SS_name'],
        )
        obs = obs.loc[:, ['SS_name', 'Atom_type', 'Shift']]
        obs['SS_name'] = obs['SS_name'].astype(str)
        obs = obs.dropna(subset=['Shift'])
        obs.loc[obs['Atom_type'] == 'HN', 'Atom_type'] = 'H'

        return obs

    def read_predicted_shifts(self, file_path: Path) -> Optional[pd.DataFrame]:
        return None


class MarsReader(FormatReader):
    """MARS format reader"""

    @property
    def format_name(self) -> str:
        return 'mars'

    @property
    def file_extensions(self) -> List[str]:
        return ['.txt', '.tab']

    def can_read(self, file_path: Path) -> bool:
        try:
            with open(file_path) as f:
                header = next((line.strip() for line in f if line.strip()), '')

            columns = set(header.split())

            required = {
                'H',
                'N',
                'CA',
                'CA-1',
                'CB',
                'CB-1',
                'CO',
                'CO-1',
            }

            return required.issubset(columns)

        except (OSError, UnicodeError):
            return False

    def read_observed_shifts(self, file_path: Path) -> Optional[pd.DataFrame]:
        obs_wide = pd.read_table(file_path, sep=r'\s+', na_values='-')
        obs_wide = obs_wide.rename(
            columns={
                'CO': 'C',
                'CO-1': 'C_m1',
                'CA-1': 'CA_m1',
                'CB-1': 'CB_m1',
            }
        )
        if (
            'HA-1' in obs_wide.keys()
        ):   # Added this if so it wouldn't fail if there was no HA-1
            obs_wide = obs_wide.drop(columns='HA-1')
        obs_wide['SS_name'] = obs_wide.index.astype(str)

        obs = obs_wide.melt(
            id_vars='SS_name',
            value_vars=[
                'H',
                'HA',
                'N',
                'C',
                'CA',
                'CB',
                'C_m1',
                'CA_m1',
                'CB_m1',
            ],
            var_name='Atom_type',
            value_name='Shift',
        )

        return obs

    def read_predicted_shifts(self, file_path: Path) -> Optional[pd.DataFrame]:
        return None


class NEFReader(FormatReader):
    """NEF format reader"""

    @property
    def format_name(self) -> str:
        return 'nef'

    @property
    def file_extensions(self) -> List[str]:
        return ['.nef']

    def can_read(self, file_path: Path) -> bool:
        try:
            with open(file_path) as f:
                content = f.read()

            return (
                'data_' in content
                and 'save_nef_' in content
                and '_nef_chemical_shift.' in content
                and 'loop_' in content
            )

        except (OSError, UnicodeError):
            return False

    def read_observed_shifts(
        self, file_path: Path, chain: str = 'A'
    ) -> Optional[pd.DataFrame]:
        obs = read_nef_obs_shifts_from_file_to_pandas(file_path, chain=chain)
        return obs

    def read_predicted_shifts(
        self, file_path: Path, chain: str = 'A'
    ) -> Optional[pd.DataFrame]:
        preds = read_nef_pred_shifts_from_file_to_pandas(file_path, chain=chain)
        preds = preds[['Res_N', 'Res_type', 'Atom_type', 'Shift']]
        preds['Res_N'] = preds['Res_N'].astype(int)

        return preds
