from pathlib import Path
from contextlib import contextmanager
import sys
import pandas as pd
from pynmrstar import Saveframe, Loop
from refactor.dataclasses import AssignmentResult, OutputSpec
from numpy import arange
import string
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from numpy import arange
from pynmrstar import Saveframe, Loop
from tabulate import tabulate

from SNAPS_importer import SNAPS_importer
from SNAPS_assigner import SNAPS_assigner

# TODO remove these from NEF importer
from NEF_reader import _split_path_and_frame, TRANSLATIONS_1_3_PROTEIN
import logging
from typing import Optional, List, Type, Dict
from refactor.dataclasses import OutputSpec
from refactor.SNAPSconfig import SNAPSConfig, build_config

class FormatOutput():
    headings = '''
        Res_name Res_N Res_type SS_name Dummy_res Dummy_SS CA CA_pred HA HA_pred H H_pred CB CB_pred
         C C_pred N N_pred Log_prob Max_mismatch_m1 Max_mismatch_p1 Num_good_links_m1 
    '''.split()

    def __init__(self, config: Optional[SNAPSConfig] = None):
        if config is None:
            self.config = build_config()
        else:
            self.config = config

    @staticmethod
    @contextmanager
    def _open_or_stdout(filename, *args, **kwargs):
        if filename == '-':
            yield sys.stdout
        else:
            with open(filename, *args, **kwargs) as f:
                yield f

    def _output(self, result: AssignmentResult, obs_chain: str = 'A'):
        ...

    def output(self, result: AssignmentResult, spec: OutputSpec):
        output = self._output(result)
        with self._open_or_stdout(spec.output_file, 'w') as fp:
            print(output, file=fp)

        print("Finished writing results to %s", spec.output_file)
        #### Write chemical shift lists
        # Currently unimplemented
        # if spec.shift_output_file is not None:
        #     assigner.output_shiftlist(spec.shift_output_file, spec.shift_output_type,
        #                               confidence_list=spec.shift_output_confidence)

class OutputRegistry:
    _outputs: Dict[str, Type[FormatOutput]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a strategy"""

        def decorator(strategy_class):
            strategy_class.name = name
            cls._outputs[name] = strategy_class
            return strategy_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[FormatOutput]:
        """Get strategy class by name"""
        if name not in cls._outputs:
            available = ', '.join(cls._outputs.keys())
            raise ValueError(f"Unknown strategy: {name}. Available: {available}")
        return cls._outputs[name]

    @classmethod
    def list_names(cls) -> List[str]:
        """List all registered strategy names"""
        return sorted(cls._outputs.keys())

    @classmethod
    def create(cls, name: str, config: dict) -> FormatOutput:
        """Factory method to create a strategy instance"""
        output = cls.get(name)
        return output(config)

@OutputRegistry.register('NEF')
class NEFOutput(FormatOutput):
    def _output(self, result: AssignmentResult, spec: OutputSpec):
        save_frame = Saveframe.from_scratch('nefpls_assignments_snaps', 'nefpls_assignments')

        save_frame.add_tag('category', f'nefpls_assignments_snaps')

        loop = Loop.from_scratch('nefpls_assignments')
        save_frame.add_loop(loop)

        # fragment_id
        tags = 'index chain_code sequence_code residue_name unassigned_sequence_code assigned merit'.split()
        loop.add_tag(tags)  # snaps_log_prob snaps_max_mismatch_m1 snaps_max_mismatch_p1 snaps_num_good_links_m1

        assign_df = result.assign_df.sort_index()

        min_log_prob = -min(assign_df['Log_prob'])

        nef_out_frame = pd.DataFrame(assign_df[['SS_name', 'Res_N', 'Res_name', 'Res_type', 'Log_prob']])

        nef_out_frame = nef_out_frame.rename(columns={
            'SS_name': 'unassigned_sequence_code',
            'Res_N': 'sequence_code',
            'Res_type': 'residue_name',
            'Log_prob': 'merit'
        })

        nef_out_frame['assigned'] = ~nef_out_frame['Res_name'].str.startswith('DR')

        nef_out_frame = nef_out_frame.sort_values('sequence_code')

        # min_residue_sequence = sequence['Res_N'].min()
        # max_residue_sequence = sequence['Res_N'].max()

        min_residue = nef_out_frame['sequence_code'].min()
        max_residue = nef_out_frame['sequence_code'].max()

        # min_residue = min_residue_sequence if min_residue_sequence < min_residue else min_residue
        # max_residue = max_residue_sequence if max_residue_sequence > max_residue else max_residue

        new_index = pd.Index(arange(min_residue, max_residue + 1))
        nef_out_frame_assigned = nef_out_frame[nef_out_frame['assigned']]
        nef_out_frame_unassigned = nef_out_frame[~nef_out_frame['assigned']]

        nef_out_frame_assigned = nef_out_frame_assigned.set_index('sequence_code').reindex(new_index)

        nef_out_frame_assigned['sequence_code'] = nef_out_frame_assigned.index

        nef_out_frame_assigned = nef_out_frame_assigned.reset_index()
        # nef_out_frame_assigned.index += 1
        #
        # nef_out_frame_assigned['sequence_code'] = nef_out_frame_assigned['sequence_code'].astype(int)

        nef_out_frame_assigned['assigned'] = nef_out_frame['assigned'].replace({'NaN': False})

        nef_out_frame_assigned['merit'] = ((min_log_prob - nef_out_frame_assigned['merit']) / min_log_prob) - 1

        nef_out_frame_assigned['residue_name'] = nef_out_frame_assigned['residue_name'].replace(
            TRANSLATIONS_1_3_PROTEIN)

        nef_out_frame_assigned['unassigned_sequence_code'] = nef_out_frame_assigned[
            'unassigned_sequence_code'].str.rstrip(string.ascii_letters)
        # nef_out_frame_assigned = nef_out_frame_assigned.replace({float.NaN: '.'})

        nef_out_frame_assigned['chain_code'] = spec.obs_chain

        # nef_out_frame_assigned['fragment_id'] = (nef_out_frame_assigned['sequence_code'].diff() > 1).cumsum() + 1

        nef_out_frame_assigned['sequence_code'] = nef_out_frame_assigned['sequence_code'].astype(int).astype('str')

        nef_out_frame_assigned['index'] = nef_out_frame_assigned['index'] - nef_out_frame_assigned['index'][0] + 1
        nef_out_frame_assigned['index'] = nef_out_frame_assigned['index'].astype(int).astype(str)

        nef_out_frame_assigned['unassigned_sequence_code'] = nef_out_frame_assigned['unassigned_sequence_code'].astype(
            str).replace({'nan': '.'})
        nef_out_frame_assigned.drop(columns=['Res_name'], inplace=True)

        nef_out_frame_assigned['residue_name'] = nef_out_frame_assigned['residue_name'].astype(str).replace(
            {'nan': '.'})

        nef_out_frame_assigned_bad_residues = nef_out_frame_assigned[nef_out_frame_assigned['residue_name'] == '.'][
            'sequence_code'].astype(int)

        nef_out_frame_unassigned = nef_out_frame_unassigned.drop(columns=['Res_name'])
        nef_out_frame_unassigned = nef_out_frame_unassigned.reset_index()
        nef_out_frame_unassigned['unassigned_sequence_code'] = nef_out_frame_unassigned[
            'unassigned_sequence_code'].str.rstrip(string.ascii_letters)
        nef_out_frame_unassigned['sequence_code'] = nef_out_frame_unassigned['sequence_code'].astype('str').replace(
            {'nan': '.'})
        nef_out_frame_unassigned['residue_name'] = nef_out_frame_unassigned['residue_name'].astype('str').replace(
            {'nan': '.'})
        nef_out_frame_unassigned.index = nef_out_frame_unassigned.index + nef_out_frame_assigned.index.max() + 1
        nef_out_frame_unassigned['index'] = nef_out_frame_unassigned.index + 1

        nef_out_frame_unassigned['chain_code'] = '.'

        output_frame = pd.concat([nef_out_frame_assigned, nef_out_frame_unassigned], axis=0)
        output_frame['merit'] = output_frame['merit'].apply(lambda x: f'%0.3f' % x)
        output_frame['merit'] = output_frame['merit'].replace({'nan': '0.000'})

        loop.add_data(output_frame.to_dict('records'))
        return str(save_frame)

@OutputRegistry.register('SNAPS')
class SNAPSOutput(FormatOutput):
    def _output(self, result: AssignmentResult, obs_chain: str = 'A'):
        table = []
        for df_index, df_row in result.assign_df.iterrows():

            table_row = []
            table.append(table_row)
            for heading in self.headings:
                table_row.append(df_row[heading])
        return tabulate(table, tablefmt='plain', headers=self.headings)

# Factory function (simplified - just delegates to registry)
def create_output(strategy_name: str, config: dict = None) -> FormatOutput:
    """
    Factory function to create assignment strategies.
    Strategies are auto-registered via @StrategyRegistry.register decorator.
    """
    return OutputRegistry.create(strategy_name, config)