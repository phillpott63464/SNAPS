from pathlib import Path
from contextlib import contextmanager
import sys
import pandas as pd
import numpy as np
from bokeh.embed import json_item
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, LabelSet, Range1d, Span
from bokeh.plotting import figure, output_file, save
from pynmrstar import Saveframe, Loop
from refactor.dataclasses import AssignmentResult
from refactor.dataclasses import ShiftData
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
from refactor.SNAPSconfig import SNAPSConfig, build_config


class FormatOutput:
    headings = """
        Res_name Res_N Res_type SS_name Dummy_res Dummy_SS CA CA_pred HA HA_pred H H_pred CB CB_pred
         C C_pred N N_pred Log_prob Max_mismatch_m1 Max_mismatch_p1 Num_good_links_m1 
    """.split()

    def __init__(self, config: Optional[SNAPSConfig] = None):
        if config is None:
            self.config = build_config()
        else:
            self.config = config

    @staticmethod
    @contextmanager
    def _open_or_stdout(filename, *args, **kwargs):
        if str(filename) == '-':
            yield sys.stdout
        else:
            with open(filename, *args, **kwargs) as f:
                yield f

    def _output(self, result: AssignmentResult, config: SNAPSConfig = None):
        ...

    def output(self, result: AssignmentResult, config: SNAPSConfig) -> None:
        output = self._output(result, config)
        if config.output_file is None:
            raise ValueError(
                'output_file must be configured before writing output'
            )
        with self._open_or_stdout(config.output_file, 'w') as fp:
            print(output, file=fp)

        print('Finished writing results to %s', config.output_file)
        #### Write chemical shift lists
        # Currently unimplemented
        # if spec.shift_output_file is not None:
        #     assigner.output_shiftlist(spec.shift_output_file, spec.shift_output_type,
        #                               confidence_list=spec.shift_output_confidence)

    def plot_graphs(self, assignment_result: AssignmentResult, data: Optional[ShiftData] = None):
        if self.config.hsqc_plot_file is not None:
            self.plot_hsqc(
                assignment_result,
                outfile=self.config.hsqc_plot_file,
            )
            self.config.logger.info('Wrote HSQC plot to %s', self.config.hsqc_plot_file)
        if self.config.strip_plot_file is not None:
            strip_plot = self.plot_strips(
                assignment_result,
                data,
                config=self.config,
                outfile=self.config.strip_plot_file,
            )
            if strip_plot is not None:
                self.config.logger.info('Wrote strip plot to %s', self.config.strip_plot_file)

    @staticmethod
    def _write_plot(plot, outfile, format):
        if outfile is None:
            return
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        if format == 'html':
            output_file(outfile)
            save(plot)
        elif format == 'png':
            from bokeh.io import export_png

            export_png(plot, filename=outfile)
        else:
            raise ValueError(f"Unsupported plot format: {format}")

    @staticmethod
    def _sequential_atoms_present(columns):
        return any(
            atom in columns and f'{atom}_m1' in columns
            for atom in ('C', 'CA', 'CB')
        )

    def plot_strips(
        self,
        result: AssignmentResult,
        data: ShiftData,
        config: SNAPSConfig = None,
        outfile=None,
        format='html',
        return_json=True,
        plot_width=1000,
    ):
        """Create a sequential strip plot from an assignment result."""
        df = result.assign_df
        if not self._sequential_atoms_present(df.columns):
            return None

        sequence = data.pred_shifts[['Res_name', 'Res_N', 'Res_type']].copy()
        df = pd.merge(sequence, df, how='left', on=['Res_name', 'Res_N', 'Res_type'])
        df['Dummy_res'] = df['Dummy_res'].fillna(False)
        df['Dummy_SS'] = df['Dummy_SS'].fillna(True)
        plot_range = figure(x_range=df['Res_name'].tolist()).x_range
        plots = []
        colourmap = {
            'High': 'green',
            'Medium': 'yellowgreen',
            'Low': 'orange',
            'Unreliable': 'red',
            'Undefined': 'grey',
        }

        confidence = figure(
            title=(
                'Confidence plot'
                ' (pan/zoom tools can be accessed at top right;'
                ' mouse over the plot to see observed spin system name)'
            ),
            x_range=plot_range,
            y_range=Range1d(0, 1.5),
            tools='xpan, xwheel_zoom,hover,save,reset',
            tooltips=[('Pred', '@Res_name'), ('Obs', '@SS_name')],
            height=100,
            width=plot_width,
        )
        for level, colour in colourmap.items():
            source = ColumnDataSource(df[df['Confidence'] == level])
            confidence.vbar(
                x='Res_name',
                top=1,
                width=1,
                color=colour,
                legend_label=level,
                source=source,
            )
        confidence.legend.orientation = 'horizontal'
        confidence.legend.location = 'top_center'
        confidence.legend.padding = 0
        confidence.legend.margin = 0
        confidence.xaxis.major_label_orientation = np.pi / 2
        confidence.axis.visible = False
        plots.append(confidence)

        mismatch = figure(
            title='Mismatch plot',
            y_axis_label='Mismatch (ppm)',
            x_range=plot_range,
            tools='xpan, xwheel_zoom,save,reset',
            height=200,
            width=plot_width,
        )
        mismatch.vbar(
            x=df['Res_name'],
            top=df[['Max_mismatch_m1', 'Max_mismatch_p1']].max(axis=1),
            width=1,
        )
        threshold = config.seq_link_threshold if config else 0.45
        mismatch.add_layout(
            Span(
                location=threshold,
                dimension='width',
                line_dash='dashed',
                line_color='red',
            )
        )
        mismatch.xaxis.major_label_orientation = np.pi / 2
        plots.append(mismatch)

        carbons = [
            atom for atom in ('C', 'CA', 'CB')
            if atom in df.columns and f'{atom}_m1' in df.columns
        ]
        for atom in carbons:
            plot = figure(
                y_axis_label=f'{atom} (ppm)',
                x_range=plot_range,
                tools='xpan, xwheel_zoom,save,reset',
                height=200,
                width=plot_width,
            )
            valid = df.loc[
                ~df['Dummy_res'].astype(bool),
                ['Res_N', 'Res_name', atom, f'{atom}_m1'],
            ]
            lines = valid[['Res_name', atom, f'{atom}_m1']].copy()
            lines[f'{atom}_z'] = np.nan
            lines = lines.melt(
                id_vars=['Res_name'],
                value_vars=[atom, f'{atom}_m1', f'{atom}_z'],
                var_name='Atom_type',
                value_name='Shift',
            ).sort_values(['Res_name', 'Atom_type'], ascending=[True, False])
            plot.line(lines['Res_name'], lines['Shift'], line_color='black', line_dash='dashed')
            horizontal = valid[['Res_name', atom, f'{atom}_m1']].copy()
            horizontal[f'{atom}_a'] = np.nan
            horizontal = horizontal.melt(
                id_vars=['Res_name'],
                value_vars=[atom, f'{atom}_m1', f'{atom}_a'],
                var_name='Atom_type',
                value_name='Shift',
            ).sort_values(['Res_name', 'Atom_type'], ascending=[True, False])
            plot.line(horizontal['Res_name'], horizontal['Shift'], line_color='black')
            plot.circle(df['Res_name'], df[atom], fill_color='blue', size=5)
            plot.circle(df['Res_name'], df[f'{atom}_m1'], fill_color='red', size=5)
            values = df[[atom, f'{atom}_m1']].max().max(), df[[atom, f'{atom}_m1']].min().min()
            plot.y_range = Range1d(values[0] + 5, values[1] - 5)
            plot.xaxis.major_label_orientation = np.pi / 2
            plot.xaxis.visible = False
            plots.append(plot)

        plots[-1].xaxis.visible = True
        plot = gridplot(plots, ncols=1)
        self._write_plot(plot, outfile, format)
        return json_item(plot, 'strip_plot') if return_json else plot

    def plot_hsqc(
        self,
        result: AssignmentResult,
        outfile=None,
        format='html',
        return_json=True,
        plot_width=750,
    ):
        """Create an HSQC plot coloured by assignment confidence."""
        assign_df = result.assign_df.copy()
        plot = figure(
            title='HSQC',
            x_axis_label='1H (ppm)',
            y_axis_label='15N (ppm)',
            height=plot_width,
            width=plot_width,
        )
        colourmap = {
            'High': 'green',
            'Medium': 'yellowgreen',
            'Low': 'orange',
            'Unreliable': 'red',
            'Undefined': 'grey',
        }
        for level, colour in colourmap.items():
            points = assign_df[assign_df['Confidence'] == level]
            plot.circle(points['H'], points['N'], color=colour, legend_label=level)
        plot.add_layout(
            LabelSet(
                x='H',
                y='N',
                text='Res_name',
                source=ColumnDataSource(assign_df),
                text_font_size='10pt',
            )
        )
        plot.x_range = Range1d(assign_df['H'].max() + 1, assign_df['H'].min() - 1)
        plot.y_range = Range1d(assign_df['N'].max() + 5, assign_df['N'].min() - 5)
        plot.legend.orientation = 'horizontal'
        self._write_plot(plot, outfile, format)
        return json_item(plot) if return_json else plot



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
            raise ValueError(
                f'Unknown strategy: {name}. Available: {available}'
            )
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


@OutputRegistry.register('nef')
class NEFOutput(FormatOutput):
    def _output(self, result: AssignmentResult, config: SNAPSConfig = None):
        save_frame = Saveframe.from_scratch(
            'nefpls_assignments_snaps', 'nefpls_assignments'
        )

        save_frame.add_tag('category', f'nefpls_assignments_snaps')

        loop = Loop.from_scratch('nefpls_assignments')
        save_frame.add_loop(loop)

        # fragment_id
        tags = 'index chain_code sequence_code residue_name unassigned_sequence_code assigned merit'.split()
        loop.add_tag(
            tags
        )  # snaps_log_prob snaps_max_mismatch_m1 snaps_max_mismatch_p1 snaps_num_good_links_m1

        assign_df = result.assign_df.sort_index()

        min_log_prob = -min(assign_df['Log_prob'])

        nef_out_frame = pd.DataFrame(
            assign_df[['SS_name', 'Res_N', 'Res_name', 'Res_type', 'Log_prob']]
        )

        nef_out_frame = nef_out_frame.rename(
            columns={
                'SS_name': 'unassigned_sequence_code',
                'Res_N': 'sequence_code',
                'Res_type': 'residue_name',
                'Log_prob': 'merit',
            }
        )

        nef_out_frame['assigned'] = ~nef_out_frame['Res_name'].str.startswith(
            'DR'
        )

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

        nef_out_frame_assigned = nef_out_frame_assigned.set_index(
            'sequence_code'
        ).reindex(new_index)

        nef_out_frame_assigned['sequence_code'] = nef_out_frame_assigned.index

        nef_out_frame_assigned = nef_out_frame_assigned.reset_index()
        # nef_out_frame_assigned.index += 1
        #
        # nef_out_frame_assigned['sequence_code'] = nef_out_frame_assigned['sequence_code'].astype(int)

        nef_out_frame_assigned['assigned'] = nef_out_frame['assigned'].replace(
            {'NaN': False}
        )

        nef_out_frame_assigned['merit'] = (
            (min_log_prob - nef_out_frame_assigned['merit']) / min_log_prob
        ) - 1

        nef_out_frame_assigned['residue_name'] = nef_out_frame_assigned[
            'residue_name'
        ].replace(TRANSLATIONS_1_3_PROTEIN)

        nef_out_frame_assigned[
            'unassigned_sequence_code'
        ] = nef_out_frame_assigned['unassigned_sequence_code'].str.rstrip(
            string.ascii_letters
        )
        # nef_out_frame_assigned = nef_out_frame_assigned.replace({float.NaN: '.'})

        nef_out_frame_assigned['chain_code'] = config.obs_chain

        # nef_out_frame_assigned['fragment_id'] = (nef_out_frame_assigned['sequence_code'].diff() > 1).cumsum() + 1

        nef_out_frame_assigned['sequence_code'] = (
            nef_out_frame_assigned['sequence_code'].astype(int).astype('str')
        )

        nef_out_frame_assigned['index'] = (
            nef_out_frame_assigned['index']
            - nef_out_frame_assigned['index'][0]
            + 1
        )
        nef_out_frame_assigned['index'] = (
            nef_out_frame_assigned['index'].astype(int).astype(str)
        )

        nef_out_frame_assigned['unassigned_sequence_code'] = (
            nef_out_frame_assigned['unassigned_sequence_code']
            .astype(str)
            .replace({'nan': '.'})
        )
        nef_out_frame_assigned.drop(columns=['Res_name'], inplace=True)

        nef_out_frame_assigned['residue_name'] = (
            nef_out_frame_assigned['residue_name']
            .astype(str)
            .replace({'nan': '.'})
        )

        nef_out_frame_assigned_bad_residues = nef_out_frame_assigned[
            nef_out_frame_assigned['residue_name'] == '.'
        ]['sequence_code'].astype(int)

        nef_out_frame_unassigned = nef_out_frame_unassigned.drop(
            columns=['Res_name']
        )
        nef_out_frame_unassigned = nef_out_frame_unassigned.reset_index()
        nef_out_frame_unassigned[
            'unassigned_sequence_code'
        ] = nef_out_frame_unassigned['unassigned_sequence_code'].str.rstrip(
            string.ascii_letters
        )
        nef_out_frame_unassigned['sequence_code'] = (
            nef_out_frame_unassigned['sequence_code']
            .astype('str')
            .replace({'nan': '.'})
        )
        nef_out_frame_unassigned['residue_name'] = (
            nef_out_frame_unassigned['residue_name']
            .astype('str')
            .replace({'nan': '.'})
        )
        nef_out_frame_unassigned.index = (
            nef_out_frame_unassigned.index
            + nef_out_frame_assigned.index.max()
            + 1
        )
        nef_out_frame_unassigned['index'] = nef_out_frame_unassigned.index + 1

        nef_out_frame_unassigned['chain_code'] = '.'

        output_frame = pd.concat(
            [nef_out_frame_assigned, nef_out_frame_unassigned], axis=0
        )
        output_frame['merit'] = output_frame['merit'].apply(
            lambda x: f'%0.3f' % x
        )
        output_frame['merit'] = output_frame['merit'].replace({'nan': '0.000'})

        loop.add_data(output_frame.to_dict('records'))
        return str(save_frame)


@OutputRegistry.register('snaps')
class SNAPSOutput(FormatOutput):
    def _output(self, result: AssignmentResult, config: SNAPSConfig = None):
        table = []
        headings = [
            heading
            for heading in self.headings
            if heading in result.assign_df.columns
        ]
        for df_index, df_row in result.assign_df.iterrows():

            table_row = []
            table.append(table_row)
            for heading in headings:
                table_row.append(df_row[heading])
        return tabulate(table, tablefmt='plain', headers=headings)


# Factory function (simplified - just delegates to registry)
def create_output(strategy_name: str, config: dict = None) -> FormatOutput:
    """
    Factory function to create assignment strategies.
    Strategies are auto-registered via @StrategyRegistry.register decorator.
    """
    return OutputRegistry.create(strategy_name, config)
