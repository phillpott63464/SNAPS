import contextlib
import copy
import os

import numpy as np
from line_profiler import profile
from tqdm import tqdm

from refactor.data_formats import AssignmentResult, ShiftData
from refactor.strategies.base import AssignmentStrategy
from refactor.strategies.registry import StrategyRegistry
from refactor.strategies.triplet import TripletStrategy


@StrategyRegistry.register('tripletm')
class TripletStrategyModifiedConfidence(AssignmentStrategy):
    """
    Iterative triplet-informed assignment using sequential links.

    Runs on a fairly simple system:
    - For every residue:
      - remove the assigned spin system, replace with a dummy (done implicitly in preprocess_obs_preds)
      - reassign
      - Has the residue been assigned to a dummy spin system, or a real spin system?
        - If a real spin system, then there are multiple spin systems that could be assigned to this residue, and the assignment is not unique.
          - Potential to take the difference in probability and use a threshold, or just discard as not unique.
        - If a dummy spin system, then the assignment is unique.

    And then do the flip-flop:
    - For every spin system:
        - remove the assigned residue, replace with a dummy
        - reassign
        - Has the spin system been assigned to a dummy residue, or a real residue?
            - If a real residue, then there are multiple residues that could be assigned to this spin system, and the assignment is not unique.
        - If a dummy residue, then the assignment is unique.

    Then create confidences:
    - Both are unique, high
    - One is unique, one is not, medium
    - Both are not unique, unreliable
    """

    @profile
    def assign(self, data: ShiftData) -> AssignmentResult:
        triplet = TripletStrategy(config=self.config)
        initial_result = triplet.assign(data)
        confidence = {}
        confidence['residues'] = {}
        confidence['spin_systems'] = {}

        with tqdm(
            total=len(initial_result.assignments) * 2,
            desc='Calculating confidence',
        ) as pbar:
            for row in initial_result.assignments.iterrows():

                obs = data.obs_shifts.copy()
                obs = obs[obs['SS_name'] != row[1]['SS_name']]
                assert len(obs) == len(data.obs_shifts) - 1
                temp_data = copy.deepcopy(data)
                temp_data.obs_shifts = obs
                with open(os.devnull, 'w') as f:  # noqa: SIM117
                    with contextlib.redirect_stdout(
                        f
                    ), contextlib.redirect_stderr(f):
                        output = triplet.assign(temp_data)
                new_ss = output.assignments.set_index('Res_name').loc[
                    row[1]['Res_name'], 'SS_name'
                ]
                old_ss = row[1]['SS_name']
                print(f'Residue {row[0]}: old SS {old_ss}, new SS {new_ss}')
                if new_ss.split('_')[0] == 'DSS':
                    confidence['residues'][row[0]] = True
                else:
                    confidence['residues'][row[0]] = False
                pbar.update(1)

            for row in initial_result.assignments.iterrows():
                preds = data.pred_shifts.copy()
                # Replace the assigned residue with a dummy, keeping the same position
                # This is important because otherwise everything will explode because it tries to assign i+1 and i-1 as neighbours, which is an incorrect assumption
                preds = self.blank_residue(
                    data.pred_shifts, row[1]['Res_name']
                )
                assert len(preds) == len(data.pred_shifts)
                temp_data = copy.deepcopy(data)
                temp_data.pred_shifts = preds
                with open(os.devnull, 'w') as f:  # noqa: SIM117
                    with contextlib.redirect_stdout(
                        f
                    ), contextlib.redirect_stderr(f):
                        output = triplet.assign(temp_data)
                old_ss = row[1]['SS_name']
                new_residue = output.assignments.set_index('SS_name').loc[
                    old_ss, 'Res_name'
                ]
                print(
                    f'Spin system {row[0]}: old residue {row[1]["Res_name"]}, new residue {new_residue}'
                )
                confidence['spin_systems'][row[0]] = (
                    new_residue.split('_')[0] == 'DR'
                )
                pbar.update(1)

        # Replace initial_result.assign_df with a new DataFrame that includes the confidence information as row Confidence
        # There is a Confidence row already, so remove it and replace it with the new one
        # if both are unique, high
        # if one is unique, one is not, medium
        # if both are not unique, unreliable
        res_unique = {
            r['Res_name']: confidence['residues'][i]
            for i, r in initial_result.assignments.iterrows()
        }
        ss_unique = {
            r['Res_name']: confidence['spin_systems'][i]
            for i, r in initial_result.assignments.iterrows()
        }

        df = initial_result.assign_df.drop('Confidence', axis=1)
        df['Confidence'] = [
            'high'
            if res_unique[r] and ss_unique[r]
            else 'medium'
            if res_unique[r] or ss_unique[r]
            else 'unreliable'
            for r in df['Res_name']
        ]
        initial_result.assign_df = df
        return initial_result

    def blank_residue(self, preds, res_name):
        """Turn one predicted residue into a dummy, keeping its position."""
        preds = preds.copy()
        dummy = self.config.dummy_predicted_prefix + '1'   # -> 'DR_1'
        keep = {'Res_N', 'Res_name', 'Res_name_m1', 'Res_name_p1'}

        mask = preds['Res_name'] == res_name
        preds.loc[mask, [c for c in preds.columns if c not in keep]] = np.nan
        preds.loc[mask, 'Res_name'] = dummy

        # Neighbours must point at the dummy, or preprocessing will NaN the
        # reference (it drops m1/p1 names that aren't in Res_name).
        for c in ('Res_name_m1', 'Res_name_p1'):
            preds.loc[preds[c] == res_name, c] = dummy

        return preds.rename(
            index={res_name: dummy}
        )   # no-op if the index isn't Res_name
