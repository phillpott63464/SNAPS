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
    - Both are unique, High
    - One is unique, one is not, Medium
    - Both are not unique, Unreliable
    """

    @profile
    def assign(self, data: ShiftData) -> AssignmentResult:
        triplet = TripletStrategy(config=self.config)
        initial_result = triplet.assign(data)
        confidence = {}
        confidence['residues'] = {}
        confidence['spin_systems'] = {}

        rows = list(initial_result.assignments.iterrows())

        with tqdm(total=2 * len(rows), desc='Calculating confidence') as pbar:
            for idx, row in rows:
                res_name = row['Res_name']
                old_ss = row['SS_name']

                if res_name.startswith(self.config.dummy_predicted_prefix):
                    confidence['residues'][idx] = False
                    confidence['spin_systems'][idx] = False
                    pbar.update(2)
                    continue

                if old_ss not in data.obs_shifts['SS_name'].values:
                    # Synthetic preprocessing row, such as a proline dummy.
                    confidence['residues'][idx] = True
                    confidence['spin_systems'][idx] = True
                    pbar.update(2)
                    continue

                # Residue confidence: remove its observed spin system and reassign.
                obs = data.obs_shifts[data.obs_shifts['SS_name'] != old_ss]
                assert len(obs) == len(data.obs_shifts) - 1

                temp_data = copy.deepcopy(data)
                temp_data.obs_shifts = obs
                with open(os.devnull, 'w') as f:
                    with contextlib.redirect_stdout(
                        f
                    ), contextlib.redirect_stderr(f):
                        output = triplet.assign(temp_data)

                new_ss = output.assignments.set_index('Res_name').loc[
                    res_name, 'SS_name'
                ]
                confidence['residues'][idx] = new_ss.split('_')[0] == 'DSS'
                pbar.update(1)

                # Spin-system confidence: blank the predicted residue and reassign.
                # preds = self.blank_residue(data.pred_shifts, res_name)
                # assert len(preds) == len(data.pred_shifts)
                preds = data.pred_shifts[
                    data.pred_shifts['Res_name'] != res_name
                ]
                assert len(preds) == len(data.pred_shifts) - 1

                temp_data = copy.deepcopy(data)
                temp_data.pred_shifts = preds
                with open(os.devnull, 'w') as f:
                    with contextlib.redirect_stdout(
                        f
                    ), contextlib.redirect_stderr(f):
                        output = triplet.assign(temp_data)

                new_residue = output.assignments.set_index('SS_name').loc[
                    old_ss, 'Res_name'
                ]
                confidence['spin_systems'][idx] = (
                    new_residue.split('_')[0] == 'DR'
                )
                pbar.update(1)

        # Replace initial_result.assign_df with a new DataFrame that includes the confidence information as row Confidence
        # There is a Confidence row already, so remove it and replace it with the new one
        # if both are unique, High
        # if one is unique, one is not, Medium
        # if both are not unique, Unreliable
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
            'High'
            if res_unique[r] and ss_unique[r]
            else 'Medium'
            if res_unique[r] or ss_unique[r]
            else 'Unreliable'
            for r in df['Res_name']
        ]
        initial_result.assign_df = df
        return initial_result

    def blank_residue(self, preds, res_name):
        """Turn one predicted residue into a dummy, keeping its position."""
        preds = preds.copy()
        dummy = (
            self.config.dummy_predicted_prefix + 'removed'
        )   # -> 'DR_removed', avoids collision with other dummy residues that are added by preprocessing in triplet.assign
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
