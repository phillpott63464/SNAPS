import contextlib
import copy
import os
from typing import Optional, Union

import pandas as pd
from tqdm import tqdm

from python.refactor.data_formats import AnalysisResult
from refactor.data_formats import AssignmentResult, ShiftData
from refactor.SNAPSconfig import SNAPSConfig, build_config
from refactor.strategies.triplet import TripletStrategy

# TODO: separate out
# TODO: change these to output as AnalysisResults instead of overwriting the AssignmentResult Confidences
# TODO: make it work with any strategy 

class AnalyzerType:
    def __init__(self, config: Optional[Union[SNAPSConfig, dict]] = None):
        if config is None:
            self.config = build_config()
        elif isinstance(config, dict):
            self.config = build_config(**config)
        else:
            self.config = config

    modifies_assignment = False

    def analyze(
        self, assignment: AssignmentResult, data: ShiftData
    ) -> pd.DataFrame:
        ...


def create_analyzer(
    analyzer_name: str,
    config: Optional[Union[SNAPSConfig, dict]] = None,
) -> AnalyzerType:
    """
    Factory function to create assignment Analyzers.
    Analyzers are auto-registered via @AnalyzerRegistry.register decorator.
    """
    return AnalyzerRegistry.create(analyzer_name, config)


class AnalyzerRegistry:
    """Registry for self-registering assignment Analyzers"""

    _analyzers: dict[str, type[AnalyzerType]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register an analyzer"""

        def decorator(analyzer_class):
            analyzer_class.name = name
            cls._analyzers[name] = analyzer_class
            return analyzer_class

        return decorator

    @classmethod
    def get(cls, name: str) -> type[AnalyzerType]:
        """Get analyzer class by name"""
        if name not in cls._analyzers:
            available = ', '.join(cls._analyzers.keys())
            raise ValueError(
                f'Unknown analyzer: {name}. Available: {available}'
            )
        return cls._analyzers[name]

    @classmethod
    def list_names(cls) -> list[str]:
        """List all registered analyzer names"""
        return sorted(cls._analyzers.keys())

    @classmethod
    def create(
        cls,
        name: str,
        config: Optional[Union[SNAPSConfig, dict]] = None,
    ) -> AnalyzerType:
        """Factory method to create an analyzer instance"""
        analyzer_class = cls.get(name)
        return analyzer_class(config)


class AnalysisPipeline:
    def __init__(
        self,
        config: Optional[SNAPSConfig] = None,
        analyzers: list[str] = None,
    ):
        if config is None:
            self.config = build_config()
        else:
            self.config = config

        self.analyzers = [
            create_analyzer(name, self.config) for name in analyzers
        ]

    def run(
        self, assignment: AssignmentResult, data: ShiftData
    ) -> AnalysisResult:
        # If multiple analyzers modify the assignment, reject the inputs
        if (
            sum(analyzer.modifies_assignment for analyzer in self.analyzers)
            > 1
        ):
            raise ValueError(
                'Multiple analyzers modify the assignment. Only one is allowed.'
            )
        if not self.analyzers:
            raise ValueError('No analyzers provided.')
        for analyzer in self.analyzers:
            if analyzer.modifies_assignment:
                assignment = analyzer.analyze(assignment, data)
            else:
                analyzer.analyze(assignment, data)
        return assignment


@AnalyzerRegistry.register('leave_one_out')
class LeaveOneOutAnalyzer(AnalyzerType):
    modifies_assignment = False

    def analyze(
        self, assignment: AssignmentResult, data: ShiftData
    ) -> pd.DataFrame:
        """
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

        Then create Confidences:
        - Both are unique, High
        - One is unique, one is not, Medium
        - Both are not unique, Unreliable
        """
        Confidence = {}
        Confidence['residues'] = {}
        Confidence['spin_systems'] = {}
        triplet = TripletStrategy(config=self.config)

        rows = list(assignment.assignments.iterrows())

        with tqdm(total=2 * len(rows), desc='Calculating Confidence') as pbar:
            for idx, row in rows:
                res_name = row['Res_name']
                old_ss = row['SS_name']

                if res_name.startswith(self.config.dummy_predicted_prefix):
                    Confidence['residues'][idx] = False
                    Confidence['spin_systems'][idx] = False
                    pbar.update(2)
                    continue

                if old_ss not in data.obs_shifts['SS_name'].values:
                    # Synthetic preprocessing row, such as a proline dummy.
                    Confidence['residues'][idx] = True
                    Confidence['spin_systems'][idx] = True
                    pbar.update(2)
                    continue

                # Residue Confidence: remove its observed spin system and reassign.
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
                Confidence['residues'][idx] = new_ss.split('_')[0] == 'DSS'
                pbar.update(1)

                # Spin-system Confidence: blank the predicted residue and reassign.
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
                Confidence['spin_systems'][idx] = (
                    new_residue.split('_')[0] == 'DR'
                )
                pbar.update(1)

        # Replace initial_result.assign_df with a new DataFrame that includes the Confidence information as row Confidence
        # There is a Confidence row already, so remove it and replace it with the new one
        # if both are unique, High
        # if one is unique, one is not, Medium
        # if both are not unique, Unreliable
        res_unique = {
            r['Res_name']: Confidence['residues'][i]
            for i, r in assignment.assignments.iterrows()
        }
        ss_unique = {
            r['Res_name']: Confidence['spin_systems'][i]
            for i, r in assignment.assignments.iterrows()
        }

        df = pd.DataFrame(
            {
                'Res_name': assignment.assignments['Res_name'].values,
                'SS_name': assignment.assignments['SS_name'].values,
            }
        )
        df['Confidence'] = [
            'High'
            if res_unique[r] and ss_unique[r]
            else 'Medium'
            if res_unique[r] or ss_unique[r]
            else 'Unreliable'
            for r in df['Res_name']
        ]

        return df


@AnalyzerRegistry.register('non_diagonal')
class NonDiagonalAnalyze(AnalyzerType):
    modifies_assignment = True
    """"
    Runs on a much more refined system for determining Confidence than triplet_modified_condifence

    It takes the non diagonal elements of the log probability matrix, finds the maximum of that,
    takes the difference between that and the diagonal element, and uses that as a measure of Confidence.
    The larger the difference, the more confident we are in the assignment.

    Currently three cutoffs are used to determine Confidence:
    - High Confidence: difference > 0.5
    - Medium Confidence: 0.1 < difference <= 0.5
    - Unreliable: difference <= 0.1
    """

    def analyze(
        self, assignment: AssignmentResult, data: ShiftData
    ) -> pd.DataFrame:
        log_probs = assignment.log_prob_matrix
        matching = (
            assignment.assignments
        )  # DataFrame with SS_name and Res_name

        # For each residue, find the Highest-scoring row other than its assigned row.
        assigned_ss = matching.set_index('Res_name')['SS_name']

        second_best = {}
        for residue, ss in assigned_ss.items():
            alternatives = log_probs[residue].drop(index=ss, errors='ignore')
            if not alternatives.empty:
                second_best[residue] = {
                    'SS_name': alternatives.idxmax(),
                    'score': alternatives.max(),
                }

        # For each residue, find the difference between the assigned score and the second-best score.
        Confidence = {}

        for residue, ss in assigned_ss.items():
            if residue not in log_probs.columns or ss not in log_probs.index:
                Confidence[residue] = 'Unreliable'
                continue

            assigned_score = log_probs.at[ss, residue]
            alternatives = log_probs[residue].drop(index=ss, errors='ignore')

            if alternatives.empty:
                Confidence[residue] = 'High'  # no alternative to compare
                continue

            difference = assigned_score - alternatives.max()

            if difference > self.config.cutoff_2:
                Confidence[residue] = 'High'
            elif difference > self.config.cutoff_1:
                Confidence[residue] = 'Medium'
            else:
                Confidence[residue] = 'Unreliable'

        df = pd.DataFrame(
            {
                'Res_name': assignment.assignments['Res_name'].values,
                'SS_name': assignment.assignments['SS_name'].values,
                'Confidence': assignment.assignments['Res_name']
                .map(Confidence)
                .values,
            }
        )

        return df

        return assignment


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


from __future__ import annotations
