from line_profiler import profile

from refactor.data_formats import AssignmentResult, ShiftData
from refactor.strategies.base import AssignmentStrategy
from refactor.strategies.registry import StrategyRegistry
from refactor.strategies.triplet import TripletStrategy


@StrategyRegistry.register('tripletm2')
class TripletStrategyModifiedConfidence2(AssignmentStrategy):
    """
    Iterative triplet-informed assignment using sequential links.

    Runs on a much more refined system for determining confidence than triplet_modified_condifence

    It takes the non diagonal elements of the log probability matrix, finds the maximum of that,
    takes the difference between that and the diagonal element, and uses that as a measure of confidence.
    The larger the difference, the more confident we are in the assignment.

    Currently three cutoffs are used to determine confidence:
    - High confidence: difference > 0.5
    - Medium confidence: 0.1 < difference <= 0.5
    - Unreliable: difference <= 0.1
    """

    @profile
    def assign(self, data: ShiftData) -> AssignmentResult:
        triplet = TripletStrategy(config=self.config)
        result = triplet.assign(data)

        log_probs = result.log_prob_matrix
        matching = result.assignments  # DataFrame with SS_name and Res_name

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
        confidence = {}

        for residue, ss in assigned_ss.items():
            if residue not in log_probs.columns or ss not in log_probs.index:
                confidence[residue] = 'Unreliable'
                continue

            assigned_score = log_probs.at[ss, residue]
            alternatives = log_probs[residue].drop(index=ss, errors='ignore')

            if alternatives.empty:
                confidence[residue] = 'High'  # no alternative to compare
                continue

            difference = assigned_score - alternatives.max()

            if difference > self.config.cutoff_2:
                confidence[residue] = 'High'
            elif difference > self.config.cutoff_1:
                confidence[residue] = 'Medium'
            else:
                confidence[residue] = 'Unreliable'

        result.assign_df['Confidence'] = result.assign_df['Res_name'].map(
            confidence
        )

        return result
