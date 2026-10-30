
import numpy as np
import pandas as pd

from refactor.dataclasses import AssignmentResult, ShiftData
from refactor.strategies.base import AssignmentStrategy
from refactor.strategies.direct import DirectStrategy
from refactor.strategies.registry import StrategyRegistry


@StrategyRegistry.register("triplet")
class TripletStrategy(AssignmentStrategy):
    """Iterative triplet-informed assignment using sequential links."""

    def assign(self, data: ShiftData) -> AssignmentResult:
        direct = DirectStrategy(config=self.config)

        # Prepare observations, predictions and individual log scores.
        obs, preds, original_log_prob = direct.prepare(data)

        # Derive initial probabilities from log10 scores.
        # Row scaling avoids overflow from large positive scores.
        prob_matrix = 10.0 ** (
            original_log_prob.sub(
                original_log_prob.max(axis=1), axis=0
            )
        )
        prob_matrix = self._normalise(prob_matrix)

        # Identify compatible sequential spin-system pairs.
        mismatch_result = self.calc_mismatch_matrix(
            obs=obs,
            threshold=self.config.seq_link_threshold,
            logger=self.config.logger,
        )

        if mismatch_result is None:
            result = direct.assign_from_matrix(
                obs, preds, original_log_prob
            )
            result.metadata["strategy"] = "triplet"
            result.metadata["iterations"] = 0
            return result

        mismatch_matrix, _ = mismatch_result

        allowed_links = (
            mismatch_matrix < self.config.seq_link_threshold
        ).astype(int)

        np.fill_diagonal(allowed_links.values, 0)

        # Initial assignment, equivalent to iteration zero.
        result = direct.assign_from_matrix(
            obs, preds, original_log_prob
        )

        iterations = self.config.triplet_prob_iterations

        for iteration in range(iterations):
            # Calculate neighbouring contributions.
            left = allowed_links.T.dot(prob_matrix)
            right = allowed_links.dot(prob_matrix)

            left = left.shift(1, axis=1).fillna(1.0)
            right = right.shift(-1, axis=1).fillna(1.0)

            # Triplet probability matrix.
            prob_matrix = left * prob_matrix * right
            prob_matrix.index.name = "SS_name"
            prob_matrix.columns.name = "Res_name"

            # Normalise as in alex-dev.
            prob_matrix = self._normalise(prob_matrix)

            # Convert to log10 scores for Hungarian matching.
            log_prob = np.log10(
                prob_matrix.clip(lower=1e-200)
            )

            result = direct.assign_from_matrix(
                obs, preds, log_prob
            )

            self.config.logger.info(
                "Completed triplet iteration %d",
                iteration + 1,
            )

        result.metadata["strategy"] = "triplet"
        result.metadata["iterations"] = iterations

        return result

    @staticmethod
    def _normalise(prob_matrix: pd.DataFrame) -> pd.DataFrame:
        """Alternating column/row normalisation."""
        matrix = prob_matrix.copy()

        for _ in range(20):
            col_sums = matrix.sum(axis=0).replace(0, np.nan)
            matrix = matrix.div(col_sums, axis=1)

            row_sums = matrix.sum(axis=1).replace(0, np.nan)
            matrix = matrix.div(row_sums, axis=0)

        return matrix.fillna(0.0)
