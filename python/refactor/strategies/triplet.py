
import numpy as np
import pandas as pd

from refactor.dataclasses import AssignmentResult, ShiftData
from refactor.strategies.base import AssignmentStrategy
from refactor.strategies.registry import StrategyRegistry
from refactor.strategies.utils.output import make_assign_df


@StrategyRegistry.register("triplet")
class TripletStrategy(AssignmentStrategy):
    """Iterative triplet-informed assignment using sequential links."""

    def assign(self, data: ShiftData) -> AssignmentResult:
        # Prepare observations and predictions directly.
        obs, preds = self.preprocess_obs_preds(
            data.obs_shifts,
            data.pred_shifts,
        )

        original_log_prob = self.calc_log_prob_matrix(
            obs=obs,
            preds=preds,
        )

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
            return self._assign_from_matrix(
                obs=obs,
                preds=preds,
                log_prob=original_log_prob,
                mismatch_result=None,
                iterations=0,
            )

        mismatch_matrix, _ = mismatch_result

        allowed_links = (
            mismatch_matrix < self.config.seq_link_threshold
        ).astype(int)

        np.fill_diagonal(allowed_links.values, 0)

        iterations = self.config.triplet_prob_iterations
        log_prob = original_log_prob

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

            self.config.logger.info(
                "Completed triplet iteration %d",
                iteration + 1,
            )

        return self._assign_from_matrix(
            obs=obs,
            preds=preds,
            log_prob=log_prob,
            mismatch_result=mismatch_result,
            iterations=iterations,
        )

    def _assign_from_matrix(
        self,
        obs: pd.DataFrame,
        preds: pd.DataFrame,
        log_prob: pd.DataFrame,
        mismatch_result: tuple | None,
        iterations: int,
    ) -> AssignmentResult:
        """Construct an assignment without using DirectStrategy."""

        # Hungarian assignment.
        matching = self.find_best_assignment(
            score_matrix=log_prob,
            maximise=self.config.maximise_assignments,
        )

        # Construct the assignment dataframe.
        assign_df = make_assign_df(
            config=self.config,
            obs=obs,
            preds=preds,
            log_prob_matrix=log_prob,
            matching=matching,
        )

        # Add sequential consistency information.
        if mismatch_result is not None:
            mismatch_matrix, consistent_links_matrix = mismatch_result
        else:
            mismatch_matrix = None
            consistent_links_matrix = None

        assign_df = self.add_consistency_info(
            assign_df=assign_df,
            preds=preds,
            mismatch_matrix=mismatch_matrix,
            consistent_links_matrix=consistent_links_matrix,
            threshold=self.config.seq_link_threshold,
            logger=self.config.logger,
        )

        confidence = (
            assign_df[["SS_name", "Res_name", "Confidence"]]
            if "Confidence" in assign_df.columns
            else None
        )

        return AssignmentResult(
            assignments=matching,
            assign_df=assign_df,
            log_prob_matrix=log_prob,
            mismatch_matrix=mismatch_matrix,
            confidence=confidence,
            metadata={
                "strategy": "triplet",
                "iterations": iterations,
                "config": self.config,
            },
        )

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
