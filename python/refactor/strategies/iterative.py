import pandas as pd
from refactor.dataclasses import AssignmentResult, ShiftData
from refactor.strategies.base import AssignmentStrategy
from refactor.strategies.direct import DirectStrategy
from refactor.strategies.registry import StrategyRegistry


@StrategyRegistry.register("iterative")
class IterativeStrategy(AssignmentStrategy):
    """Iterative refinement of direct assignment."""

    def assign(self, data: ShiftData) -> AssignmentResult:
        direct = DirectStrategy(config=self.config)
        obs, preds, original_log_prob = direct.prepare(data)
        initial_result = direct.assign_from_matrix(obs, preds, original_log_prob)
        return self._iterate_until_consistent(
            obs=obs,
            preds=preds,
            original_log_prob=original_log_prob,
            direct=direct,
            initial_result=initial_result,
        )

    def _iterate_until_consistent(
        self,
        obs: pd.DataFrame,
        preds: pd.DataFrame,
        original_log_prob: pd.DataFrame,
        direct: DirectStrategy,
        initial_result: AssignmentResult,
    ) -> AssignmentResult:
        """Iterate assignments until sequentially consistent"""
        threshold = self.config.seq_link_threshold
        best_result = initial_result
        best_count = (
            initial_result.assign_df["Confidence"].isin(["High", "Medium"]).sum()
        )

        mismatch_result = direct.calc_mismatch_matrix(
            obs=obs, threshold=threshold, logger=self.config.logger
        )
        if mismatch_result is None:
            best_result.log_prob_matrix = original_log_prob
            best_result.metadata["strategy"] = "iterative"
            return best_result
        mismatch_matrix, _ = mismatch_result

        while True:
            confident = best_result.assign_df.loc[
                best_result.assign_df["Confidence"].isin(["High", "Medium"]),
                ["Res_name", "SS_name"],
            ]
            working_log_prob = original_log_prob.copy()
            assignment_by_residue = confident.set_index("Res_name")["SS_name"]

            for res, ss in assignment_by_residue.items():
                res_m1 = preds.loc[res, "Res_name_m1"]
                res_p1 = preds.loc[res, "Res_name_p1"]
                penalty = working_log_prob.min().min()

                if not pd.isna(res_m1):
                    allowed_ss = mismatch_matrix.loc[:, ss] <= threshold
                    working_log_prob.loc[~allowed_ss, res_m1] += penalty
                if not pd.isna(res_p1):
                    allowed_ss = mismatch_matrix.loc[ss, :] <= threshold
                    working_log_prob.loc[~allowed_ss, res_p1] += penalty

            candidate = direct.assign_from_matrix(obs, preds, working_log_prob)
            candidate_count = (
                candidate.assign_df["Confidence"].isin(["High", "Medium"]).sum()
            )
            self.config.logger.info(
                "At the current stage, there are %d high or medium "
                "confidence assignments",
                candidate_count,
            )

            if candidate_count <= best_count:
                break
            best_result = candidate
            best_count = candidate_count

        best_result.log_prob_matrix = original_log_prob
        best_result.metadata["strategy"] = "iterative"
        return best_result
