import pandas as pd
from refactor.dataclasses import AssignmentResult, ShiftData
from refactor.strategies.base import AssignmentStrategy
from refactor.strategies.registry import StrategyRegistry
from refactor.strategies.utils.output import make_assign_df


@StrategyRegistry.register("direct")
class DirectStrategy(AssignmentStrategy):
    def prepare(self, data: ShiftData):
        """Prepare input data and calculate the unmodified score matrix."""
        obs, preds = self.preprocess_obs_preds(data.obs_shifts, data.pred_shifts)
        return obs, preds, self.calc_log_prob_matrix(obs, preds)

    def assign(self, data: ShiftData, set_assign_df: bool = True) -> AssignmentResult:
        """Perform direct assignment from raw shift data."""
        obs, preds, log_prob = self.prepare(data)
        return self.assign_from_matrix(obs, preds, log_prob)

    def assign_from_matrix(
        self,
        obs: pd.DataFrame,
        preds: pd.DataFrame,
        log_prob: pd.DataFrame,
    ) -> AssignmentResult:
        """Solve an assignment using an already calculated score matrix."""
        matching = self.find_best_assignment(
            score_matrix=log_prob,
            maximise=self.config.maximise_assignments,
        )
        assign_df = self.make_assign_df(
            obs=obs,
            preds=preds,
            log_prob_matrix=log_prob,
            matching=matching,
        )

        result = self.calc_mismatch_matrix(
            obs=obs,
            threshold=self.config.seq_link_threshold,
            logger=self.config.logger,
        )
        if result is not None:
            mismatch_matrix, consistent_links_matrix = result
        else:
            mismatch_matrix, consistent_links_matrix = (None, None)

        assign_df = self.add_consistency_info(
            assign_df=assign_df,
            preds=preds,
            mismatch_matrix=mismatch_matrix,
            consistent_links_matrix=consistent_links_matrix,
            threshold=self.config.seq_link_threshold,
            logger=self.config.logger,
        )

        return AssignmentResult(
            assignments=matching,
            log_prob_matrix=log_prob,
            assign_df=assign_df,
            mismatch_matrix=mismatch_matrix,
            confidence=assign_df[["SS_name", "Res_name", "Confidence"]]
            if "Confidence" in assign_df
            else None,
            metadata={
                "strategy": "direct",
                "config": self.config,
            },
        )

    def make_assign_df(
        self,
        obs: pd.DataFrame,
        preds: pd.DataFrame,
        log_prob_matrix: pd.DataFrame,
        matching: pd.DataFrame,
    ):

        return make_assign_df(
            config=self.config,
            obs=obs,
            preds=preds,
            log_prob_matrix=log_prob_matrix,
            matching=matching,
        )
