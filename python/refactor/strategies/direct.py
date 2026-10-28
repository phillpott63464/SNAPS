import pandas as pd
from modules.assigner_functions import df_lookup
from refactor.dataclasses import AssignmentResult, ShiftData
from refactor.strategies.base import AssignmentStrategy
from refactor.strategies.registry import StrategyRegistry


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
        """Make a dataframe with full assignment information, given a dataframe
        of SS_name and Res_name.

        Matching may have additional columns, which will also be kept.
        """
        obs = obs.copy()
        preds = preds.copy()
        obs.index.name = "name"  # Needed to avoid error when merging dataframes.
        valid_atoms = sorted(self.config.atom_set)
        extra_cols = set(matching.columns).difference({"SS_name", "Res_name"})

        if not {"SS_name", "Res_name"}.issubset(matching.columns):
            self.config.logger.warning(
                "Cannot make assignment dataframe, as matching"
                + " dataframe does not have the correct columns"
            )

        assign_df = pd.merge(
            matching,
            preds.loc[:, ["Res_N", "Res_type", "Res_name", "Dummy_res"]],
            on="Res_name",
            how="left",
        )
        assign_df = assign_df[
            ["Res_name", "Res_N", "Res_type", "SS_name", "Dummy_res"]
            + sorted(extra_cols)
        ]
        assign_df = pd.merge(
            assign_df,
            obs.loc[:, obs.columns.isin(["SS_name", "Dummy_SS"] + valid_atoms)],
            on="SS_name",
            how="left",
        )
        assign_df = pd.merge(
            assign_df,
            preds.loc[:, preds.columns.isin(valid_atoms + ["Res_name"])],
            on="Res_name",
            suffixes=("", "_pred"),
            how="left",
        )

        # assign_df["Log_prob"] = log_prob_matrix.lookup(
        #                                     assign_df["SS_name"],
        #                                     assign_df["Res_name"])
        assign_df["Log_prob"] = df_lookup(
            log_prob_matrix, assign_df["SS_name"], assign_df["Res_name"]
        )
        # Careful above not to get rows/columns confused

        assign_df = assign_df.sort_values(by="Res_N")

        return assign_df
