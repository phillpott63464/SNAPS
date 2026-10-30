import pandas as pd
from refactor.dataclasses import AssignmentResult, ShiftData
from refactor.strategies.base import AssignmentStrategy
from refactor.strategies.direct import DirectStrategy
from refactor.strategies.registry import StrategyRegistry
from refactor.strategies.utils.iterative import _iterate_until_consistent


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

        return _iterate_until_consistent(
            self.config,
            obs=obs,
            preds=preds,
            original_log_prob=original_log_prob,
            direct=direct,
            initial_result=initial_result,
        )
