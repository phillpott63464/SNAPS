from abc import ABC, abstractmethod
import pandas as pd

from refactor.dataclasses import AssignmentResult, ShiftData
from refactor.SNAPSconfig import SNAPSConfig, build_config
from refactor.strategies.utils import matching, preprocessing, consistency, probability


class AssignmentStrategy(ABC):
    """Base class for assignment strategies."""

    name: str = ""
    config: SNAPSConfig | None = None

    def __init__(self, config: SNAPSConfig | dict | None = None):
        if config is None:
            self.config = build_config()
        elif isinstance(config, dict):
            self.config = build_config(cli_overrides=config)
        else:
            self.config = config

    @abstractmethod
    def assign(self, data: ShiftData) -> AssignmentResult:
        """Perform assignment and return results"""

    def find_best_assignment(
        self,
        score_matrix,
        maximise=True,
        inc=None,
        exc=None,
        dummy_rows=None,
        dummy_cols=None,
        return_none_all_dummy=False,
    ):
        """Delegate to the shared assignment utility."""
        return matching.find_best_assignment(self.config, score_matrix=score_matrix, maximise=maximise, inc=inc, exc=exc, dummy_rows=dummy_rows, dummy_cols=dummy_cols, return_none_all_dummy=return_none_all_dummy)

    def preprocess_obs_preds(self, obs: pd.DataFrame, preds: pd.DataFrame):
        """Delegate to the shared assignment utility."""
        return preprocessing.preprocess_obs_preds(self.config, obs=obs, preds=preds)

    def calc_log_prob_matrix(
        self,
        obs: pd.DataFrame,
        preds: pd.DataFrame,
    ) -> pd.DataFrame:
        """Delegate to the shared assignment utility."""
        return probability.calc_log_prob_matrix(self.config, obs=obs, preds=preds)

    calc_mismatch_matrix = staticmethod(consistency.calc_mismatch_matrix)

    add_consistency_info = staticmethod(consistency.add_consistency_info)

