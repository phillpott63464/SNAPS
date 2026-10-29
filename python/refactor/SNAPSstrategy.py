from abc import ABC, abstractmethod
from refactor.dataclasses import ShiftData, AssignmentResult
from refactor.SNAPSconfig import SNAPSConfig, build_config
import pandas as pd
import numpy as np
from typing import Dict, Type, List, Optional, Union
from scipy.stats import norm, multivariate_normal
from math import log10
from modules.assigner_functions import df_lookup

from scipy.optimize import linear_sum_assignment

from refactor.strategies.registry import StrategyRegistry
from refactor.strategies.base import AssignmentStrategy

# Factory function (simplified - just delegates to registry)
def create_strategy(
    strategy_name: str,
    config: Optional[Union[SNAPSConfig, dict]] = None,
) -> AssignmentStrategy:
    """
    Factory function to create assignment strategies.
    Strategies are auto-registered via @StrategyRegistry.register decorator.
    """
    return StrategyRegistry.create(strategy_name, config)
