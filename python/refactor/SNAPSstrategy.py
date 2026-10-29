
from refactor.SNAPSconfig import SNAPSConfig
from refactor.strategies.base import AssignmentStrategy
from refactor.strategies.registry import StrategyRegistry


# Factory function (simplified - just delegates to registry)
def create_strategy(
    strategy_name: str,
    config: SNAPSConfig | dict | None = None,
) -> AssignmentStrategy:
    """
    Factory function to create assignment strategies.
    Strategies are auto-registered via @StrategyRegistry.register decorator.
    """
    return StrategyRegistry.create(strategy_name, config)
