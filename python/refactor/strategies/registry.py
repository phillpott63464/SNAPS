from refactor.SNAPSconfig import SNAPSConfig
from refactor.strategies.base import AssignmentStrategy

class StrategyRegistry:
    """Registry for self-registering assignment strategies"""

    _strategies: dict[str, type[AssignmentStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a strategy"""

        def decorator(strategy_class):
            strategy_class.name = name
            cls._strategies[name] = strategy_class
            return strategy_class

        return decorator

    @classmethod
    def get(cls, name: str) -> type[AssignmentStrategy]:
        """Get strategy class by name"""
        if name not in cls._strategies:
            available = ", ".join(cls._strategies.keys())
            raise ValueError(f"Unknown strategy: {name}. Available: {available}")
        return cls._strategies[name]

    @classmethod
    def list_names(cls) -> list[str]:
        """List all registered strategy names"""
        return sorted(cls._strategies.keys())

    @classmethod
    def create(
        cls,
        name: str,
        config: SNAPSConfig | dict | None = None,
    ) -> AssignmentStrategy:
        """Factory method to create a strategy instance"""
        strategy_class = cls.get(name)
        return strategy_class(config)
