from pathlib import Path
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
import yaml


class SNAPSConfig(BaseModel):
    """Configuration model with validation"""

    # Atom settings
    atom_set: List[str] = Field(
        default=['H', 'N', 'HA', 'CA', 'CB', 'C'],
        description="Atom types to use in assignment"
    )
    atom_sd: Dict[str, float] = Field(
        default={
            'H': 0.1726, 'N': 1.1169, 'HA': 0.1231,
            'CA': 0.4412, 'CB': 0.5163, 'C': 0.5256
        },
        description="Standard deviations for each atom type"
    )

    # Assignment settings
    seq_link_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Carbon shift threshold for sequential links"
    )
    iterate_until_consistent: bool = Field(
        default=True,
        description="Iterate assignments until sequentially consistent"
    )
    use_ss_class_info: bool = Field(
        default=False,
        description="Use amino acid class restraints"
    )
    delta_correlation: bool = Field(
        default=True,
        description="Account for correlations in prediction errors"
    )

    # Strategy
    strategy: str = Field(
        default='direct',
        description="Assignment strategy: iterative, direct, bayesian"
    )

    @classmethod
    def load_default(cls) -> 'SNAPSConfig':
        """Load default configuration"""
        # Built-in defaults
        return cls()

    @classmethod
    def load_from_file(cls, config_path: Path) -> 'SNAPSConfig':
        """Load configuration from YAML file"""
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def merge(self, overrides: dict) -> 'SNAPSConfig':
        """Merge with overrides, overrides take priority"""
        merged = self.model_dump()
        merged.update({k: v for k, v in overrides.items() if v is not None})
        return SNAPSConfig(**merged)


def build_config(
        config_file: Optional[Path] = None,
        cli_overrides: Optional[dict] = None
) -> SNAPSConfig:
    """
    Build configuration with proper priority.

    Priority: CLI overrides > config file > defaults
    """
    # Start with defaults
    config = SNAPSConfig.load_default()

    # Merge config file if provided
    if config_file and config_file.exists():
        file_config = SNAPSConfig.load_from_file(config_file)
        config = config.merge(file_config.model_dump())

    # Apply CLI overrides
    if cli_overrides:
        config = config.merge(cli_overrides)

    # Double triple check that everything is deterministic
    config.atom_set = sorted(config.atom_set)

    return config