from pathlib import Path
from typing import Optional, List, Dict, Set, Literal
import logging
from pydantic import BaseModel, Field
import yaml


class SNAPSConfig(BaseModel):
    """Configuration model with validation"""

    # Input and output
    obs_file: Optional[Path] = Field(default=None, description="Observed shifts file")
    pred_file: Optional[Path] = Field(default=None, description="Predicted shifts file")
    obs_format: Optional[str] = Field(default=None, description="Observed shifts format")
    pred_format: Optional[str] = Field(default=None, description="Predicted shifts format")
    obs_frame: Optional[str] = Field(default=None, description="Observed shifts frame")
    pred_frame: Optional[str] = Field(default=None, description="Predicted shifts frame")
    obs_chain: str = Field(default='A', description="Observed chain identifier")
    pred_chain: str = Field(default='A', description="Predicted chain identifier")
    aa_restraints_file: Optional[Path] = Field(default=None, description="Amino acid restraints file")
    aa_restraints_frame: Optional[str] = Field(default=None, description="Amino acid restraints frame")
    output_file: Optional[Path] = Field(default=None, description="Assignment output file")
    output_type: str = Field(default='snaps', description="Assignment output format")
    shift_output_file: Optional[Path] = Field(default=None, description="Chemical shift output file")
    shift_output_type: Optional[str] = Field(default=None, description="Chemical shift output format")
    shift_output_confidence: Optional[str] = Field(default=None, description="Chemical shift output confidence")

    logger: logging.Logger = Field(
        default_factory=lambda: logging.getLogger('snaps'),
        exclude=True,
        description="Logger used by the assignment pipeline"
    )

    # Enables pydantic to hold a logger, otherwise it complains
    model_config = {
        'arbitrary_types_allowed': True,
    }

    # Atom settings
    atom_set: Set[str] = Field(
        default={'H', 'N', 'HA', 'CA', 'CB', 'C', 'C_m1', 'CA_m1', 'CB_m1'},
        description="Atom types to use in assignment"
    )
    atom_sd: Dict[str, float] = Field(
        default={
            'H': 0.1726, 'N': 1.1169, 'HA': 0.1231,
            'CA': 0.4412, 'CB': 0.5163, 'C': 0.5256
        },
        description="Standard deviations for each atom type"
    )
    supported_atom_types: Set[str] = Field(
        default={'H', 'N', 'C', 'CA', 'CB', 'C_m1', 'CA_m1', 'CB_m1', 'HA'},
        description="Atom columns accepted during preprocessing"
    )

    # Assignment settings
    default_probability: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
        description="Probability assigned to a missing shift"
    )
    shift_scale_factor: float = Field(
        default=1.0,
        gt=0.0,
        description="Scale factor applied to atom standard deviations"
    )
    pred_offset: int = Field(
        default=0,
        description="Offset applied to predicted residue numbers"
    )
    maximise_assignments: bool = Field(
        default=True,
        description="Maximise assignment scores rather than minimise them"
    )
    dummy_predicted_prefix: str = Field(
        default='DR_',
        min_length=1,
        description="Prefix used for dummy predicted residues"
    )
    dummy_observed_prefix: str = Field(
        default='DSS_',
        min_length=1,
        description="Prefix used for dummy observed spin systems"
    )
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
    pred_correction: bool = Field(
        default=False,
        description="Use prediction correction"
    )
    pred_correction_file: str = Field(
        default='config/lin_model_shiftx2.csv',
        description="File containing parameters for linear correction to predicted shifts"
    )
    delta_correlation_mean_corrected_file: str = Field(
        default='config/dd_mean.csv',
        description="File containing mean prediction errors for corrected predictions"
    )
    delta_correlation_cov_corrected_file: str = Field(
        default='config/dd_cov.csv',
        description="File containing prediction-error covariances for corrected predictions"
    )
    delta_correlation_mean_file: str = Field(
        default='config/d_mean.csv',
        description="File containing mean prediction errors"
    )
    delta_correlation_cov_file: str = Field(
        default='config/d_cov.csv',
        description="File containing prediction-error covariances"
    )

    # Strategy
    strategy: Literal['direct', 'iterative', 'bayesian'] = Field(
        default='direct',
        description="Assignment strategy: iterative, direct, bayesian"
    )

    @classmethod
    def load_default(cls) -> 'SNAPSConfig':
        """Load default configuration"""
        # Built-in defaults
        return cls.model_validate({})

    @classmethod
    def load_from_file(cls, config_path: Path) -> 'SNAPSConfig':
        """Load configuration from YAML file"""
        with open(config_path) as f:
            data = yaml.safe_load(f)
        # Resource paths explicitly written relative to the config file should
        # remain usable when the pipeline is invoked from another directory.
        path_fields = (
            'pred_correction_file',
            'delta_correlation_mean_corrected_file',
            'delta_correlation_cov_corrected_file',
            'delta_correlation_mean_file',
            'delta_correlation_cov_file',
        )
        for field in path_fields:
            value = data.get(field)
            if isinstance(value, str) and value.startswith('./'):
                data[field] = str(config_path.parent / value[2:])
        return cls.model_validate(data)

    def merge(self, overrides: dict) -> 'SNAPSConfig':
        """Merge with overrides, overrides take priority"""
        merged = self.model_dump()
        merged['logger'] = self.logger
        merged.update({k: v for k, v in overrides.items() if v is not None})
        return SNAPSConfig.model_validate(merged)


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
    config.atom_set = config.atom_set

    return config