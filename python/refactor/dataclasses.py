from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
import pandas as pd
from pathlib import Path


@dataclass
class InputSpec:
    """Input file specifications for the SNAPS pipeline."""

    # Required input files
    obs_file: Path
    pred_file: Path

    # Input file formats - not required as format_readers can identify file types
    obs_format: Optional[str] = None
    pred_format: Optional[str] = None

    # Coordinate/reference frames - not required as only NEF files have frames, and we can identify the frames automatically
    obs_frame: Optional[str] = None
    pred_frame: Optional[str] = None

    # Chain identifiers
    obs_chain: Optional[str] = None
    pred_chain: Optional[str] = None

    # Predicted residue numbering offset
    pred_offset: int = 0

    # Amino acid type restraints (optional)
    aa_restraints_file: Optional[str] = None
    aa_restraints_frame: Optional[str] = None

@dataclass
class OutputSpec:
    output_file: Path
    output_type: str = 'SNAPS'

    shift_output_file: Optional[Path] = None
    shift_output_type: Optional[str] = None
    shift_output_confidence: Optional[str] = None
    obs_chain: str = 'A' # This is also in the InputSpec, there's a problem there


@dataclass
class ShiftData:
    """Container for all imported shift data"""
    obs_shifts: pd.DataFrame  # Observed shifts (i, i-1)
    pred_shifts: pd.DataFrame  # Predicted shifts (i, i-1)
    sequence: Optional[pd.DataFrame] = None  # Protein sequence
    aa_restraints: Optional[pd.DataFrame] = None  # Amino acid type restraints # This is currently unimplemented as it was unimplemented in the original SNAPS
    unlabel_restraints: Optional[pd.DataFrame] = None  # Unlabelling info
    metadata: Dict[str, Any] = field(default_factory=dict)  # Sources, timestamps

    def __post_init__(self):
        """Add timestamp if not present"""
        if 'import_timestamp' not in self.metadata:
            self.metadata['import_timestamp'] = datetime.now().isoformat()


@dataclass
class AssignmentResult:
    """Results from assignment strategy"""
    assignments: pd.DataFrame  # Main assignment table
    assign_df: pd.DataFrame    # Assignment dataframe as originally implemented in SNAPS
    log_prob_matrix: Optional[pd.DataFrame] = None
    mismatch_matrix: Optional[pd.DataFrame] = None
    confidence: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON (web service)"""
        return {
            'assignments': self.assignments.to_dict(orient='records'),
            'metadata': self.metadata,
        }


@dataclass
class AnalysisResult:
    """Results from analysis"""
    consistency_metrics: Optional[Dict[str, Any]]
    quality_metrics: Optional[Dict[str, Any]]
    enriched_assignments: Optional[pd.DataFrame]

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        return {
            'total_assignments': len(self.enriched_assignments),
            'consistency': self.consistency_metrics,
            'quality': self.quality_metrics,
        }