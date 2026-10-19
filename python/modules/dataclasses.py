from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
import pandas as pd


@dataclass
class ShiftData:
    """Container for all imported shift data"""
    obs_shifts: pd.DataFrame  # Observed shifts (i, i-1)
    pred_shifts: pd.DataFrame  # Predicted shifts (i, i-1)
    sequence: Optional[pd.DataFrame] = None  # Protein sequence
    aa_restraints: Optional[pd.DataFrame] = None  # Amino acid type restraints
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
    consistency_metrics: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    enriched_assignments: pd.DataFrame

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        return {
            'total_assignments': len(self.enriched_assignments),
            'consistency': self.consistency_metrics,
            'quality': self.quality_metrics,
        }