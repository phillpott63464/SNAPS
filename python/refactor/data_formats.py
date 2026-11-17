from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd


@dataclass
class ShiftData:
    """Container for all imported shift data"""

    obs_shifts: pd.DataFrame  # Observed shifts (i, i-1)
    pred_shifts: pd.DataFrame  # Predicted shifts (i, i-1)
    sequence: Optional[pd.DataFrame] = None  # Protein sequence
    aa_restraints: Optional[pd.DataFrame] = (
        None  # Amino acid type restraints # This is currently unimplemented as it was unimplemented in the original SNAPS
    )
    unlabel_restraints: Optional[pd.DataFrame] = None  # Unlabelling info
    metadata: dict[str, Any] = field(default_factory=dict)  # Sources, timestamps

    def __post_init__(self):
        """Add timestamp if not present"""
        if "import_timestamp" not in self.metadata:
            self.metadata["import_timestamp"] = datetime.now().isoformat()


@dataclass
class AssignmentResult:
    """Results from assignment strategy"""

    assignments: pd.DataFrame  # Main assignment table
    assign_df: pd.DataFrame  # Assignment dataframe as originally implemented in SNAPS
    assign_dfs: Optional[list[pd.DataFrame]] = (
        None  # Dataframes for iterative strategies
    )
    log_prob_matrix: Optional[pd.DataFrame] = None
    mismatch_matrix: Optional[pd.DataFrame] = None
    confidence: Optional[pd.DataFrame] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON (web service)"""
        return {
            "assignments": self.assignments.to_dict(orient="records"),
            "metadata": self.metadata,
        }


@dataclass
class AnalysisResult:
    """Results from analysis"""

    consistency_metrics: Optional[dict[str, Any]]
    quality_metrics: Optional[dict[str, Any]]
    enriched_assignments: Optional[pd.DataFrame]

    def summary(self) -> dict[str, Any]:
        """Generate summary statistics"""
        return {
            "total_assignments": len(self.enriched_assignments),
            "consistency": self.consistency_metrics,
            "quality": self.quality_metrics,
        }
