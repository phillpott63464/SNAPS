from refactor.SNAPSconfig import SNAPSConfig, build_config
from refactor.dataclasses import AssignmentResult, ShiftData, Analysis

from python.refactor.dataclasses import AnalysisResult

Result
from typing import Optional, List
import pandas as pd

class AnalyzerType():
    def analyze(self, config: SNAPSConfig) -> pd.DataFrame:
        ...

class AnalysisPipeline():
    def __init__(self, config: Optional[SNAPSConfig] = None, analyzers: List[AnalyzerType] = None):
        if config is None:
            self.config = build_config()
        else:
            self.config = config

        self.analyzers = analyzers

    def run(self, assignment: AssignmentResult, data: ShiftData) -> AnalysisResult:
        for analyzer in self.analyzers:
            analyzer.analyze(assignment, data)

class ConsistencyAnalyzer(AnalyzerType):
    ...

class QualityAnalyzer(AnalyzerType):
    ...