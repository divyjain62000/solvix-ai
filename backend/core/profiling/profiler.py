from backend.core.profiling.analyzers.schema_analyzer import SchemaAnalyzer
from backend.core.profiling.analyzers.quality_analyzer import QualityAnalyzer
from backend.core.profiling.analyzers.statistics_analyzer import StatisticsAnalyzer
from backend.core.profiling.models import DatasetProfile, ProfilingConfig

import pandas as pd


class Profiler:
    """Orchestrates the complete dataset profiling process."""

    def __init__(self):
        self.schema_analyzer = SchemaAnalyzer()
        self.quality_analyzer = QualityAnalyzer()
        self.statistics_analyzer = StatisticsAnalyzer()

    def profile(
        self,
        dataframe: pd.DataFrame,
        dataset_id: str,
        config: ProfilingConfig | None = None,
    ) -> DatasetProfile:
        
        if config is None:
            config = ProfilingConfig()
        
        schema_column_profiles = self.schema_analyzer.analyze(dataframe)
        
        quality_analysis_result = self.quality_analyzer.analyze(
            dataframe, schema_column_profiles, config
        )
        
        statistics_analysis_column_profiles = self.statistics_analyzer.analyze(
            dataframe, quality_analysis_result.columns, config
        )
        
        
        total_rows, total_columns = dataframe.shape
        
        dataset_profile = DatasetProfile(
            dataset_id=dataset_id,
            total_rows=total_rows,
            total_columns=total_columns,
            total_missing_values=quality_analysis_result.total_missing_values,
            duplicate_rows=quality_analysis_result.duplicate_rows,
            
            memory_usage_bytes=int(dataframe.memory_usage(deep=True).sum()),
            columns=statistics_analysis_column_profiles
        )
        return dataset_profile
