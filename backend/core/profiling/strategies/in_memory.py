import pandas as pd

from backend.core.profiling.models import (
    DatasetProfile,
    ProfilingConfig,
)

from backend.core.profiling.strategies.base import ProfilingStrategy

from backend.core.profiling.analyzers.schema_analyzer import SchemaAnalyzer
from backend.core.profiling.analyzers.quality_analyzer import QualityAnalyzer
from backend.core.profiling.analyzers.statistics_analyzer import StatisticsAnalyzer


class InMemoryProfilingStrategy(ProfilingStrategy):
    """Profiles datasets loaded completely into memory."""

    def __init__(self, config: ProfilingConfig | None = None):

        self.config = config or ProfilingConfig()

        self.schema_analyzer = SchemaAnalyzer()
        self.quality_analyzer = QualityAnalyzer()
        self.statistics_analyzer = StatisticsAnalyzer()


    def profile(
        self,
        data: pd.DataFrame,
        dataset_id: str,
        config: ProfilingConfig | None = None,
    ) -> DatasetProfile:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        active_config = config or self.config

        schema_column_profiles = self.schema_analyzer.analyze(data)

        quality_result = self.quality_analyzer.analyze(
            dataframe=data,
            column_profiles=schema_column_profiles,
            config=active_config,
        )

        statistics_column_profiles = self.statistics_analyzer.analyze(
            dataframe=data,
            column_profiles=quality_result.columns,
            config=active_config,
        )

        total_rows, total_columns = data.shape

        return DatasetProfile(
            dataset_id=dataset_id,
            total_rows=total_rows,
            total_columns=total_columns,
            total_missing_values=quality_result.total_missing_values,
            duplicate_rows=quality_result.duplicate_rows,
            memory_usage_bytes=int(data.memory_usage(deep=True).sum()),
            columns=statistics_column_profiles,
        )
