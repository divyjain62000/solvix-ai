
from typing import Iterator

import pandas as pd

from backend.core.profiling.models import DatasetProfile, ProfilingConfig
from backend.core.profiling.strategies.base import ProfilingStrategy
from backend.core.profiling.analyzers.chunked_schema_analyzer import (
    ChunkedSchemaAnalyzer,
)
from backend.core.profiling.analyzers.chunked_quality_analyzer import (
    ChunkedQualityAnalyzer,
)
from backend.core.profiling.analyzers.chunked_statistics_analyzer import (
    ChunkedStatisticsAnalyzer,
)


class ChunkedProfilingStrategy(ProfilingStrategy):
    """Profiling strategy for datasets processed in chunks."""

    def __init__(self, config: ProfilingConfig | None = None):
        self.config = config or ProfilingConfig()

        self.schema_analyzer = ChunkedSchemaAnalyzer(self.config)
        self.quality_analyzer = ChunkedQualityAnalyzer(self.config)
        self.statistics_analyzer = ChunkedStatisticsAnalyzer(self.config)

    def profile(
        self,
        data: Iterator[pd.DataFrame],
        dataset_id: str,
        config: ProfilingConfig | None = None,
    ) -> DatasetProfile:
        """Profile a dataset using a single-pass chunk iterator."""

        active_config = config or self.config

        # Create fresh analyzers for every profiling execution.
        # This prevents state leakage between repeated calls.
        self.schema_analyzer = ChunkedSchemaAnalyzer(active_config)
        self.quality_analyzer = ChunkedQualityAnalyzer(active_config)
        self.statistics_analyzer = ChunkedStatisticsAnalyzer(active_config)

        has_chunks = False

        try:
            for chunk in data:
                has_chunks = True

                if not isinstance(chunk, pd.DataFrame):
                    raise TypeError(
                        "Every item in data must be a pandas DataFrame."
                    )

                self.schema_analyzer.update(chunk)
                self.quality_analyzer.update(chunk)
                self.statistics_analyzer.update(chunk)

            # Do not finalize analyzers when no chunks were received.
            if not has_chunks:
                raise ValueError(
                    "Cannot profile an empty chunk iterator."
                )

            # Finalize schema analysis.
            column_profiles = self.schema_analyzer.finalize()

            # Finalize data quality analysis.
            quality_result = self.quality_analyzer.finalize(
                column_profiles
            )

            # Finalize statistical analysis.
            column_profiles = self.statistics_analyzer.finalize(
                quality_result.columns
            )

            return DatasetProfile(
                dataset_id=dataset_id,
                total_rows=self.quality_analyzer.total_rows,
                total_columns=len(column_profiles),
                total_missing_values=quality_result.total_missing_values,
                duplicate_rows=quality_result.duplicate_rows,
                memory_usage_bytes=0,
                columns=column_profiles,
            )

        finally:
            # Cleanup must happen after both success and failure.
            self.quality_analyzer.close()
            self.statistics_analyzer.close()
