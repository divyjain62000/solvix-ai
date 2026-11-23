
from dataclasses import replace

import pandas as pd

from backend.core.profiling.models import (
    ColumnProfile,
    ProfilingConfig,
    QualityAnalysisResult,
)


class QualityAnalyzer:
    """Analyzes dataset quality, including missing values and duplicates."""

    def analyze(
        self,
        dataframe: pd.DataFrame,
        column_profiles: list[ColumnProfile],
        config: ProfilingConfig,
    ) -> QualityAnalysisResult:
        """
        Coordinate all dataset quality checks.
        """
        updated_profiles = [
            replace(profile) for profile in column_profiles
        ]

        total_missing_values = 0
        duplicate_rows = 0

        if config.include_missing_values:
            updated_profiles, total_missing_values = (
                self._analyze_missing_values(
                    dataframe=dataframe,
                    column_profiles=updated_profiles,
                )
            )

        if config.include_duplicate_rows:
            duplicate_rows = self._count_duplicate_rows(dataframe)

        return QualityAnalysisResult(
            columns=updated_profiles,
            total_missing_values=total_missing_values,
            duplicate_rows=duplicate_rows,
        )

    def _analyze_missing_values(
        self,
        dataframe: pd.DataFrame,
        column_profiles: list[ColumnProfile],
    ) -> tuple[list[ColumnProfile], int]:
        """
        Calculate missing values and percentages for every column.
        """
        total_missing_values = 0

        profiles_by_name = {
            profile.name: profile
            for profile in column_profiles
        }

        total_rows = len(dataframe)

        for column_name, series in dataframe.items():
            missing_count = int(series.isna().sum())

            missing_percentage = self._calculate_missing_percentage(
                missing_count=missing_count,
                total_rows=total_rows,
            )

            total_missing_values += missing_count

            profile = profiles_by_name.get(column_name)

            if profile is not None:
                profile.missing_values = missing_count
                profile.missing_percentage = missing_percentage

        return column_profiles, total_missing_values

    def _calculate_missing_percentage(
        self,
        missing_count: int,
        total_rows: int,
    ) -> float:
        """
        Calculate missing percentage safely.
        """
        if total_rows == 0:
            return 0.0

        return (missing_count / total_rows) * 100

    def _count_duplicate_rows(
        self,
        dataframe: pd.DataFrame,
    ) -> int:
        """
        Count duplicate rows, excluding their first occurrences.
        """
        return int(dataframe.duplicated(keep="first").sum())
