
import pandas as pd
import numpy as np
from dataclasses import replace

from backend.core.profiling.models import (
    ColumnProfile,
    ProfilingConfig,
)


class StatisticsAnalyzer:
    """Analyzes numerical and categorical statistics of a dataset."""

    TOP_CATEGORIES_LIMIT = 10

    def analyze(
        self,
        dataframe: pd.DataFrame,
        column_profiles: list[ColumnProfile],
        config: ProfilingConfig,
    ) -> list[ColumnProfile]:
        """Coordinate numerical and categorical statistical analysis."""

        updated_profiles = [
            replace(profile) for profile in column_profiles
        ]

        for column_profile in updated_profiles:
            series = dataframe[column_profile.name]

            if (
                column_profile.inferred_type == "numerical"
                and config.include_numeric_statistics
            ):
                self._analyze_numeric_column(series, column_profile)

            elif (
                column_profile.inferred_type in {"categorical", "boolean"}
                and config.include_categorical_statistics
            ):
                self._analyze_categorical_column(series, column_profile)

        return updated_profiles

    def _analyze_numeric_column(
        self,
        series: pd.Series,
        profile: ColumnProfile,
    ) -> None:
        """Calculate numerical statistics and update the column profile."""

        numeric_series = pd.to_numeric(series, errors="coerce")

        # Exclude infinite values from statistical calculations.
        numeric_series = numeric_series.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if numeric_series.empty:
            profile.numeric_statistics = {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "25%": None,
                "50%": None,
                "75%": None,
                "max": None,
                "variance": None,
                "mode": None,
            }
            return

        statistics = numeric_series.describe().to_dict()

        statistics["variance"] = (
            float(numeric_series.var())
            if len(numeric_series) > 1
            else None
        )

        # Calculate mode. In case of multiple modes, select the first.
        modes = numeric_series.mode()
        statistics["mode"] = (
            float(modes.iloc[0])
            if not modes.empty
            else None
        )

        profile.numeric_statistics = {
            key: (
                float(value)
                if pd.notna(value)
                else None
            )
            for key, value in statistics.items()
        }

    def _analyze_categorical_column(
        self,
        series: pd.Series,
        profile: ColumnProfile,
    ) -> None:
        """Calculate categorical statistics and update the column profile."""

        non_missing_series = series.dropna()

        if non_missing_series.empty:
            profile.categorical_statistics = {
                "count": 0,
                "unique": 0,
                "top": None,
                "freq": 0,
                "mode": None,
                "value_counts": {},
                "top_values": {},
            }
            return

        value_counts = non_missing_series.value_counts()

        top_value = value_counts.index[0]
        top_frequency = int(value_counts.iloc[0])

        top_values = {
            key: int(value)
            for key, value in value_counts.head(
                self.TOP_CATEGORIES_LIMIT
            ).items()
        }

        profile.categorical_statistics = {
            "count": int(len(non_missing_series)),
            "unique": int(non_missing_series.nunique()),
            "top": top_value,
            "freq": top_frequency,
            "mode": top_value,
            "value_counts": top_values,
            "top_values": top_values.copy(),
        }
