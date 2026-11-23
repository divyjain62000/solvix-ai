"""Handler for detecting and handling numerical outliers."""

import math
from uuid import uuid4
from typing import Iterable, Iterator

import pandas as pd

from backend.core.cleaning.models import (
    CleaningConfig,
    CleaningOperationResult,
    OutlierAnalysisResult,
    ColumnOutlierInfo,
)
from backend.core.profiling.models import DatasetProfile


class OutlierHandler:
    """Analyze and handle numerical outliers."""

    def analyze(
        self,
        dataframe: pd.DataFrame,
        profile: DatasetProfile,
        config: CleaningConfig,
    ) -> OutlierAnalysisResult:
        """Analyze outliers in an in-memory DataFrame."""

        self._validate_inputs(dataframe, profile, config)

        selected_columns = self._get_selected_columns(profile, config)
        column_results = []

        total_outlier_values = 0

        for column in selected_columns:
            lower_bound, upper_bound = self._calculate_bounds(
                column,
                config,
            )

            values = pd.to_numeric(
                dataframe[column.name],
                errors="coerce",
            )

            outlier_mask = values.notna() & (
                (values < lower_bound) | (values > upper_bound)
            )

            outlier_count = int(outlier_mask.sum())

            total_values = int(values.notna().sum())

            outlier_percentage = (
                outlier_count / total_values * 100 if total_values > 0 else 0.0
            )

            total_outlier_values += outlier_count

            column_results.append(
                ColumnOutlierInfo(
                    column_name=column.name,
                    total_values=total_values,
                    outlier_count=outlier_count,
                    outlier_percentage=outlier_percentage,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
            )

        return OutlierAnalysisResult(
            total_rows=len(dataframe),
            total_outlier_values=total_outlier_values,
            columns_with_outliers=sum(
                info.outlier_count > 0 for info in column_results
            ),
            columns=column_results,
        )

    def analyze_chunks(
        self,
        chunks: Iterable[pd.DataFrame],
        profile: DatasetProfile,
        config: CleaningConfig,
    ) -> OutlierAnalysisResult:
        """Analyze outliers across chunks using global profile statistics."""

        if chunks is None:
            raise ValueError("chunks cannot be None.")

        selected_columns = self._get_selected_columns(profile, config)

        bounds = {
            column.name: self._calculate_bounds(column, config)
            for column in selected_columns
        }

        outlier_counts = {column.name: 0 for column in selected_columns}

        valid_value_counts = {column.name: 0 for column in selected_columns}

        total_rows = 0

        for chunk in chunks:
            self._validate_dataframe(chunk, selected_columns)

            total_rows += len(chunk)

            for column in selected_columns:
                values = pd.to_numeric(
                    chunk[column.name],
                    errors="coerce",
                )

                lower_bound, upper_bound = bounds[column.name]

                outlier_mask = values.notna() & (
                    (values < lower_bound) | (values > upper_bound)
                )

                outlier_counts[column.name] += int(outlier_mask.sum())

                valid_value_counts[column.name] += int(values.notna().sum())

        column_results = []

        for column in selected_columns:
            outlier_count = outlier_counts[column.name]
            total_values = valid_value_counts[column.name]

            lower_bound, upper_bound = bounds[column.name]

            percentage = outlier_count / total_values * 100 if total_values > 0 else 0.0

            column_results.append(
                ColumnOutlierInfo(
                    column_name=column.name,
                    total_values=total_values,
                    outlier_count=outlier_count,
                    outlier_percentage=percentage,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
            )

        total_outlier_values = sum(outlier_counts.values())

        return OutlierAnalysisResult(
            total_rows=total_rows,
            total_outlier_values=total_outlier_values,
            columns_with_outliers=sum(count > 0 for count in outlier_counts.values()),
            columns=column_results,
        )

    def clean(
        self,
        dataframe: pd.DataFrame,
        config: CleaningConfig,
        profile: DatasetProfile,
    ) -> tuple[pd.DataFrame, CleaningOperationResult]:
        """Handle outliers in an in-memory DataFrame."""

        self._validate_inputs(dataframe, profile, config)

        result = dataframe.copy(deep=True)

        selected_columns = self._get_selected_columns(profile, config)

        rows_before = len(result)
        values_changed = 0
        rows_removed = 0

        parameters = {
            "method": config.outlier_method,
            "action": config.outlier_action,
            "iqr_multiplier": config.iqr_multiplier,
            "zscore_threshold": config.zscore_threshold,
        }

        if config.outlier_action == "remove":
            combined_mask = pd.Series(
                True,
                index=result.index,
            )

            for column in selected_columns:
                lower_bound, upper_bound = self._calculate_bounds(
                    column,
                    config,
                )

                values = pd.to_numeric(
                    result[column.name],
                    errors="coerce",
                )

                outlier_mask = values.notna() & (
                    (values < lower_bound) | (values > upper_bound)
                )

                combined_mask &= ~outlier_mask

            rows_removed = int((~combined_mask).sum())

            result = result.loc[combined_mask].reset_index(drop=True)

        elif config.outlier_action == "cap":
            for column in selected_columns:
                lower_bound, upper_bound = self._calculate_bounds(
                    column,
                    config,
                )

                values = pd.to_numeric(
                    result[column.name],
                    errors="coerce",
                )

                outlier_mask = values.notna() & (
                    (values < lower_bound) | (values > upper_bound)
                )

                values_changed += int(outlier_mask.sum())

                result.loc[outlier_mask, column.name] = values.loc[outlier_mask].clip(
                    lower=lower_bound,
                    upper=upper_bound,
                )

        else:
            raise ValueError(f"Unsupported outlier action: {config.outlier_action}")

        operation_result = CleaningOperationResult(
            operation_id=str(uuid4()),
            operation_type="outlier_handling",
            description=f"Handled numerical outliers using {config.outlier_method}.",
            affected_columns=[column.name for column in selected_columns],
            rows_before=rows_before,
            rows_after=len(result),
            rows_removed=rows_removed,
            values_changed=values_changed,
            parameters=parameters,
        )

        return result, operation_result

    def clean_chunks(
        self,
        chunks: Iterable[pd.DataFrame],
        config: CleaningConfig,
        profile: DatasetProfile,
    ) -> tuple[Iterator[pd.DataFrame], CleaningOperationResult]:
        """Handle outliers across chunks using global profile statistics."""

        if chunks is None:
            raise ValueError("chunks cannot be None.")

        if not isinstance(config, CleaningConfig):
            raise ValueError("config must be a CleaningConfig.")

        if not isinstance(profile, DatasetProfile):
            raise ValueError("profile must be a DatasetProfile.")

        if config.outlier_action not in ("remove", "cap"):
            raise ValueError(f"Unsupported outlier action: {config.outlier_action}")

        selected_columns = self._get_selected_columns(profile, config)

        bounds = {
            column.name: self._calculate_bounds(column, config)
            for column in selected_columns
        }

        rows_before = 0
        rows_after = 0
        rows_removed = 0
        values_changed = 0

        operation_result = CleaningOperationResult(
            operation_id=str(uuid4()),
            operation_type="outlier_handling",
            description=(f"Handled numerical outliers using {config.outlier_method}."),
            affected_columns=[column.name for column in selected_columns],
            parameters={
                "method": config.outlier_method,
                "action": config.outlier_action,
                "iqr_multiplier": config.iqr_multiplier,
                "zscore_threshold": config.zscore_threshold,
                "processing_mode": "chunked",
            },
        )

        def generate():
            nonlocal rows_before
            nonlocal rows_after
            nonlocal rows_removed
            nonlocal values_changed

            for chunk in chunks:
                self._validate_dataframe(chunk, selected_columns)

                result = chunk.copy(deep=True)
                rows_before += len(result)

                if config.outlier_action == "remove":
                    combined_mask = pd.Series(
                        True,
                        index=result.index,
                    )

                    for column in selected_columns:
                        lower_bound, upper_bound = bounds[column.name]

                        values = pd.to_numeric(
                            result[column.name],
                            errors="coerce",
                        )

                        outlier_mask = values.notna() & (
                            (values < lower_bound) | (values > upper_bound)
                        )

                        combined_mask &= ~outlier_mask

                    rows_removed += int((~combined_mask).sum())

                    result = result.loc[combined_mask]

                else:  # cap
                    for column in selected_columns:
                        lower_bound, upper_bound = bounds[column.name]

                        values = pd.to_numeric(
                            result[column.name],
                            errors="coerce",
                        )

                        outlier_mask = values.notna() & (
                            (values < lower_bound) | (values > upper_bound)
                        )

                        values_changed += int(outlier_mask.sum())

                        result.loc[outlier_mask, column.name] = values.loc[
                            outlier_mask
                        ].clip(
                            lower=lower_bound,
                            upper=upper_bound,
                        )

                rows_after += len(result)

                yield result

        def tracked_chunks():
            try:
                yield from generate()
            finally:
                operation_result.rows_before = rows_before
                operation_result.rows_after = rows_after
                operation_result.rows_removed = rows_removed
                operation_result.values_changed = values_changed

        return tracked_chunks(), operation_result

    def _validate_inputs(
        self,
        dataframe: pd.DataFrame,
        profile: DatasetProfile,
        config: CleaningConfig,
    ) -> None:
        """Validate handler inputs."""

        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("dataframe must be a pandas DataFrame.")

        if not isinstance(profile, DatasetProfile):
            raise ValueError("profile must be a DatasetProfile.")

        if not isinstance(config, CleaningConfig):
            raise ValueError("config must be a CleaningConfig.")

        self._validate_dataframe(
            dataframe,
            self._get_selected_columns(profile, config),
        )

    def _validate_dataframe(
        self,
        dataframe: pd.DataFrame,
        columns: list,
    ) -> None:
        """Validate the DataFrame and selected columns."""

        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Every chunk must be a pandas DataFrame.")

        missing_columns = [
            column.name for column in columns if column.name not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(f"Columns missing from DataFrame: {missing_columns}")

        for column in columns:
            if not pd.api.types.is_numeric_dtype(dataframe[column.name]):
                raise ValueError(f"Column '{column.name}' must contain numerical data.")

    def _get_selected_columns(
        self,
        profile: DatasetProfile,
        config: CleaningConfig,
    ) -> list:
        """Get selected numerical columns from the profile."""

        columns_by_name = {column.name: column for column in profile.columns}

        if config.columns_to_clean is None:
            selected_names = [
                column.name
                for column in profile.columns
                if column.inferred_type == "numerical"
            ]
        else:
            selected_names = config.columns_to_clean

        selected_columns = []

        for column_name in selected_names:
            column = columns_by_name.get(column_name)

            if column is None:
                raise ValueError(f"Column '{column_name}' does not exist in profile.")

            if column.inferred_type != "numerical":
                raise ValueError(f"Column '{column_name}' is not numerical.")

            selected_columns.append(column)

        return selected_columns

    def _calculate_bounds(
        self,
        column,
        config: CleaningConfig,
    ) -> tuple[float, float]:
        """Calculate outlier bounds from global profile statistics."""

        statistics = column.numeric_statistics

        if config.outlier_method == "iqr":
            q1 = statistics.get("25%")
            q3 = statistics.get("75%")

            if q1 is None or q3 is None:
                raise ValueError(f"IQR statistics are unavailable for '{column.name}'.")

            q1 = self._validate_statistic(q1, "25%", column.name)
            q3 = self._validate_statistic(q3, "75%", column.name)

            if config.iqr_multiplier <= 0:
                raise ValueError("iqr_multiplier must be greater than zero.")

            iqr = q3 - q1

            return (
                q1 - config.iqr_multiplier * iqr,
                q3 + config.iqr_multiplier * iqr,
            )

        if config.outlier_method == "zscore":
            mean = statistics.get("mean")
            std = statistics.get("std")

            if mean is None or std is None:
                raise ValueError(
                    f"Z-score statistics are unavailable for '{column.name}'."
                )

            mean = self._validate_statistic(mean, "mean", column.name)
            std = self._validate_statistic(std, "std", column.name)

            if std <= 0:
                raise ValueError(
                    f"Standard deviation must be greater than zero "
                    f"for column '{column.name}'."
                )

            if config.zscore_threshold <= 0:
                raise ValueError("zscore_threshold must be greater than zero.")

            return (
                mean - config.zscore_threshold * std,
                mean + config.zscore_threshold * std,
            )

        raise ValueError(f"Unsupported outlier method: {config.outlier_method}")

    @staticmethod
    def _validate_statistic(
        value,
        statistic_name: str,
        column_name: str,
    ) -> float:
        """Validate a numeric profiling statistic."""

        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Invalid {statistic_name} statistic for '{column_name}'."
            ) from exc

        if not math.isfinite(value):
            raise ValueError(
                f"Non-finite {statistic_name} statistic for '{column_name}'."
            )

        return value
