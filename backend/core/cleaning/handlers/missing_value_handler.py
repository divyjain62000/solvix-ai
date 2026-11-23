
"""Handler for detecting and cleaning missing values."""

from uuid import uuid4
from collections.abc import Iterable, Iterator

import pandas as pd

from backend.core.cleaning.models import (
    CleaningConfig,
    CleaningOperationResult,
    ColumnMissingInfo,
    MissingValueAnalysisResult,
)
from backend.core.profiling.models import (
    ColumnProfile,
    DatasetProfile,
)
from backend.core.cleaning.exceptions import MissingValueError


class MissingValueHandler:
    """Analyze and handle missing values in datasets."""

    def analyze(
        self,
        profile: DatasetProfile,
    ) -> MissingValueAnalysisResult:
        """Analyze missing values using an existing dataset profile."""

        if not isinstance(profile, DatasetProfile):
            raise MissingValueError("Invalid dataset profile.")

        missing_value_columns = []
        total_missing_values = 0
        columns_with_missing = 0

        for column in profile.columns:
            if column.missing_values > 0:
                missing_value_columns.append(
                    ColumnMissingInfo(
                        column_name=column.name,
                        total_values=column.total_values,
                        missing_values=column.missing_values,
                        missing_percentage=column.missing_percentage,
                    )
                )

                total_missing_values += column.missing_values
                columns_with_missing += 1

        return MissingValueAnalysisResult(
            total_rows=profile.total_rows,
            total_missing_values=total_missing_values,
            columns_with_missing=columns_with_missing,
            columns=missing_value_columns,
        )

    def clean(
        self,
        dataframe: pd.DataFrame,
        config: CleaningConfig,
        profile: DatasetProfile | None = None,
    ) -> tuple[pd.DataFrame, CleaningOperationResult]:
        """Clean missing values using the configured strategy."""

        if not isinstance(dataframe, pd.DataFrame):
            raise MissingValueError("Input must be a Pandas DataFrame.")

        if not isinstance(config, CleaningConfig):
            raise MissingValueError("Invalid cleaning configuration.")

        if profile is not None and not isinstance(profile, DatasetProfile):
            raise MissingValueError("Invalid dataset profile.")

        cleaned_df = dataframe.copy(deep=True)

        rows_before = len(cleaned_df)
        values_changed = 0
        rows_removed = 0
        affected_columns = []

        if not config.handle_missing_values:
            return cleaned_df, self._build_result(
                rows_before=rows_before,
                rows_after=rows_before,
                affected_columns=[],
                values_changed=0,
                config=config,
            )

        if config.columns_to_clean is None:
            selected_columns = list(cleaned_df.columns)
        else:
            missing_columns = (
                set(config.columns_to_clean) - set(cleaned_df.columns)
            )

            if missing_columns:
                raise MissingValueError(
                    f"Columns not found: {sorted(missing_columns)}"
                )

            selected_columns = config.columns_to_clean.copy()

        columns_with_missing = [
            column
            for column in selected_columns
            if cleaned_df[column].isna().any()
        ]

        if config.missing_value_strategy == "drop":
            if columns_with_missing:
                rows_removed = int(
                    cleaned_df[columns_with_missing].isna().any(axis=1).sum()
                )

                cleaned_df = cleaned_df.dropna(subset=selected_columns)

                affected_columns = columns_with_missing

        else:
            column_profiles = (
                {col.name: col for col in profile.columns}
                if profile is not None
                else {}
            )

            for column in columns_with_missing:
                missing_count = int(cleaned_df[column].isna().sum())

                if config.missing_value_strategy == "mean":
                    if (
                        not pd.api.types.is_numeric_dtype(cleaned_df[column])
                        or pd.api.types.is_bool_dtype(cleaned_df[column])
                    ):
                        raise MissingValueError(
                            f"Mean strategy requires a numerical column: "
                            f"'{column}'."
                        )

                    fill_value = self._get_statistic(
                        cleaned_df,
                        column,
                        column_profiles,
                        "mean",
                    )

                elif config.missing_value_strategy == "median":
                    if (
                        not pd.api.types.is_numeric_dtype(cleaned_df[column])
                        or pd.api.types.is_bool_dtype(cleaned_df[column])
                    ):
                        raise MissingValueError(
                            f"Median strategy requires a numerical column: "
                            f"'{column}'."
                        )

                    fill_value = self._get_statistic(
                        cleaned_df,
                        column,
                        column_profiles,
                        "median",
                    )

                elif config.missing_value_strategy == "mode":
                    fill_value = self._get_mode(
                        dataframe=cleaned_df,
                        column=column,
                        column_profiles=column_profiles,
                    )

                elif config.missing_value_strategy == "constant":
                    fill_value = config.missing_value_constant

                else:
                    raise MissingValueError(
                        "Unsupported missing value strategy."
                    )

                if pd.isna(fill_value):
                    raise MissingValueError(
                        f"Invalid fill value for column '{column}'."
                    )

                cleaned_df[column] = cleaned_df[column].fillna(fill_value)

                values_changed += missing_count
                affected_columns.append(column)

        rows_after = len(cleaned_df)

        operation_result = self._build_result(
            rows_before=rows_before,
            rows_after=rows_after,
            affected_columns=affected_columns,
            values_changed=values_changed,
            config=config,
        )

        operation_result.rows_removed = rows_removed

        return cleaned_df, operation_result

    def clean_chunks(
        self,
        chunks: Iterable[pd.DataFrame],
        config: CleaningConfig,
        profile: DatasetProfile | None = None,
    ) -> tuple[Iterator[pd.DataFrame], CleaningOperationResult]:
        """Clean missing values from a chunked dataset."""

        if chunks is None:
            raise MissingValueError("Chunks cannot be None.")

        if not isinstance(config, CleaningConfig):
            raise MissingValueError("Invalid cleaning configuration.")

        if profile is not None and not isinstance(profile, DatasetProfile):
            raise MissingValueError("Invalid dataset profile.")

        if (
            config.handle_missing_values
            and config.missing_value_strategy
            in {"mean", "median", "mode"}
            and profile is None
        ):
            raise MissingValueError(
                "Dataset profile is required for "
                f"{config.missing_value_strategy} chunked cleaning."
            )

        iterator = iter(chunks)

        rows_before = 0
        rows_after = 0
        rows_removed = 0
        values_changed = 0
        affected_columns = set()

        def generate() -> Iterator[pd.DataFrame]:
            nonlocal rows_before
            nonlocal rows_after
            nonlocal rows_removed
            nonlocal values_changed

            for chunk in iterator:
                if not isinstance(chunk, pd.DataFrame):
                    raise MissingValueError(
                        "Every chunk must be a Pandas DataFrame."
                    )

                cleaned_chunk, operation = self.clean(
                    dataframe=chunk,
                    config=config,
                    profile=profile,
                )

                rows_before += operation.rows_before
                rows_after += operation.rows_after
                rows_removed += operation.rows_removed
                values_changed += operation.values_changed

                affected_columns.update(operation.affected_columns)

                yield cleaned_chunk

        result = CleaningOperationResult(
            operation_id=str(uuid4()),
            operation_type="missing_value_handling",
            description=(
                "Missing values handled using "
                f"'{config.missing_value_strategy}' strategy."
            ),
            affected_columns=[],
            parameters={
                "strategy": config.missing_value_strategy,
                "constant": (
                    config.missing_value_constant
                    if config.missing_value_strategy == "constant"
                    else None
                ),
            },
        )

        def tracked_generator() -> Iterator[pd.DataFrame]:
            yield from generate()

            result.rows_before = rows_before
            result.rows_after = rows_after
            result.rows_removed = rows_removed
            result.values_changed = values_changed
            result.affected_columns = sorted(affected_columns)

        return tracked_generator(), result

    @staticmethod
    def _get_statistic(
        dataframe: pd.DataFrame,
        column: str,
        column_profiles: dict[str, ColumnProfile],
        statistic: str,
    ):
        """Get a numerical statistic from profile or DataFrame."""

        if column in column_profiles:
            statistics = column_profiles[column].numeric_statistics

            # Profiling stores median as the 50th percentile.
            profile_key = "50%" if statistic == "median" else statistic

            value = statistics.get(profile_key)

        else:
            value = getattr(dataframe[column], statistic)()

        if value is None or pd.isna(value):
            raise MissingValueError(
                f"{statistic.capitalize()} unavailable for "
                f"column '{column}'."
            )

        return value

    @staticmethod
    def _get_mode(
        dataframe: pd.DataFrame,
        column: str,
        column_profiles: dict[str, ColumnProfile],
    ):
        """Get the dataset-wide mode from profile or local DataFrame."""

        if column in column_profiles:
            column_profile = column_profiles[column]

            if column_profile.inferred_type == "numerical":
                mode = column_profile.numeric_statistics.get("mode")
            else:
                mode = column_profile.categorical_statistics.get("mode")

            if mode is None or pd.isna(mode):
                raise MissingValueError(
                    f"Mode unavailable for column '{column}'."
                )

            return mode

        # Local mode is suitable for in-memory cleaning when no
        # dataset-level profile is supplied.
        mode_values = dataframe[column].mode(dropna=True)

        if mode_values.empty:
            raise MissingValueError(
                f"No mode found for column '{column}'."
            )

        return mode_values.iloc[0]

    @staticmethod
    def _build_result(
        rows_before: int,
        rows_after: int,
        affected_columns: list[str],
        values_changed: int,
        config: CleaningConfig,
    ) -> CleaningOperationResult:
        """Build cleaning operation metadata."""

        return CleaningOperationResult(
            operation_id=str(uuid4()),
            operation_type="missing_value_handling",
            description=(
                "Missing values handled using "
                f"'{config.missing_value_strategy}' strategy."
            ),
            affected_columns=affected_columns,
            rows_before=rows_before,
            rows_after=rows_after,
            rows_removed=rows_before - rows_after,
            values_changed=values_changed,
            parameters={
                "strategy": config.missing_value_strategy,
                "constant": (
                    config.missing_value_constant
                    if config.missing_value_strategy == "constant"
                    else None
                ),
            },
        )
