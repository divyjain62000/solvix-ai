
import math
import numbers
import os
import pickle
import sqlite3
import tempfile

from dataclasses import replace

import numpy as np
import pandas as pd
from datasketch import HyperLogLog

from backend.core.profiling.analyzers.schema_analyzer import SchemaAnalyzer
from backend.core.profiling.models import (
    ColumnProfile,
    ProfilingConfig,
)


class ChunkedStatisticsAnalyzer:
    """Calculates dataset statistics incrementally across chunks."""

    TOP_CATEGORIES_LIMIT = 10

    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.schema_analyzer = SchemaAnalyzer()

        self.column_names = []
        self.numeric_states = {}
        self.categorical_columns = set()
        self.unique_sketches = {}

        self.sqlite_connection = None
        self.database_path = None

        self.initialized = False
        self.finalized = False

        if self.config.unique_count_mode not in {
            "exact",
            "approximate",
        }:
            raise ValueError(
                "unique_count_mode must be 'exact' or 'approximate'."
            )

    def _initialize_schema(self, chunk: pd.DataFrame) -> None:
        """Initialize column-specific statistics state from the first chunk."""

        if not isinstance(chunk, pd.DataFrame):
            raise TypeError("Chunk must be a pandas DataFrame.")

        if not chunk.columns.is_unique:
            raise ValueError("Duplicate column names are not supported.")

        self.column_names = list(chunk.columns)

        column_profiles = self.schema_analyzer.analyze(chunk)

        for profile in column_profiles:
            column_name = profile.name
            inferred_type = profile.inferred_type

            if (
                inferred_type == "numerical"
                and self.config.include_numeric_statistics
            ):
                self.numeric_states[column_name] = {
                    "count": 0,
                    "mean": 0.0,
                    "m2": 0.0,
                    "min": None,
                    "max": None,
                }

            elif (
                inferred_type in {"categorical", "boolean"}
                and self.config.include_categorical_statistics
            ):
                self.categorical_columns.add(column_name)

        if self.config.unique_count_mode == "approximate":
            for column_name in self.column_names:
                self.unique_sketches[column_name] = HyperLogLog(
                    p=self.config.hll_precision
                )

        self.initialized = True

    def _validate_chunk(self, chunk: pd.DataFrame) -> None:
        """Validate incoming chunks against the initialized schema."""

        if not isinstance(chunk, pd.DataFrame):
            raise TypeError("Chunk must be a pandas DataFrame.")

        if not self.initialized:
            raise RuntimeError("Schema has not been initialized.")

        if not chunk.columns.is_unique:
            raise ValueError("Duplicate column names are not supported.")

        if list(chunk.columns) != self.column_names:
            raise ValueError(
                "Chunk schema does not match the initialized schema."
            )

        if self.finalized:
            raise RuntimeError(
                "Cannot update statistics after finalization."
            )

    @staticmethod
    def _normalize_value(value):
        """Normalize values without losing ordinary integer precision."""

        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, bool):
            return value

        if isinstance(value, numbers.Integral):
            return int(value)

        if isinstance(value, numbers.Real):
            value = float(value)

            if math.isfinite(value) and value.is_integer():
                return int(value)

            return value

        return value

    @classmethod
    def _serialize_value(cls, value) -> bytes:
        """Serialize a normalized value."""

        normalized_value = cls._normalize_value(value)

        return pickle.dumps(
            normalized_value,
            protocol=5,
        )

    def _initialize_database(self) -> None:
        """Initialize the temporary SQLite database."""

        if self.sqlite_connection is not None:
            return

        file_descriptor, self.database_path = tempfile.mkstemp(
            prefix="solvixai_statistics_",
            suffix=".sqlite3",
        )

        os.close(file_descriptor)

        try:
            self.sqlite_connection = sqlite3.connect(
                self.database_path
            )

            self.sqlite_connection.executescript("""
                CREATE TABLE column_value_frequencies (
                    column_name TEXT NOT NULL,
                    value BLOB NOT NULL,
                    frequency INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (column_name, value)
                );

                CREATE TABLE exact_unique_values (
                    column_name TEXT NOT NULL,
                    value BLOB NOT NULL,
                    PRIMARY KEY (column_name, value)
                ) WITHOUT ROWID;

                CREATE TABLE numeric_column_values (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    column_name TEXT NOT NULL,
                    value NUMERIC NOT NULL
                );

                CREATE INDEX idx_numeric_column_values
                ON numeric_column_values(column_name, value);
            """)

            self.sqlite_connection.commit()

        except Exception:
            self.close()
            raise

    def _update_numeric_statistics(
        self,
        chunk: pd.DataFrame,
    ) -> None:
        """Update numeric statistics using Welford's algorithm."""

        if not self.config.include_numeric_statistics:
            return

        for column_name, state in self.numeric_states.items():

            values = pd.to_numeric(
                chunk[column_name],
                errors="coerce",
            )

            values = values.replace(
                [np.inf, -np.inf],
                np.nan,
            ).dropna()

            for value in values:
                value = float(value)

                state["count"] += 1

                delta = value - state["mean"]
                state["mean"] += delta / state["count"]

                delta_after_update = value - state["mean"]
                state["m2"] += delta * delta_after_update

                if state["min"] is None or value < state["min"]:
                    state["min"] = value

                if state["max"] is None or value > state["max"]:
                    state["max"] = value

    def _update_numeric_values(
        self,
        chunk: pd.DataFrame,
    ) -> None:
        """Store numerical values in SQLite for exact percentile calculation."""

        if not self.config.include_numeric_statistics:
            return

        if not self.numeric_states:
            return

        self._initialize_database()

        for column_name in self.numeric_states:

            values = pd.to_numeric(
                chunk[column_name],
                errors="coerce",
            )

            values = values.replace(
                [np.inf, -np.inf],
                np.nan,
            ).dropna()

            records = (
                (
                    column_name,
                    self._normalize_value(value),
                )
                for value in values
            )

            self.sqlite_connection.executemany(
                """
                INSERT INTO numeric_column_values (
                    column_name,
                    value
                )
                VALUES (?, ?)
                """,
                records,
            )

        self.sqlite_connection.commit()

    def _update_value_frequencies(
        self,
        column_name: str,
        values,
    ) -> None:
        """Aggregate value frequencies across chunks using SQLite."""

        records = (
            (
                column_name,
                self._serialize_value(value),
            )
            for value in values
        )

        self.sqlite_connection.executemany(
            """
            INSERT INTO column_value_frequencies (
                column_name,
                value,
                frequency
            )
            VALUES (?, ?, 1)
            ON CONFLICT(column_name, value)
            DO UPDATE SET frequency = frequency + 1
            """,
            records,
        )

    def _update_numeric_frequencies(
        self,
        chunk: pd.DataFrame,
    ) -> None:
        """Track numerical frequencies for exact mode calculation."""

        if not self.config.include_numeric_statistics:
            return

        if not self.numeric_states:
            return

        self._initialize_database()

        for column_name in self.numeric_states:

            values = pd.to_numeric(
                chunk[column_name],
                errors="coerce",
            )

            values = values.replace(
                [np.inf, -np.inf],
                np.nan,
            ).dropna()

            self._update_value_frequencies(
                column_name,
                values,
            )

        self.sqlite_connection.commit()

    def _update_categorical_statistics(
        self,
        chunk: pd.DataFrame,
    ) -> None:
        """Update categorical frequencies using SQLite."""

        if not self.config.include_categorical_statistics:
            return

        if not self.categorical_columns:
            return

        self._initialize_database()

        for column_name in self.categorical_columns:

            values = chunk[column_name].dropna()

            self._update_value_frequencies(
                column_name,
                values,
            )

        self.sqlite_connection.commit()

    def _update_unique_counts(
        self,
        chunk: pd.DataFrame,
    ) -> None:
        """Update exact or approximate unique counts."""

        if self.config.unique_count_mode == "exact":

            self._initialize_database()

            for column_name in self.column_names:

                records = (
                    (
                        column_name,
                        self._serialize_value(value),
                    )
                    for value in chunk[column_name].dropna()
                )

                self.sqlite_connection.executemany(
                    """
                    INSERT OR IGNORE INTO exact_unique_values (
                        column_name,
                        value
                    )
                    VALUES (?, ?)
                    """,
                    records,
                )

            self.sqlite_connection.commit()

        else:
            for column_name, sketch in self.unique_sketches.items():

                for value in chunk[column_name].dropna():
                    sketch.update(
                        self._serialize_value(value)
                    )

    def update(self, chunk: pd.DataFrame) -> None:
        """Process one chunk and update all statistics."""

        if self.finalized:
            raise RuntimeError(
                "Cannot update statistics after finalization."
            )

        if not isinstance(chunk, pd.DataFrame):
            raise TypeError("Chunk must be a pandas DataFrame.")

        if not self.initialized:
            self._initialize_schema(chunk)

        self._validate_chunk(chunk)

        try:
            self._update_numeric_statistics(chunk)
            self._update_numeric_values(chunk)
            self._update_numeric_frequencies(chunk)
            self._update_categorical_statistics(chunk)
            self._update_unique_counts(chunk)

        except Exception:
            # The in-memory and SQLite states may be partially updated.
            # Prevent further use of an inconsistent analyzer.
            self.finalized = True
            self.close()
            raise

    def _get_unique_count(self, column_name: str) -> int:
        """Return the exact or approximate unique count."""

        if self.config.unique_count_mode == "exact":

            cursor = self.sqlite_connection.execute(
                """
                SELECT COUNT(*)
                FROM exact_unique_values
                WHERE column_name = ?
                """,
                (column_name,),
            )

            return cursor.fetchone()[0]

        return int(
            self.unique_sketches[column_name].count()
        )

    def _get_mode_statistics(
        self,
        column_name: str,
    ) -> tuple[object | None, int]:
        """Return the most frequent numerical value.

        When multiple values have the same frequency, the smallest
        numerical value is selected, matching Pandas Series.mode().
        """

        if self.sqlite_connection is None:
            return None, 0

        cursor = self.sqlite_connection.execute(
            """
            SELECT MAX(frequency)
            FROM column_value_frequencies
            WHERE column_name = ?
            """,
            (column_name,),
        )

        max_frequency = cursor.fetchone()[0]

        if max_frequency is None:
            return None, 0

        cursor = self.sqlite_connection.execute(
            """
            SELECT value
            FROM column_value_frequencies
            WHERE column_name = ?
              AND frequency = ?
            """,
            (column_name, max_frequency),
        )

        tied_values = [
            pickle.loads(row[0])
            for row in cursor.fetchall()
        ]

        mode = min(tied_values)

        return mode, int(max_frequency)

    def _get_percentile(
        self,
        column_name: str,
        percentile: float,
        count: int,
    ) -> float | None:
        """Calculate an exact percentile using SQLite and linear interpolation."""

        if count == 0:
            return None

        if self.sqlite_connection is None:
            return None

        if not 0.0 <= percentile <= 1.0:
            raise ValueError("Percentile must be between 0 and 1.")

        position = (count - 1) * percentile

        lower_index = math.floor(position)
        upper_index = math.ceil(position)

        cursor = self.sqlite_connection.execute(
            """
            SELECT value
            FROM numeric_column_values
            WHERE column_name = ?
            ORDER BY value
            LIMIT 2 OFFSET ?
            """,
            (column_name, lower_index),
        )

        rows = cursor.fetchall()

        if not rows:
            raise RuntimeError(
                f"Unable to calculate percentile for '{column_name}'."
            )

        lower_value = float(rows[0][0])

        if lower_index == upper_index:
            return lower_value

        if len(rows) < 2:
            raise RuntimeError(
                f"Unable to calculate percentile for '{column_name}'."
            )

        upper_value = float(rows[1][0])

        fraction = position - lower_index

        return (
            lower_value
            + (upper_value - lower_value) * fraction
        )

    def _get_categorical_statistics(
        self,
        column_name: str,
    ) -> dict:
        """Return dataset-wide categorical statistics."""

        empty_statistics = {
            "count": 0,
            "unique": 0,
            "top": None,
            "freq": 0,
            "mode": None,
            "value_counts": {},
            "top_values": {},
        }

        if self.sqlite_connection is None:
            return empty_statistics

        aggregate_cursor = self.sqlite_connection.execute(
            """
            SELECT
                COUNT(*),
                COALESCE(SUM(frequency), 0)
            FROM column_value_frequencies
            WHERE column_name = ?
            """,
            (column_name,),
        )

        unique_count, total_count = aggregate_cursor.fetchone()

        if unique_count == 0:
            return empty_statistics

        cursor = self.sqlite_connection.execute(
            """
            SELECT value, frequency
            FROM column_value_frequencies
            WHERE column_name = ?
            ORDER BY frequency DESC, rowid ASC
            LIMIT ?
            """,
            (
                column_name,
                self.TOP_CATEGORIES_LIMIT,
            ),
        )

        value_counts = {
            pickle.loads(serialized_value): int(frequency)
            for serialized_value, frequency in cursor.fetchall()
        }

        mode = next(iter(value_counts))

        return {
            "count": int(total_count),
            "unique": int(unique_count),
            "top": mode,
            "freq": value_counts[mode],
            "mode": mode,
            "value_counts": value_counts,
            "top_values": value_counts.copy(),
        }

    def _get_numeric_statistics(
        self,
        column_name: str,
    ) -> dict:
        """Return numerical statistics accumulated across chunks."""

        state = self.numeric_states[column_name]

        count = state["count"]

        if count == 0:
            return {
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

        variance = (
            state["m2"] / (count - 1)
            if count > 1
            else None
        )

        standard_deviation = (
            math.sqrt(max(variance, 0.0))
            if variance is not None
            else None
        )

        first_quartile = self._get_percentile(
            column_name,
            0.25,
            count,
        )

        median = self._get_percentile(
            column_name,
            0.50,
            count,
        )

        third_quartile = self._get_percentile(
            column_name,
            0.75,
            count,
        )

        mode, _ = self._get_mode_statistics(column_name)

        return {
            "count": count,
            "mean": state["mean"],
            "std": standard_deviation,
            "min": state["min"],
            "25%": first_quartile,
            "50%": median,
            "75%": third_quartile,
            "max": state["max"],
            "variance": variance,
            "mode": mode,
        }

    def finalize(
        self,
        column_profiles: list[ColumnProfile],
    ) -> list[ColumnProfile]:
        """Finalize statistics and populate column profiles."""

        if self.finalized:
            raise RuntimeError(
                "Analyzer has already been finalized."
            )

        if not self.initialized:
            raise RuntimeError(
                "Cannot finalize before processing a chunk."
            )

        if [profile.name for profile in column_profiles] != self.column_names:
            raise ValueError(
                "Column profiles do not match the initialized schema."
            )

        self.finalized = True

        try:
            results = []

            for profile in column_profiles:
                updated_profile = replace(profile)

                updated_profile.unique_values = (
                    self._get_unique_count(profile.name)
                )

                if (
                    self.config.include_numeric_statistics
                    and profile.name in self.numeric_states
                ):
                    updated_profile.numeric_statistics = (
                        self._get_numeric_statistics(profile.name)
                    )

                if (
                    self.config.include_categorical_statistics
                    and profile.name in self.categorical_columns
                ):
                    updated_profile.categorical_statistics = (
                        self._get_categorical_statistics(profile.name)
                    )

                results.append(updated_profile)

            return results

        finally:
            self.close()

    def close(self) -> None:
        """Close SQLite connection and remove temporary database."""

        if self.sqlite_connection is not None:
            self.sqlite_connection.close()
            self.sqlite_connection = None

        if self.database_path is not None:
            try:
                os.remove(self.database_path)
            except FileNotFoundError:
                pass

            self.database_path = None
