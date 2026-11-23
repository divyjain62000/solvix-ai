
import hashlib
import math
import os
import pickle
import sqlite3
import tempfile

from dataclasses import replace
from datetime import date, datetime
from numbers import Integral, Real

import pandas as pd

from backend.core.profiling.models import (
    ColumnProfile,
    ProfilingConfig,
    QualityAnalysisResult,
)


class ChunkedQualityAnalyzer:
    """Incrementally analyzes dataset quality across DataFrame chunks."""

    def __init__(self, config: ProfilingConfig):
        self.config = config

        self.total_rows = 0
        self.total_missing_values = 0
        self.duplicate_rows = 0

        self.column_names: list[str] = []
        self.missing_counts: dict[str, int] = {}

        self.initialized = False
        self.finalized = False

        self.sqlite_connection: sqlite3.Connection | None = None
        self.database_path: str | None = None

        if self.config.include_duplicate_rows:
            self._initialize_database()

    def _initialize_database(self) -> None:
        """Create an isolated temporary SQLite database."""

        file_descriptor, self.database_path = tempfile.mkstemp(
            prefix="solvixai_profile_",
            suffix=".sqlite3",
        )

        os.close(file_descriptor)

        try:
            self.sqlite_connection = sqlite3.connect(
                self.database_path
            )

            self.sqlite_connection.execute(
                """
                CREATE TABLE seen_rows (
                    row_hash BLOB NOT NULL,
                    row_payload BLOB NOT NULL,
                    PRIMARY KEY (row_hash, row_payload)
                ) WITHOUT ROWID
                """
            )

            self.sqlite_connection.commit()

        except Exception:
            self._cleanup_database()
            raise

    def update(self, chunk: pd.DataFrame) -> None:
        """Process one chunk and update accumulated quality metrics."""

        if self.finalized:
            raise RuntimeError(
                "Cannot update analyzer after finalization"
            )

        if not isinstance(chunk, pd.DataFrame):
            raise TypeError("chunk must be a pandas DataFrame")

        if not chunk.columns.is_unique:
            raise ValueError("Duplicate column names are not supported")

        if not self.initialized:
            self._initialize_schema(chunk)
        else:
            self._validate_schema(chunk)

        if self.config.include_missing_values:
            self._update_missing_values(chunk)

        if self.config.include_duplicate_rows:
            self._update_duplicates(chunk)

        self.total_rows += len(chunk)

    def _initialize_schema(self, chunk: pd.DataFrame) -> None:
        """Initialize column tracking from the first chunk."""

        if len(chunk.columns) == 0:
            raise ValueError("Dataset must contain at least one column")

        self.column_names = list(chunk.columns)

        self.missing_counts = {
            column_name: 0
            for column_name in self.column_names
        }

        self.initialized = True

    def _validate_schema(self, chunk: pd.DataFrame) -> None:
        """Validate column names and ordering across chunks."""

        if list(chunk.columns) != self.column_names:
            raise ValueError(
                "Schema mismatch: chunk columns do not match "
                "the initial dataset schema"
            )

    def _update_missing_values(self, chunk: pd.DataFrame) -> None:
        """Increment missing-value counters."""

        missing_counts = chunk.isna().sum()

        for column_name, count in missing_counts.items():
            self.missing_counts[column_name] += int(count)

            self.total_missing_values += int(count)

    def _normalize_value(self, value):
        """Normalize values before row serialization."""

        try:
            missing = pd.isna(value)

            if isinstance(missing, (bool,)) and missing:
                return ("missing",)

            # Handle NumPy boolean scalar results without importing NumPy.
            if type(missing).__name__ == "bool_" and bool(missing):
                return ("missing",)

        except (TypeError, ValueError):
            pass

        # Normalize NumPy scalar values into Python scalar values.
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            try:
                value = value.item()
            except (ValueError, TypeError):
                pass

        if isinstance(value, bool):
            return ("boolean", value)

        # Treat equivalent integer and floating-point values consistently.
        if isinstance(value, Integral):
            return ("number", int(value))

        if isinstance(value, Real):
            numeric_value = float(value)

            if math.isfinite(numeric_value):
                if numeric_value.is_integer():
                    return ("number", int(numeric_value))

                return ("number", numeric_value.hex())

            return ("number", repr(numeric_value))

        if isinstance(value, (pd.Timestamp, datetime, date)):
            return ("datetime", value.isoformat())

        if isinstance(value, bytes):
            return ("bytes", value)

        if isinstance(value, str):
            return ("string", value)

        return (
            "object",
            type(value).__module__,
            type(value).__qualname__,
            repr(value),
        )

    def _update_duplicates(self, chunk: pd.DataFrame) -> None:
        """Detect duplicate rows using disk-backed SQLite storage."""

        if self.sqlite_connection is None:
            raise RuntimeError("SQLite database is not initialized")

        records = []

        for row in chunk.itertuples(index=False, name=None):
            normalized_row = tuple(
                self._normalize_value(value)
                for value in row
            )

            row_payload = pickle.dumps(
                normalized_row,
                protocol=5,
            )

            row_hash = hashlib.sha256(row_payload).digest()

            records.append((row_hash, row_payload))

        with self.sqlite_connection:
            for row_hash, row_payload in records:
                cursor = self.sqlite_connection.execute(
                    """
                    INSERT OR IGNORE INTO seen_rows
                    (row_hash, row_payload)
                    VALUES (?, ?)
                    """,
                    (row_hash, row_payload),
                )

                if cursor.rowcount == 0:
                    self.duplicate_rows += 1

    def finalize(
        self,
        column_profiles: list[ColumnProfile],
    ) -> QualityAnalysisResult:
        """Return accumulated quality results and release resources."""

        if self.finalized:
            raise RuntimeError("Analyzer has already been finalized")

        self.finalized = True

        try:
            if [profile.name for profile in column_profiles] != self.column_names:
                raise ValueError(
                    "Column profiles do not match the analyzed schema"
                )

            profiles = []

            for profile in column_profiles:
                updated_profile = replace(profile)

                updated_profile.total_values = self.total_rows

                if self.config.include_missing_values:
                    missing_count = self.missing_counts[profile.name]

                    updated_profile.missing_values = missing_count

                    updated_profile.missing_percentage = (
                        missing_count / self.total_rows * 100
                        if self.total_rows > 0
                        else 0.0
                    )
                else:
                    updated_profile.missing_values = 0
                    updated_profile.missing_percentage = 0.0

                profiles.append(updated_profile)

            return QualityAnalysisResult(
                columns=profiles,
                total_missing_values=(
                    self.total_missing_values
                    if self.config.include_missing_values
                    else 0
                ),
                duplicate_rows=(
                    self.duplicate_rows
                    if self.config.include_duplicate_rows
                    else 0
                ),
            )

        finally:
            self._cleanup_database()

    def _cleanup_database(self) -> None:
        """Close SQLite connection and remove temporary database."""

        if self.sqlite_connection is not None:
            self.sqlite_connection.close()
            self.sqlite_connection = None

        if self.database_path is not None:
            if os.path.exists(self.database_path):
                os.remove(self.database_path)

            self.database_path = None

    def close(self) -> None:
        """Release resources when processing is interrupted."""

        self.finalized = True
        self._cleanup_database()
