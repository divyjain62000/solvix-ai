
"""Handler for detecting and removing duplicate rows."""

import pickle
import sqlite3
import tempfile

from collections.abc import Iterable, Iterator
from uuid import uuid4

import pandas as pd

from backend.core.cleaning.exceptions import DuplicateHandlingError
from backend.core.cleaning.models import (
    CleaningConfig,
    CleaningOperationResult,
    DuplicateAnalysisResult,
)
from backend.core.profiling.models import DatasetProfile


class DuplicateHandler:
    """Analyze and remove duplicate rows from datasets."""

    def analyze(
        self,
        profile: DatasetProfile,
    ) -> DuplicateAnalysisResult:
        """Analyze duplicates using an existing dataset profile."""

        if not isinstance(profile, DatasetProfile):
            raise DuplicateHandlingError("Invalid dataset profile.")

        duplicate_percentage = (
            profile.duplicate_rows / profile.total_rows * 100
            if profile.total_rows > 0
            else 0.0
        )

        return DuplicateAnalysisResult(
            total_rows=profile.total_rows,
            duplicate_rows=profile.duplicate_rows,
            duplicate_percentage=duplicate_percentage,
        )

    def clean(
        self,
        dataframe: pd.DataFrame,
        config: CleaningConfig,
    ) -> tuple[pd.DataFrame, CleaningOperationResult]:
        """Remove duplicate rows using the configured keep strategy."""

        if not isinstance(dataframe, pd.DataFrame):
            raise DuplicateHandlingError(
                "Input must be a Pandas DataFrame."
            )

        if not isinstance(config, CleaningConfig):
            raise DuplicateHandlingError(
                "Invalid cleaning configuration."
            )

        cleaned_df = dataframe.copy(deep=True)
        rows_before = len(cleaned_df)

        if not config.remove_duplicates:
            return cleaned_df, self._build_result(
                rows_before=rows_before,
                rows_after=rows_before,
                config=config,
            )

        keep = (
            config.duplicate_keep
            if config.duplicate_keep is not None
            else False
        )

        try:
            cleaned_df = cleaned_df.drop_duplicates(keep=keep)
        except Exception as exc:
            raise DuplicateHandlingError(
                f"Failed to remove duplicate rows: {exc}"
            ) from exc

        rows_after = len(cleaned_df)

        result = self._build_result(
            rows_before=rows_before,
            rows_after=rows_after,
            config=config,
        )

        return cleaned_df, result

    def clean_chunks(
        self,
        chunks: Iterable[pd.DataFrame],
        config: CleaningConfig,
    ) -> tuple[Iterator[pd.DataFrame], CleaningOperationResult]:
        """Remove duplicate rows across a chunked dataset."""

        if chunks is None:
            raise DuplicateHandlingError("Chunks cannot be None.")

        if not isinstance(config, CleaningConfig):
            raise DuplicateHandlingError(
                "Invalid cleaning configuration."
            )

        try:
            iterator = iter(chunks)
        except TypeError as exc:
            raise DuplicateHandlingError(
                "Chunks must be an iterable of Pandas DataFrames."
            ) from exc

        result = self._build_result(
            rows_before=0,
            rows_after=0,
            config=config,
        )

        rows_before = 0
        rows_after = 0

        def generate() -> Iterator[pd.DataFrame]:
            nonlocal rows_before
            nonlocal rows_after

            # No disk staging is required when duplicate removal is disabled.
            if not config.remove_duplicates:
                for chunk in iterator:
                    if not isinstance(chunk, pd.DataFrame):
                        raise DuplicateHandlingError(
                            "Every chunk must be a Pandas DataFrame."
                        )

                    rows_before += len(chunk)
                    rows_after += len(chunk)

                    yield chunk.copy(deep=True)

                return

            try:
                with tempfile.TemporaryDirectory(
                    prefix="solvixai_duplicates_"
                ) as temp_dir:

                    database_path = f"{temp_dir}/duplicates.sqlite"
                    connection = sqlite3.connect(database_path)

                    try:
                        connection.execute(
                            """
                            CREATE TABLE staged_chunks (
                                chunk_id INTEGER PRIMARY KEY,
                                start_ordinal INTEGER NOT NULL,
                                row_count INTEGER NOT NULL,
                                payload BLOB NOT NULL
                            )
                            """
                        )

                        connection.execute(
                            """
                            CREATE TABLE duplicate_groups (
                                group_id INTEGER PRIMARY KEY,
                                row_hash TEXT NOT NULL,
                                row_payload BLOB NOT NULL,
                                first_ordinal INTEGER NOT NULL,
                                last_ordinal INTEGER NOT NULL,
                                occurrence_count INTEGER NOT NULL
                            )
                            """
                        )

                        connection.execute(
                            """
                            CREATE INDEX idx_duplicate_hash
                            ON duplicate_groups(row_hash)
                            """
                        )

                        columns = None
                        chunk_id = 0
                        current_ordinal = 0

                        # Phase 1: Stage chunks and identify global duplicates.
                        for chunk in iterator:
                            if not isinstance(chunk, pd.DataFrame):
                                raise DuplicateHandlingError(
                                    "Every chunk must be a Pandas DataFrame."
                                )

                            if columns is None:
                                columns = list(chunk.columns)
                            elif list(chunk.columns) != columns:
                                raise DuplicateHandlingError(
                                    "All chunks must have identical columns "
                                    "in the same order."
                                )

                            chunk_size = len(chunk)
                            chunk_start = current_ordinal

                            rows_before += chunk_size

                            # Preserve the original DataFrame, including
                            # its index and dtypes.
                            payload = sqlite3.Binary(
                                pickle.dumps(
                                    chunk,
                                    protocol=pickle.HIGHEST_PROTOCOL,
                                )
                            )

                            connection.execute(
                                """
                                INSERT INTO staged_chunks (
                                    chunk_id,
                                    start_ordinal,
                                    row_count,
                                    payload
                                )
                                VALUES (?, ?, ?, ?)
                                """,
                                (
                                    chunk_id,
                                    chunk_start,
                                    chunk_size,
                                    payload,
                                ),
                            )

                            if chunk_size:
                                try:
                                    hashes = (
                                        pd.util.hash_pandas_object(
                                            chunk,
                                            index=False,
                                        )
                                        .to_numpy(dtype="uint64")
                                    )
                                except Exception as exc:
                                    raise DuplicateHandlingError(
                                        "Unable to calculate row hashes: "
                                        f"{exc}"
                                    ) from exc

                                for position, row in enumerate(
                                    chunk.itertuples(
                                        index=False,
                                        name=None,
                                    )
                                ):
                                    ordinal = current_ordinal + position
                                    row_hash = str(int(hashes[position]))

                                    row_payload = pickle.dumps(
                                        tuple(row),
                                        protocol=pickle.HIGHEST_PROTOCOL,
                                    )

                                    candidates = connection.execute(
                                        """
                                        SELECT group_id, row_payload
                                        FROM duplicate_groups
                                        WHERE row_hash = ?
                                        """,
                                        (row_hash,),
                                    ).fetchall()

                                    matching_group = None

                                    for group_id, candidate_payload in candidates:
                                        candidate_row = pickle.loads(
                                            candidate_payload
                                        )

                                        if self._rows_equal(
                                            candidate_row,
                                            tuple(row),
                                            columns,
                                        ):
                                            matching_group = group_id
                                            break

                                    if matching_group is not None:
                                        connection.execute(
                                            """
                                            UPDATE duplicate_groups
                                            SET
                                                last_ordinal = ?,
                                                occurrence_count =
                                                    occurrence_count + 1
                                            WHERE group_id = ?
                                            """,
                                            (ordinal, matching_group),
                                        )
                                    else:
                                        connection.execute(
                                            """
                                            INSERT INTO duplicate_groups (
                                                row_hash,
                                                row_payload,
                                                first_ordinal,
                                                last_ordinal,
                                                occurrence_count
                                            )
                                            VALUES (?, ?, ?, ?, 1)
                                            """,
                                            (
                                                row_hash,
                                                sqlite3.Binary(row_payload),
                                                ordinal,
                                                ordinal,
                                            ),
                                        )

                            current_ordinal += chunk_size
                            chunk_id += 1

                            connection.commit()

                        # Phase 2: Select rows according to keep strategy.
                        if config.duplicate_keep == "first":
                            selected_query = """
                                SELECT first_ordinal AS selected_ordinal
                                FROM duplicate_groups
                            """

                        elif config.duplicate_keep == "last":
                            selected_query = """
                                SELECT last_ordinal AS selected_ordinal
                                FROM duplicate_groups
                            """

                        else:
                            selected_query = """
                                SELECT first_ordinal AS selected_ordinal
                                FROM duplicate_groups
                                WHERE occurrence_count = 1
                            """

                        # Phase 3: Replay original chunks in original order.
                        staged_cursor = connection.execute(
                            """
                            SELECT
                                start_ordinal,
                                row_count,
                                payload
                            FROM staged_chunks
                            ORDER BY chunk_id
                            """
                        )

                        for (
                            start_ordinal,
                            chunk_size,
                            payload,
                        ) in staged_cursor:

                            end_ordinal = start_ordinal + chunk_size

                            selected_ordinals = {
                                row[0]
                                for row in connection.execute(
                                    f"""
                                    SELECT selected_ordinal
                                    FROM ({selected_query})
                                    WHERE selected_ordinal >= ?
                                      AND selected_ordinal < ?
                                    """,
                                    (start_ordinal, end_ordinal),
                                ).fetchall()
                            }

                            original_chunk = pickle.loads(payload)

                            selected_positions = [
                                ordinal - start_ordinal
                                for ordinal in sorted(selected_ordinals)
                            ]

                            cleaned_chunk = original_chunk.iloc[
                                selected_positions
                            ].copy(deep=True)

                            rows_after += len(cleaned_chunk)

                            yield cleaned_chunk

                    finally:
                        connection.close()

            except DuplicateHandlingError:
                raise
            except Exception as exc:
                raise DuplicateHandlingError(
                    f"Failed to clean duplicate chunks: {exc}"
                ) from exc

        def tracked_generator() -> Iterator[pd.DataFrame]:
            yield from generate()

            result.rows_before = rows_before
            result.rows_after = rows_after
            result.rows_removed = rows_before - rows_after

        return tracked_generator(), result

    @staticmethod
    def _rows_equal(
        first_row: tuple,
        second_row: tuple,
        columns: list,
    ) -> bool:
        """Verify row equality when two rows have the same hash."""

        comparison = pd.DataFrame(
            [first_row, second_row],
            columns=columns,
        )

        return bool(
            comparison.duplicated(keep=False).all()
        )

    @staticmethod
    def _build_result(
        rows_before: int,
        rows_after: int,
        config: CleaningConfig,
    ) -> CleaningOperationResult:
        """Build duplicate-removal operation metadata."""

        return CleaningOperationResult(
            operation_id=str(uuid4()),
            operation_type="duplicate_removal",
            description="Duplicate rows removed.",
            affected_columns=[],
            rows_before=rows_before,
            rows_after=rows_after,
            rows_removed=rows_before - rows_after,
            values_changed=0,
            parameters={
                "keep": config.duplicate_keep,
                "subset": None,
            },
        )
