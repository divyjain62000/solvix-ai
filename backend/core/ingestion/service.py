from pathlib import Path
from uuid import uuid4
from typing import Any

import pandas as pd

from backend.core.ingestion.models import (
    IngestionConfig,
    IngestionResult,
    DatasetMetadata,
)

from backend.core.ingestion.validator import CSVValidator
from backend.core.ingestion.loader import CSVLoader


class IngestionService:

    def __init__(self, config: IngestionConfig):
        self.config = config

    def ingest(self, file_path: Path | str) -> IngestionResult:
        """
        1. Validate the CSV file.
        2. Select the processing mode.
        3. Load the CSV.
        4. Build metadata.
        5. Return IngestionResult.
        """

        file_path = Path(file_path)

        # Step 1: Validate CSV
        validation_result = CSVValidator.validate(
            file_path=file_path,
            max_file_size_mb=self.config.max_file_size_mb,
            encoding=self.config.encoding,
            delimiter=self.config.delimiter,
        )

        # Step 2: Select processing mode
        processing_mode = self._select_processing_mode(
            file_path=file_path,
            total_rows=validation_result["row_count"],
        )

        # Step 3: Load dataset
        dataframe = None
        chunks = None

        if processing_mode == "in_memory":

            dataframe = CSVLoader.load_in_memory(
                file_path=file_path,
                encoding=self.config.encoding,
                delimiter=self.config.delimiter,
            )

        else:

            chunks = CSVLoader.load_in_chunks(
                file_path=file_path,
                chunk_size=self.config.chunk_size,
                encoding=self.config.encoding,
                delimiter=self.config.delimiter,
            )

        # Step 4: Build metadata
        metadata = self._build_metadata(
            file_path=file_path,
            validation_result=validation_result,
            processing_mode=processing_mode,
            dataframe=dataframe,
        )

        # Step 5: Return result
        return IngestionResult(
            dataset_id=str(uuid4()),
            filename=file_path.name,
            file_path=file_path,
            metadata=metadata,
            dataframe=dataframe,
            chunks=chunks,
        )

    def _select_processing_mode(
        self,
        file_path: Path,
        total_rows: int,
    ) -> str:
        """
        Determine whether the dataset should be
        loaded in memory or in chunks.
        """

        # Explicit user configuration
        if self.config.mode == "in_memory":
            return "in_memory"

        if self.config.mode == "chunked":
            return "chunked"

        # Automatic mode selection
        memory_usage_bytes = CSVLoader.estimate_memory_usage(
            file_path=file_path,
            total_rows=total_rows,
            encoding=self.config.encoding,
            delimiter=self.config.delimiter,
        )

        memory_usage_mb = memory_usage_bytes / (1024 * 1024)

        if memory_usage_mb <= self.config.in_memory_threshold_mb:
            return "in_memory"

        return "chunked"

    def _build_metadata(
        self,
        file_path: Path,
        validation_result: dict[str, Any],
        processing_mode: str,
        dataframe: pd.DataFrame | None = None,
    ) -> DatasetMetadata:
        """
        Generate metadata for the ingested dataset.
        """

        # Get column data types when DataFrame is available.
        data_types = {}

        if dataframe is not None:
            data_types = {
                column: str(dtype) for column, dtype in dataframe.dtypes.items()
            }

        return DatasetMetadata(
            filename=file_path.name,
            file_size_bytes=validation_result["file_size_bytes"],
            row_count=validation_result["row_count"],
            column_count=validation_result["column_count"],
            columns=validation_result["headers"],
            data_types=data_types,
            processing_mode=processing_mode,
        )
