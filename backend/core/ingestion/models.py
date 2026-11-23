from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Iterator
from datetime import datetime

import pandas as pd


@dataclass
class IngestionConfig:
    """Configuration for CSV ingestion."""

    mode: Literal["auto", "in_memory", "chunked"] = "auto"

    max_file_size_mb: int | None = 500
    in_memory_threshold_mb: int = 100
    chunk_size: int = 100_000

    encoding: str = "utf-8"
    delimiter: str = ","

    storage_dir: Path = Path("data/raw")

    def __post_init__(self):
        if self.mode not in {"auto", "in_memory", "chunked"}:
            raise ValueError(f"Unsupported ingestion mode: {self.mode}")

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if self.in_memory_threshold_mb <= 0:
            raise ValueError("in_memory_threshold_mb must be greater than zero")

        if self.max_file_size_mb is not None and self.max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be greater than zero")


@dataclass
class DatasetMetadata:
    """Metadata describing an ingested dataset."""

    filename: str
    file_size_bytes: int

    row_count: int | None = None
    column_count: int = 0

    columns: list[str] = field(default_factory=list)
    data_types: dict[str, str] = field(default_factory=dict)

    processing_mode: str = "in_memory"


@dataclass
class IngestionResult:
    """Standard result returned after successful ingestion."""

    dataset_id: str
    filename: str
    file_path: Path

    metadata: DatasetMetadata

    # Populated only when using in-memory ingestion.
    dataframe: pd.DataFrame | None = None

    # Populated only for chunked ingestion.
    chunks: Iterator[pd.DataFrame] | None = None


@dataclass
class PersistentDatasetMetadata:
    """Persistent metadata for an ingested dataset."""

    dataset_id: str
    filename: str
    original_file_key: str
    file_size_bytes: int
    processing_mode: Literal["in_memory", "chunked"]
    ingestion_config: dict[str, Any] = field(default_factory=dict)
    active_version_id: str = "v1"
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.dataset_id:
            raise ValueError("dataset_id cannot be empty")

        if not self.filename:
            raise ValueError("filename cannot be empty")

        if not self.original_file_key:
            raise ValueError("original_file_key cannot be empty")

        if self.file_size_bytes < 0:
            raise ValueError("file_size_bytes cannot be negative")

        if self.processing_mode not in {"in_memory", "chunked"}:
            raise ValueError("Unsupported processing mode")

        if not self.active_version_id:
            raise ValueError("active_version_id cannot be empty")
