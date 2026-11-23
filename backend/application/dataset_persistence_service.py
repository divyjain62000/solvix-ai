
"""Coordinate dataset persistence across storage and database."""

from dataclasses import asdict

from backend.core.ingestion.models import (
    IngestionConfig,
    IngestionResult,
    PersistentDatasetMetadata,
)
from backend.infrastructure.database.repositories.dataset_metadata_repository import (
    DatasetMetadataRepository,
)
from backend.infrastructure.storage.dataset_repository import DatasetRepository


class DatasetPersistenceService:
    """Coordinate dataset persistence across storage and database."""

    def __init__(
        self,
        dataset_repository: DatasetRepository | None = None,
        metadata_repository: DatasetMetadataRepository | None = None,
    ) -> None:
        self.dataset_repository = (
            dataset_repository or DatasetRepository()
        )
        self.metadata_repository = (
            metadata_repository or DatasetMetadataRepository()
        )

    async def persist(
        self,
        result: IngestionResult,
        config: IngestionConfig,
    ) -> PersistentDatasetMetadata:
        """Persist the original dataset, initial version, and metadata."""

        original_saved = False
        version_saved = False

        try:
            # 1. Persist immutable original CSV.
            original_file_key = self.dataset_repository.save_original(
                source_path=result.file_path,
                dataset_id=result.dataset_id,
            )
            original_saved = True

            # 2. Persist initial dataset version as Parquet.
            version_id = "v1"

            if result.dataframe is not None:
                self.dataset_repository.save_version(
                    dataframe=result.dataframe,
                    dataset_id=result.dataset_id,
                    version_id=version_id,
                )

            elif result.chunks is not None:
                self.dataset_repository.save_version_from_chunks(
                    chunks=result.chunks,
                    dataset_id=result.dataset_id,
                    version_id=version_id,
                )

            else:
                raise ValueError(
                    "IngestionResult contains no dataset data"
                )

            version_saved = True

            # 3. Prepare persistent ingestion configuration.
            ingestion_config = asdict(config)
            ingestion_config["storage_dir"] = str(
                ingestion_config["storage_dir"]
            )

            # 4. Persist MongoDB metadata.
            metadata = PersistentDatasetMetadata(
                dataset_id=result.dataset_id,
                filename=result.filename,
                original_file_key=original_file_key,
                file_size_bytes=result.metadata.file_size_bytes,
                processing_mode=result.metadata.processing_mode,
                ingestion_config=ingestion_config,
                active_version_id=version_id,
            )

            await self.metadata_repository.save(metadata)

            return metadata

        except Exception as persistence_error:
            # Roll back files if persistence fails.
            if version_saved:
                try:
                    self.dataset_repository.delete_version(
                        result.dataset_id,
                        "v1",
                    )
                except Exception as cleanup_error:
                    persistence_error.add_note(
                        f"Failed to clean up initial version: "
                        f"{cleanup_error}"
                    )

            if original_saved:
                try:
                    self.dataset_repository.delete_original(
                        result.dataset_id
                    )
                except Exception as cleanup_error:
                    persistence_error.add_note(
                        f"Failed to clean up original dataset: "
                        f"{cleanup_error}"
                    )

            raise
