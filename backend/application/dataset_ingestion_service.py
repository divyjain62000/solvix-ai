"""Application service for dataset ingestion workflow."""

from pathlib import Path

from backend.core.ingestion.models import IngestionConfig, PersistentDatasetMetadata
from backend.core.ingestion.service import IngestionService
from backend.application.dataset_persistence_service import (
    DatasetPersistenceService,
)
from backend.application.dataset_profiling_service import (
    DatasetProfilingService,
)


class DatasetIngestionService:
    """Orchestrate dataset ingestion, persistence, and initial profiling."""

    def __init__(
        self,
        config: IngestionConfig | None = None,
        ingestion_service: IngestionService | None = None,
        persistence_service: DatasetPersistenceService | None = None,
        profiling_service: DatasetProfilingService | None = None,
    ) -> None:
        self.config = config or IngestionConfig()
        self.ingestion_service = ingestion_service or IngestionService(self.config)
        self.persistence_service = persistence_service or DatasetPersistenceService()
        self.profiling_service = profiling_service or DatasetProfilingService()

    async def ingest(
        self,
        file_path: Path | str,
    ) -> PersistentDatasetMetadata:
        """Ingest, persist, and initially profile a dataset."""

        ingestion_result = self.ingestion_service.ingest(file_path)

        metadata = await self.persistence_service.persist(
            result=ingestion_result,
            config=self.config,
        )

        await self.profiling_service.profile_dataset(
            dataset_id=metadata.dataset_id,
            version_id=metadata.active_version_id,
        )

        return metadata
