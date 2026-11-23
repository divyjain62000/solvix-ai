from pathlib import Path

from backend.core.ingestion.models import (
    DatasetMetadata,
    IngestionResult,
)

from backend.core.profiling.models import (
    DatasetProfile,
    ProfilingConfig,
)
from backend.core.profiling.service import ProfilingService
from backend.infrastructure.database.repositories.dataset_metadata_repository import (
    DatasetMetadataRepository,
)
from backend.infrastructure.database.repositories.profile_repository import (
    ProfileRepository,
)
from backend.infrastructure.storage.dataset_repository import (
    DatasetRepository,
)


class DatasetProfilingService:
    """Coordinate dataset loading, profiling, and profile persistence."""

    def __init__(
        self,
        profiling_config: ProfilingConfig | None = None,
        dataset_repository: DatasetRepository | None = None,
        metadata_repository: DatasetMetadataRepository | None = None,
        profile_repository: ProfileRepository | None = None,
    ) -> None:
        self.profiling_service = ProfilingService(profiling_config)

        self.dataset_repository = dataset_repository or DatasetRepository()

        self.metadata_repository = metadata_repository or DatasetMetadataRepository()

        self.profile_repository = profile_repository or ProfileRepository()

    async def profile_dataset(
        self,
        dataset_id: str,
        version_id: str | None = None,
    ) -> DatasetProfile:
        """Profile a dataset version and persist its profile."""

        # 1. Retrieve dataset metadata
        metadata = await self.metadata_repository.get_by_id(dataset_id)

        if metadata is None:
            raise ValueError(f"Dataset not found: {dataset_id}")

        # 2. Resolve the version to profile
        target_version_id = (
            version_id if version_id is not None else metadata.active_version_id
        )

        if not target_version_id or not target_version_id.strip():
            raise ValueError("Dataset version ID cannot be empty.")

        # 3. Verify that the requested version exists
        if not self.dataset_repository.version_exists(
            dataset_id,
            target_version_id,
        ):
            raise FileNotFoundError(
                f"Dataset version not found: " f"{dataset_id}/{target_version_id}"
            )

        # 4. Load the dataset according to its processing mode
        if metadata.processing_mode == "in_memory":
            dataframe = self.dataset_repository.load_dataframe(
                dataset_id,
                target_version_id,
            )
            chunks = None

        elif metadata.processing_mode == "chunked":
            chunk_size = metadata.ingestion_config.get("chunk_size", 100_000)

            if type(chunk_size) is not int or chunk_size <= 0:
                raise ValueError("Invalid chunk_size in ingestion configuration.")

            dataframe = None
            chunks = self.dataset_repository.load_chunks(
                dataset_id,
                target_version_id,
                chunk_size,
            )

        else:
            raise ValueError(f"Unsupported processing mode: {metadata.processing_mode}")

        # 5. Reconstruct the runtime ingestion result
        ingestion_metadata = DatasetMetadata(
            filename=metadata.filename,
            file_size_bytes=metadata.file_size_bytes,
            processing_mode=metadata.processing_mode,
        )

        ingestion_result = IngestionResult(
            dataset_id=dataset_id,
            filename=metadata.filename,
            file_path=Path(metadata.filename),
            metadata=ingestion_metadata,
            dataframe=dataframe,
            chunks=chunks,
        )

        # 6. Generate the profile
        profile = self.profiling_service.profile(ingestion_result)

        # 7. Persist the profile against the exact dataset version
        await self.profile_repository.save(
            profile=profile,
            version_id=target_version_id,
        )

        # 8. Return the generated profile
        return profile
