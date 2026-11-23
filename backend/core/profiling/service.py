from backend.core.profiling.models import (
    DatasetProfile,
    ProfilingConfig,
)
from backend.core.ingestion.models import IngestionResult

from backend.core.profiling.strategies.in_memory import (
    InMemoryProfilingStrategy,
)
from backend.core.profiling.strategies.chunked import (
    ChunkedProfilingStrategy,
)


class ProfilingService:
    """Orchestrates dataset profiling."""

    def __init__(self, config: ProfilingConfig | None = None):
        self.config = config

        self.in_memory_strategy = InMemoryProfilingStrategy(self.config)
        self.chunked_strategy = ChunkedProfilingStrategy(self.config)

    def profile(
        self,
        ingestion_result: IngestionResult,
    ) -> DatasetProfile:

        if ingestion_result is None:
            raise ValueError("ingestion_result cannot be None")

        has_dataframe = ingestion_result.dataframe is not None
        has_chunks = ingestion_result.chunks is not None

        if has_dataframe == has_chunks:
            raise ValueError(
                "IngestionResult must contain exactly one of "
                "'dataframe' or 'chunks'."
            )

        if ingestion_result.metadata.processing_mode == "in_memory":
            if not has_dataframe:
                raise ValueError("In-memory mode requires a dataframe.")

            return self.in_memory_strategy.profile(
                ingestion_result.dataframe,
                ingestion_result.dataset_id,
                self.config,
            )

        if ingestion_result.metadata.processing_mode == "chunked":
            if not has_chunks:
                raise ValueError("Chunked mode requires a chunks iterator.")

            return self.chunked_strategy.profile(
                ingestion_result.chunks,
                ingestion_result.dataset_id,
                self.config,
            )

        raise ValueError(
            f"Unsupported processing mode: "
            f"{ingestion_result.metadata.processing_mode}"
        )
