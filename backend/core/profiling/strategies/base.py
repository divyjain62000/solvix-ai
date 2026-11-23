
from abc import ABC, abstractmethod
from typing import Any

from backend.core.profiling.models import (
    DatasetProfile,
    ProfilingConfig,
)


class ProfilingStrategy(ABC):
    """Base interface for dataset profiling strategies."""

    @abstractmethod
    def profile(
        self,
        data: Any,
        dataset_id: str,
        config: ProfilingConfig | None = None,
    ) -> DatasetProfile:
        """
        Execute profiling and return the complete dataset profile.

        Args:
            data: DataFrame or iterator of DataFrames.
            dataset_id: Unique dataset identifier.
            config: Profiling configuration.

        Returns:
            DatasetProfile: Complete profiling results.
        """
        raise NotImplementedError
