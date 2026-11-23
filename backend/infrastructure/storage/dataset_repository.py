
"""Repository for dataset file persistence."""

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from backend.infrastructure.storage.dataset_storage import DatasetStorage


class DatasetRepository:
    """Coordinate dataset file persistence."""

    def __init__(self, storage: DatasetStorage | None = None) -> None:
        self.storage = storage or DatasetStorage()

    def save_original(
        self,
        source_path: Path,
        dataset_id: str,
    ) -> str:
        """Persist the original uploaded CSV file."""

        return self.storage.store_original(
            source_path=source_path,
            dataset_id=dataset_id,
        )

    def save_version(
        self,
        dataframe: pd.DataFrame,
        dataset_id: str,
        version_id: str,
    ) -> str:
        """Persist a DataFrame as a Parquet dataset version."""

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")

        if len(dataframe.columns) == 0:
            raise ValueError("Cannot save a DataFrame without columns")

        key = self.storage.get_version_key(dataset_id, version_id)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "data.parquet"

            dataframe.to_parquet(
                temp_path,
                engine="pyarrow",
                index=False,
            )

            return self.storage.backend.upload_file(
                source=temp_path,
                key=key,
            )

    def save_version_from_chunks(
        self,
        chunks: Iterator[pd.DataFrame],
        dataset_id: str,
        version_id: str,
    ) -> str:
        """Persist DataFrame chunks incrementally as a Parquet version."""

        key = self.storage.get_version_key(dataset_id, version_id)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "data.parquet"

            iterator = iter(chunks)
            first_chunk = next(iterator, None)

            if first_chunk is None:
                raise ValueError(
                    "No chunks provided for saving the dataset version"
                )

            if not isinstance(first_chunk, pd.DataFrame):
                raise TypeError("Every chunk must be a pandas DataFrame")

            if len(first_chunk.columns) == 0:
                raise ValueError("Cannot save chunks without columns")

            expected_columns = list(first_chunk.columns)

            first_table = pa.Table.from_pandas(
                first_chunk,
                preserve_index=False,
            )

            writer = pq.ParquetWriter(
                temp_path,
                first_table.schema,
            )

            try:
                writer.write_table(first_table)

                for chunk in iterator:
                    if not isinstance(chunk, pd.DataFrame):
                        raise TypeError(
                            "Every chunk must be a pandas DataFrame"
                        )

                    if list(chunk.columns) != expected_columns:
                        raise ValueError(
                            "All chunks must have identical columns "
                            "in the same order"
                        )

                    table = pa.Table.from_pandas(
                        chunk,
                        schema=first_table.schema,
                        preserve_index=False,
                        safe=True,
                    )

                    writer.write_table(table)

            finally:
                writer.close()

            return self.storage.backend.upload_file(
                source=temp_path,
                key=key,
            )

    def load_dataframe(
        self,
        dataset_id: str,
        version_id: str,
    ) -> pd.DataFrame:
        """Load a stored Parquet dataset version."""

        key = self.storage.get_version_key(dataset_id, version_id)

        if not self.storage.backend.exists(key):
            raise FileNotFoundError(
                f"Dataset version not found: {key}"
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "data.parquet"

            self.storage.backend.download_file(
                key=key,
                destination=temp_path,
            )

            return pd.read_parquet(
                temp_path,
                engine="pyarrow",
            )

    def load_chunks(
        self,
        dataset_id: str,
        version_id: str,
        chunk_size: int,
    ) -> Iterator[pd.DataFrame]:
        """Load a Parquet dataset version in chunks."""

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        key = self.storage.get_version_key(dataset_id, version_id)

        if not self.storage.backend.exists(key):
            raise FileNotFoundError(
                f"Dataset version not found: {key}"
            )

        def generate_chunks() -> Iterator[pd.DataFrame]:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / "data.parquet"

                self.storage.backend.download_file(
                    key=key,
                    destination=temp_path,
                )

                parquet_file = None

                try:
                    parquet_file = pq.ParquetFile(temp_path)

                    for batch in parquet_file.iter_batches(
                        batch_size=chunk_size
                    ):
                        yield batch.to_pandas()

                finally:
                    if parquet_file is not None:
                        parquet_file.close()

        return generate_chunks()

    def delete_version(
        self,
        dataset_id: str,
        version_id: str,
    ) -> None:
        """Delete a specific processed dataset version."""

        key = self.storage.get_version_key(dataset_id, version_id)

        if not self.storage.backend.exists(key):
            raise FileNotFoundError(
                f"Dataset version not found: {key}"
            )

        self.storage.delete_file(key)

    def version_exists(
        self,
        dataset_id: str,
        version_id: str,
    ) -> bool:
        """Check whether a dataset version exists."""

        key = self.storage.get_version_key(dataset_id, version_id)

        return self.storage.backend.exists(key)

    def delete_original(self, dataset_id: str) -> None:
        """Delete the original uploaded dataset."""

        key = self.storage.get_original_key(dataset_id)

        if not self.storage.backend.exists(key):
            raise FileNotFoundError(
                f"Original dataset not found: {key}"
            )

        self.storage.delete_file(key)
