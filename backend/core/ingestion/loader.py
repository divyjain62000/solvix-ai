
from pathlib import Path
from typing import Iterator

import pandas as pd


class CSVLoader:
    """Loads CSV files into Pandas DataFrames."""

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Remove leading and trailing whitespace from column names."""

        df.columns = [str(column).strip() for column in df.columns]
        return df

    @staticmethod
    def _validate_chunk_size(chunk_size: int) -> None:
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or chunk_size <= 0
        ):
            raise ValueError("chunk_size must be a positive integer.")

    @staticmethod
    def load_in_memory(
        file_path: Path,
        encoding: str = "utf-8-sig",
        delimiter: str = ",",
        optimize: bool = False,
    ) -> pd.DataFrame:
        """Load the complete CSV file into memory."""

        file_path = Path(file_path)

        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(
            file_path,
            encoding=encoding,
            sep=delimiter,
            skipinitialspace=True,
        )

        df = CSVLoader._normalize_columns(df)

        if optimize:
            df = CSVLoader.optimize_dtypes(df)

        return df

    @staticmethod
    def get_chunk_iterator(
        file_path: Path,
        chunk_size: int = 100_000,
        encoding: str = "utf-8-sig",
        delimiter: str = ",",
    ) -> pd.io.parsers.TextFileReader:
        """Return a Pandas CSV chunk reader."""

        CSVLoader._validate_chunk_size(chunk_size)

        file_path = Path(file_path)

        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        return pd.read_csv(
            file_path,
            encoding=encoding,
            sep=delimiter,
            chunksize=chunk_size,
            skipinitialspace=True,
        )

    @staticmethod
    def load_in_chunks(
        file_path: Path,
        chunk_size: int = 100_000,
        encoding: str = "utf-8-sig",
        delimiter: str = ",",
        optimize: bool = False,
    ) -> Iterator[pd.DataFrame]:

        # Executes immediately
        CSVLoader._validate_chunk_size(chunk_size)

        reader = CSVLoader.get_chunk_iterator(
            file_path=file_path,
            chunk_size=chunk_size,
            encoding=encoding,
            delimiter=delimiter,
        )

        def generate():
            try:
                for chunk in reader:
                    chunk = CSVLoader._normalize_columns(chunk)

                    if optimize:
                        chunk = CSVLoader.optimize_dtypes(chunk)

                    yield chunk
            finally:
                reader.close()

        return generate()

    @staticmethod
    def estimate_memory_usage(
        file_path: Path,
        total_rows: int,
        sample_rows: int = 10_000,
        encoding: str = "utf-8-sig",
        delimiter: str = ",",
        safety_factor: float = 1.5,
    ) -> int:
        """
        Estimate total DataFrame memory usage in bytes.

        Uses a sample of the CSV and extrapolates its memory usage.
        The result is approximate and depends on sample representativeness.
        """

        if total_rows <= 0:
            raise ValueError("total_rows must be greater than zero.")

        if sample_rows <= 0:
            raise ValueError("sample_rows must be greater than zero.")

        if safety_factor < 1:
            raise ValueError("safety_factor must be at least 1.")

        file_path = Path(file_path)

        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        sample_df = pd.read_csv(
            file_path,
            encoding=encoding,
            sep=delimiter,
            nrows=sample_rows,
            skipinitialspace=True,
        )

        sample_df = CSVLoader._normalize_columns(sample_df)

        sampled_rows = len(sample_df)

        if sampled_rows == 0:
            raise ValueError("Cannot estimate memory for an empty CSV.")

        sample_memory = int(
            sample_df.memory_usage(deep=True).sum()
        )

        estimated_memory = (
            sample_memory / sampled_rows
        ) * total_rows * safety_factor

        return int(estimated_memory)

    @staticmethod
    def optimize_dtypes(
        df: pd.DataFrame,
        category_threshold: float = 0.5,
        max_categories: int = 100,
    ) -> pd.DataFrame:
        """
        Reduce DataFrame memory usage without intentionally
        losing numeric precision.
        """

        if not 0 < category_threshold <= 1:
            raise ValueError(
                "category_threshold must be between 0 and 1."
            )

        if (
            isinstance(max_categories, bool)
            or not isinstance(max_categories, int)
            or max_categories <= 0
        ):
            raise ValueError(
                "max_categories must be a positive integer."
            )

        optimized_df = df.copy()

        for column in optimized_df.columns:

            series = optimized_df[column]

            # Downcast integer columns without changing values.
            if pd.api.types.is_integer_dtype(series.dtype):
                optimized_df[column] = pd.to_numeric(
                    series,
                    downcast="integer",
                )

            # Downcast floating-point columns only when values
            # can be represented without losing precision.
            elif pd.api.types.is_float_dtype(series.dtype):
                downcasted = pd.to_numeric(
                    series,
                    downcast="float",
                )

                if series.equals(downcasted):
                    optimized_df[column] = downcasted

            # Convert suitable string/object columns to category.
            elif (
                pd.api.types.is_object_dtype(series.dtype)
                or isinstance(series.dtype, pd.StringDtype)
            ):
                unique_count = series.nunique(dropna=True)
                row_count = len(series)

                if row_count == 0:
                    continue

                unique_ratio = unique_count / row_count

                if (
                    unique_count > 0
                    and unique_count <= max_categories
                    and unique_ratio <= category_threshold
                ):
                    optimized_df[column] = series.astype("category")

        return optimized_df
