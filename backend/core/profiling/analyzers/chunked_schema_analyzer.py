
import hashlib
import json

from narwhals import series
import pandas as pd
from datasketch import HyperLogLog

from backend.core.profiling.models import (
    ColumnProfile,
    ProfilingConfig,
)


class ChunkedSchemaAnalyzer:
    """Incrementally analyzes schema across DataFrame chunks."""

    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.precision = getattr(config, "hll_precision", 14)

        if not 4 <= self.precision <= 16:
            raise ValueError("hll_precision must be between 4 and 16")

        self.column_names: list[str] = []
        self.column_types: dict[str, str] = {}
        self.inferred_types: dict[str, str] = {}

        self.total_rows = 0
        self.unique_sketches: dict[str, HyperLogLog] = {}

        self.initialized = False

    def update(self, chunk: pd.DataFrame) -> None:
        """Process one chunk and update accumulated schema information."""

        if not isinstance(chunk, pd.DataFrame):
            raise TypeError("chunk must be a pandas DataFrame")

        if not self.initialized:
            self._initialize_schema(chunk)
        else:
            self._validate_schema(chunk)

        for column_name, series in chunk.items():
            self._update_column_type(column_name, series)

            for value in series.dropna():
                self.unique_sketches[column_name].update(
                    self._serialize_value(value)
                )

        self.total_rows += len(chunk)

    def _initialize_schema(self, chunk: pd.DataFrame) -> None:
        """Initialize schema using the first chunk."""

        if len(chunk.columns) == 0:
            raise ValueError("Dataset must contain at least one column")

        if not chunk.columns.is_unique:
            raise ValueError("Duplicate column names are not supported")

        self.column_names = list(chunk.columns)

        from backend.core.profiling.analyzers.schema_analyzer import (
            SchemaAnalyzer,
        )

        analyzer = SchemaAnalyzer()

        for column_name, series in chunk.items():
            self.column_types[column_name] = series.dtype.name
            self.inferred_types[column_name] = (
                analyzer._infer_column_type(column_name, series)
            )
            self.unique_sketches[column_name] = HyperLogLog(
                p=self.precision
            )

        self.initialized = True

    def _validate_schema(self, chunk: pd.DataFrame) -> None:
        """Validate column names and ordering across chunks."""

        if not chunk.columns.is_unique:
            raise ValueError("Duplicate column names are not supported")

        if list(chunk.columns) != self.column_names:
            raise ValueError(
                "Schema mismatch: chunk columns do not match "
                "the initial dataset schema"
            )

    def _update_column_type(
        self,
        column_name: str,
        series: pd.Series,
    ) -> None:
        """Promote compatible data types across chunks."""

        # Empty chunks must not affect schema inference.
        if series.empty:
            return

        current_type = self.column_types[column_name]
        new_type = series.dtype.name

        if current_type == new_type:
            return

        if (
            pd.api.types.is_numeric_dtype(series.dtype)
            and current_type in {
                "int8", "int16", "int32", "int64",
                "uint8", "uint16", "uint32", "uint64",
                "float16", "float32", "float64",
            }
        ):
            current_dtype = pd.api.types.pandas_dtype(current_type)

            if (
                pd.api.types.is_float_dtype(current_dtype)
                or pd.api.types.is_float_dtype(series.dtype)
            ):
                self.column_types[column_name] = "float64"
            else:
                self.column_types[column_name] = "int64"

            return

        raise ValueError(
            f"Incompatible data type in column '{column_name}': "
            f"{current_type} and {new_type}"
        )

    def _serialize_value(self, value) -> bytes:
        """Create a type-aware, deterministic representation of a value."""

        type_name = (
            f"{type(value).__module__}.{type(value).__qualname__}"
        )

        if isinstance(value, (pd.Timestamp,)):
            value = value.isoformat()
        elif isinstance(value, (bytes, bytearray)):
            value = value.hex()
        else:
            value = str(value)

        payload = json.dumps(
            [type_name, value],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        return hashlib.sha256(payload).digest()

    def finalize(self) -> list[ColumnProfile]:
        """Return the accumulated column profiles."""

        if not self.initialized:
            return []

        profiles = []

        for column_name in self.column_names:
            sketch = self.unique_sketches[column_name]

            profiles.append(
                ColumnProfile(
                    name=column_name,
                    data_type=self.column_types[column_name],
                    inferred_type=self.inferred_types[column_name],
                    total_values=self.total_rows,
                    unique_values=int(sketch.count()),
                )
            )

        return profiles
