
import re

import pandas as pd

from backend.core.profiling.models import ColumnProfile


class SchemaAnalyzer:
    """Analyzes dataset schema and infers semantic column types."""

    IDENTIFIER_PATTERNS = {
        "id",
        "uuid",
        "guid",
        "key",
        "code",
        "pincode",
        "postal_code",
        "zip_code",
    }

    def analyze(self, dataframe: pd.DataFrame) -> list[ColumnProfile]:
        """
        Analyze all columns and return their schema profiles.
        """

        profiles = []

        for column_name, series in dataframe.items():

            inferred_type = self._infer_column_type(
                column_name=column_name,
                series=series,
            )

            profile = ColumnProfile(
                name=column_name,
                data_type=series.dtype.name,
                inferred_type=inferred_type,
                total_values=series.size,
                unique_values=series.nunique(dropna=True),
            )

            profiles.append(profile)

        return profiles

    def _infer_column_type(
        self,
        column_name: str,
        series: pd.Series,
    ) -> str:
        """
        Infer the semantic type of a column.
        """

        if pd.api.types.is_bool_dtype(series):
            return "boolean"

        if self._is_identifier(column_name, series):
            return "identifier"

        if self._is_datetime(series):
            return "datetime"

        if pd.api.types.is_numeric_dtype(series):
            return "numerical"

        return "categorical"

    def _is_identifier(
        self,
        column_name: str,
        series: pd.Series,
    ) -> bool:
        """
        Detect likely identifier columns using column names
        and value characteristics.
        """

        normalized_name = re.sub(
            r"[^a-z0-9]+",
            "_",
            column_name.lower(),
        ).strip("_")

        name_parts = set(normalized_name.split("_"))

        # Identifier detection based on column name.
        if name_parts.intersection(self.IDENTIFIER_PATTERNS):
            return True

        # Empty columns cannot be identified from their values.
        non_null = series.dropna()

        if non_null.empty:
            return False

        # UUID-like string values.
        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            sample = non_null.head(100)

            if sample.map(
                lambda value: bool(
                    re.fullmatch(
                        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
                        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                        r"[0-9a-fA-F]{12}",
                        str(value),
                    )
                )
            ).all():
                return True

        # High-cardinality columns alone are not sufficient
        # to classify a column as an identifier.
        return False

    def _is_datetime(self, series: pd.Series) -> bool:
        """
        Detect datetime columns without expensive conversion
        of arbitrary object columns.
        """

        return pd.api.types.is_datetime64_any_dtype(series)
