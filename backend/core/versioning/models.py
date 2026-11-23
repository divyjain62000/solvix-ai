
"""Data models for dataset versioning and change tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


ChangeType = Literal[
    "rows_deleted",
    "values_modified",
    "rows_added",
]


@dataclass
class DatasetVersion:
    """Immutable reference to a dataset version."""

    version_id: str
    dataset_id: str
    version_number: int

    parent_version_id: str | None = None
    operation_id: str | None = None

    checkpoint_reference: str | None = None

    created_at: datetime = field(default_factory=datetime.now)
    description: str = "Dataset version"


@dataclass
class ChangeRecord:
    """Reference to changes required for undo and redo."""

    change_id: str
    operation_id: str
    dataset_id: str

    change_type: ChangeType

    storage_reference: str
    affected_rows: int = 0
    affected_values: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)

    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CleaningHistory:
    """Tracks dataset versions and the currently active version."""

    dataset_id: str

    versions: list[DatasetVersion] = field(default_factory=list)

    current_version_id: str | None = None

    # Version IDs available for redo, ordered as a stack.
    redo_version_ids: list[str] = field(default_factory=list)
