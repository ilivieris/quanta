from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class DocumentRecord:
    id: str
    content: str
    doc_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass
class ChunkRecord:
    id: str
    document_id: str
    content: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphNode:
    id: str
    title: str | None
    relation: str | None
    distance: int


@dataclass
class RetrievalResult:
    id: str
    score: float
    source: str          # "dense", "graph", or "dense+graph"
    content: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    document_id: str | None = None
