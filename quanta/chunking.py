from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from quanta.config import QuantaSettings
from quanta.exceptions import QuantaError
from quanta.types import ChunkRecord

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class TextChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, doc_id: str) -> list[ChunkRecord]: ...


class FixedSizeChunker(TextChunker):
    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, doc_id: str) -> list[ChunkRecord]:
        tokens = text.split()
        if not tokens:
            return []
        step = max(1, self.chunk_size - self.overlap)
        chunks: list[ChunkRecord] = []
        i = 0
        while i < len(tokens):
            window = tokens[i : i + self.chunk_size]
            idx = len(chunks)
            chunks.append(
                ChunkRecord(
                    id=f"{doc_id}_chunk_{idx}",
                    document_id=doc_id,
                    content=" ".join(window),
                    chunk_index=idx,
                )
            )
            i += step
        return chunks


class SentenceChunker(TextChunker):
    def __init__(self, max_sentences: int = 5, overlap_sentences: int = 1) -> None:
        self.max_sentences = max_sentences
        self.overlap_sentences = overlap_sentences

    def chunk(self, text: str, doc_id: str) -> list[ChunkRecord]:
        sentences = [s for s in _SENTENCE_RE.split(text.strip()) if s.strip()]
        if not sentences:
            return []
        step = max(1, self.max_sentences - self.overlap_sentences)
        chunks: list[ChunkRecord] = []
        i = 0
        while i < len(sentences):
            window = sentences[i : i + self.max_sentences]
            idx = len(chunks)
            chunks.append(
                ChunkRecord(
                    id=f"{doc_id}_chunk_{idx}",
                    document_id=doc_id,
                    content=" ".join(window),
                    chunk_index=idx,
                )
            )
            i += step
        return chunks


class SemanticChunker(TextChunker):
    def __init__(
        self,
        embed_fn: Callable[[str], npt.NDArray[Any]],
        threshold: float = 0.85,
        max_chunk_size: int = 512,
    ) -> None:
        self.embed_fn = embed_fn
        self.threshold = threshold
        self.max_chunk_size = max_chunk_size

    @staticmethod
    def _cosine_similarity(a: npt.NDArray[Any], b: npt.NDArray[Any]) -> float:
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def chunk(self, text: str, doc_id: str) -> list[ChunkRecord]:
        sentences = [s for s in _SENTENCE_RE.split(text.strip()) if s.strip()]
        if not sentences:
            return []
        embeddings = [self.embed_fn(s) for s in sentences]
        chunks: list[ChunkRecord] = []
        current: list[str] = [sentences[0]]
        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            current_size = len(" ".join(current).split())
            next_size = len(sentences[i].split())
            if sim >= self.threshold and current_size + next_size <= self.max_chunk_size:
                current.append(sentences[i])
            else:
                idx = len(chunks)
                chunks.append(
                    ChunkRecord(
                        id=f"{doc_id}_chunk_{idx}",
                        document_id=doc_id,
                        content=" ".join(current),
                        chunk_index=idx,
                    )
                )
                current = [sentences[i]]
        if current:
            idx = len(chunks)
            chunks.append(
                ChunkRecord(
                    id=f"{doc_id}_chunk_{idx}",
                    document_id=doc_id,
                    content=" ".join(current),
                    chunk_index=idx,
                )
            )
        return chunks


class RecursiveChunker(TextChunker):
    _DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " "]

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators if separators is not None else list(self._DEFAULT_SEPARATORS)

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split until every piece fits within chunk_size tokens."""
        if len(text.split()) <= self.chunk_size:
            return [text]
        if not separators:
            tokens = text.split()
            return [
                " ".join(tokens[i : i + self.chunk_size])
                for i in range(0, len(tokens), self.chunk_size)
            ]
        sep, *rest = separators
        result: list[str] = []
        for part in text.split(sep):
            part = part.strip()
            if not part:
                continue
            if len(part.split()) <= self.chunk_size:
                result.append(part)
            else:
                result.extend(self._split_text(part, rest))
        return result

    def chunk(self, text: str, doc_id: str) -> list[ChunkRecord]:
        if not text.strip():
            return []
        pieces = self._split_text(text.strip(), self.separators)
        if not pieces:
            return []
        chunks_text: list[str] = []
        current_tokens: list[str] = []
        for piece in pieces:
            piece_tokens = piece.split()
            if not piece_tokens:
                continue
            if current_tokens and len(current_tokens) + len(piece_tokens) > self.chunk_size:
                chunks_text.append(" ".join(current_tokens))
                overlap_toks = current_tokens[-self.overlap :] if self.overlap > 0 else []
                if len(overlap_toks) + len(piece_tokens) <= self.chunk_size:
                    current_tokens = overlap_toks + piece_tokens
                else:
                    current_tokens = list(piece_tokens)
            else:
                current_tokens.extend(piece_tokens)
        if current_tokens:
            chunks_text.append(" ".join(current_tokens))
        return [
            ChunkRecord(
                id=f"{doc_id}_chunk_{i}",
                document_id=doc_id,
                content=content,
                chunk_index=i,
            )
            for i, content in enumerate(chunks_text)
        ]


def get_chunker(
    config: QuantaSettings,
    embed_fn: Callable[[str], npt.NDArray[Any]] | None = None,
) -> TextChunker:
    strategy = config.CHUNKING_STRATEGY
    if strategy == "fixed":
        return FixedSizeChunker(config.CHUNKING_SIZE, config.CHUNKING_OVERLAP)
    if strategy == "sentence":
        return SentenceChunker(config.CHUNKING_MAX_SENTENCES)
    if strategy == "semantic":
        if embed_fn is None:
            raise QuantaError("SemanticChunker requires embed_fn")
        return SemanticChunker(embed_fn, config.CHUNKING_SEMANTIC_THRESHOLD)
    if strategy == "recursive":
        return RecursiveChunker(config.CHUNKING_SIZE, config.CHUNKING_OVERLAP)
    raise QuantaError(f"Unknown chunking strategy: {strategy!r}")
