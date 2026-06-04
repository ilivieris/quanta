"""Tests for quanta.chunking — all chunker strategies and the get_chunker factory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest

from quanta.chunking import (
    FixedSizeChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
    TextChunker,
    get_chunker,
)
from quanta.exceptions import QuantaError
from quanta.types import ChunkRecord


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_config(**kwargs: object) -> MagicMock:
    cfg = MagicMock()
    cfg.CHUNKING_STRATEGY = kwargs.get("CHUNKING_STRATEGY", "fixed")
    cfg.CHUNKING_SIZE = kwargs.get("CHUNKING_SIZE", 512)
    cfg.CHUNKING_OVERLAP = kwargs.get("CHUNKING_OVERLAP", 64)
    cfg.CHUNKING_MAX_SENTENCES = kwargs.get("CHUNKING_MAX_SENTENCES", 5)
    cfg.CHUNKING_SEMANTIC_THRESHOLD = kwargs.get("CHUNKING_SEMANTIC_THRESHOLD", 0.85)
    return cfg


def _word_tokens(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


# ── FixedSizeChunker ──────────────────────────────────────────────────────────

def test_fixed_is_text_chunker() -> None:
    assert isinstance(FixedSizeChunker(), TextChunker)


def test_fixed_chunk_count() -> None:
    # 20 tokens, chunk_size=10, overlap=3 → step=7 → starts at 0,7,14 → 3 chunks
    text = _word_tokens(20)
    chunks = FixedSizeChunker(chunk_size=10, overlap=3).chunk(text, "doc")
    assert len(chunks) == 3


def test_fixed_chunk_ids_and_indices() -> None:
    text = _word_tokens(20)
    chunks = FixedSizeChunker(chunk_size=10, overlap=3).chunk(text, "doc")
    for i, c in enumerate(chunks):
        assert c.id == f"doc_chunk_{i}"
        assert c.chunk_index == i
        assert c.document_id == "doc"


def test_fixed_overlap_preserved() -> None:
    # chunk 0 ends at token[9], chunk 1 starts at token[7] → 3-token overlap
    text = _word_tokens(20)
    chunks = FixedSizeChunker(chunk_size=10, overlap=3).chunk(text, "doc")
    end_of_first = chunks[0].content.split()[-3:]
    start_of_second = chunks[1].content.split()[:3]
    assert end_of_first == start_of_second


def test_fixed_last_chunk_may_be_partial() -> None:
    # 15 tokens, chunk_size=10, overlap=3 → step=7 → starts at 0, 7, 14 → 3 chunks
    text = _word_tokens(15)
    chunks = FixedSizeChunker(chunk_size=10, overlap=3).chunk(text, "doc")
    assert len(chunks) == 3
    assert len(chunks[1].content.split()) == 8  # tokens 7–14
    assert len(chunks[2].content.split()) == 1  # token 14 only


def test_fixed_empty_string() -> None:
    assert FixedSizeChunker().chunk("", "doc") == []


def test_fixed_whitespace_only() -> None:
    assert FixedSizeChunker().chunk("   \n\t  ", "doc") == []


def test_fixed_single_word() -> None:
    chunks = FixedSizeChunker(chunk_size=10, overlap=3).chunk("hello", "doc")
    assert len(chunks) == 1
    assert chunks[0].content == "hello"


def test_fixed_very_long_text() -> None:
    text = _word_tokens(2000)
    chunks = FixedSizeChunker(chunk_size=100, overlap=10).chunk(text, "doc")
    assert len(chunks) > 1
    for c in chunks:
        assert isinstance(c, ChunkRecord)
        assert len(c.content.split()) <= 100


def test_fixed_no_overlap_no_duplication() -> None:
    text = _word_tokens(20)
    chunks = FixedSizeChunker(chunk_size=10, overlap=0).chunk(text, "doc")
    assert len(chunks) == 2
    all_words = chunks[0].content.split() + chunks[1].content.split()
    assert all_words == text.split()


# ── SentenceChunker ───────────────────────────────────────────────────────────

def test_sentence_is_text_chunker() -> None:
    assert isinstance(SentenceChunker(), TextChunker)


def test_sentence_boundaries_respected() -> None:
    text = "Hello. World! How are you?"
    chunks = SentenceChunker(max_sentences=2, overlap_sentences=1).chunk(text, "doc")
    # sentences: ["Hello.", "World!", "How are you?"], step=1
    assert len(chunks) == 3
    assert "Hello." in chunks[0].content
    assert "World!" in chunks[0].content
    assert "World!" in chunks[1].content  # overlap sentence


def test_sentence_overlap_sentence_appears_in_consecutive_chunks() -> None:
    text = "One. Two. Three. Four. Five."
    # max_sentences=3, overlap=1 → step=2 → starts 0,2,4
    chunks = SentenceChunker(max_sentences=3, overlap_sentences=1).chunk(text, "doc")
    assert len(chunks) == 3
    # chunk 0 last sentence should appear in chunk 1 first sentence
    last_of_first = chunks[0].content.split(" ")[-1]
    assert last_of_first in chunks[1].content


def test_sentence_empty_string() -> None:
    assert SentenceChunker().chunk("", "doc") == []


def test_sentence_single_sentence() -> None:
    chunks = SentenceChunker(max_sentences=5).chunk("Just one sentence.", "doc")
    assert len(chunks) == 1
    assert chunks[0].content == "Just one sentence."


def test_sentence_no_boundary_treated_as_one_sentence() -> None:
    text = "no sentence boundary here at all"
    chunks = SentenceChunker(max_sentences=5).chunk(text, "doc")
    assert len(chunks) == 1
    assert chunks[0].content == text


def test_sentence_chunk_ids() -> None:
    text = "A. B. C. D. E."
    chunks = SentenceChunker(max_sentences=2, overlap_sentences=0).chunk(text, "doc")
    for i, c in enumerate(chunks):
        assert c.id == f"doc_chunk_{i}"
        assert c.document_id == "doc"


def test_sentence_very_long_text() -> None:
    sentences = [f"Sentence number {i}." for i in range(100)]
    text = " ".join(sentences)
    chunks = SentenceChunker(max_sentences=10, overlap_sentences=2).chunk(text, "doc")
    assert len(chunks) > 1
    for c in chunks:
        assert isinstance(c, ChunkRecord)


# ── SemanticChunker ───────────────────────────────────────────────────────────

def _make_embed_fn(
    vectors: dict[str, npt.NDArray[Any]],
) -> Callable[[str], npt.NDArray[Any]]:
    def embed(text: str) -> npt.NDArray[Any]:
        return vectors[text]
    return embed


def test_semantic_is_text_chunker() -> None:
    embed_fn: Callable[[str], npt.NDArray[Any]] = lambda t: np.array([1.0, 0.0])
    assert isinstance(SemanticChunker(embed_fn), TextChunker)


def test_semantic_merges_similar_splits_dissimilar() -> None:
    # A and B are identical (sim=1.0 >= 0.85), C is orthogonal (sim=0.0 < 0.85)
    sent_a = "Alpha sentence here."
    sent_b = "Beta sentence there."
    sent_c = "Gamma breaks the chain."
    text = f"{sent_a} {sent_b} {sent_c}"
    vectors: dict[str, npt.NDArray[Any]] = {
        sent_a: np.array([1.0, 0.0]),
        sent_b: np.array([1.0, 0.0]),  # same direction → sim=1.0
        sent_c: np.array([0.0, 1.0]),  # orthogonal → sim=0.0
    }
    chunks = SemanticChunker(_make_embed_fn(vectors), threshold=0.85).chunk(text, "doc")
    assert len(chunks) == 2
    assert sent_a in chunks[0].content
    assert sent_b in chunks[0].content
    assert sent_c in chunks[1].content


def test_semantic_all_dissimilar_gives_one_chunk_per_sentence() -> None:
    sentences = [f"Sentence {i}." for i in range(4)]
    text = " ".join(sentences)
    # each sentence gets orthogonal vector
    base = np.eye(4)
    vectors: dict[str, npt.NDArray[Any]] = {s: base[i] for i, s in enumerate(sentences)}
    chunks = SemanticChunker(_make_embed_fn(vectors), threshold=0.9).chunk(text, "doc")
    assert len(chunks) == 4


def test_semantic_max_chunk_size_forces_split() -> None:
    sent_a = "A B."
    sent_b = "C D."
    text = f"{sent_a} {sent_b}"
    # Same direction → sim=1.0, but max_chunk_size=2 forces a split
    vectors: dict[str, npt.NDArray[Any]] = {
        sent_a: np.array([1.0, 0.0]),
        sent_b: np.array([1.0, 0.0]),
    }
    chunks = SemanticChunker(
        _make_embed_fn(vectors), threshold=0.5, max_chunk_size=2
    ).chunk(text, "doc")
    assert len(chunks) == 2


def test_semantic_empty_string() -> None:
    embed_fn: Callable[[str], npt.NDArray[Any]] = lambda t: np.array([1.0, 0.0])
    assert SemanticChunker(embed_fn).chunk("", "doc") == []


def test_semantic_single_sentence() -> None:
    sent = "Only one sentence."
    vectors: dict[str, npt.NDArray[Any]] = {sent: np.array([1.0, 0.0])}
    chunks = SemanticChunker(_make_embed_fn(vectors)).chunk(sent, "doc")
    assert len(chunks) == 1
    assert chunks[0].content == sent


def test_semantic_zero_norm_vectors_do_not_crash() -> None:
    sent_a = "A."
    sent_b = "B."
    text = f"{sent_a} {sent_b}"
    vectors: dict[str, npt.NDArray[Any]] = {
        sent_a: np.array([0.0, 0.0]),
        sent_b: np.array([0.0, 0.0]),
    }
    chunks = SemanticChunker(_make_embed_fn(vectors), threshold=0.85).chunk(text, "doc")
    assert len(chunks) == 2  # sim=0.0 < 0.85, so split


# ── RecursiveChunker ──────────────────────────────────────────────────────────

def test_recursive_is_text_chunker() -> None:
    assert isinstance(RecursiveChunker(), TextChunker)


def test_recursive_separator_priority_paragraph_first() -> None:
    # Two paragraphs that each fit in chunk_size=5 tokens — splits on "\n\n" not "\n"
    text = "A B C\n\nD E F"
    chunks = RecursiveChunker(chunk_size=5, overlap=0).chunk(text, "doc")
    # A B C and D E F are separate pieces (3 tokens each, 6 total > 5)
    assert len(chunks) == 2
    assert chunks[0].content == "A B C"
    assert chunks[1].content == "D E F"


def test_recursive_falls_back_to_newline_separator() -> None:
    # No paragraph breaks; use "\n" as the separator
    text = "A B C\nD E F"
    chunks = RecursiveChunker(chunk_size=5, overlap=0).chunk(text, "doc")
    assert len(chunks) == 2
    assert "A B C" in chunks[0].content
    assert "D E F" in chunks[1].content


def test_recursive_falls_back_to_space() -> None:
    # One long line, no structural separators — falls through to " "
    words = _word_tokens(20)
    chunks = RecursiveChunker(
        chunk_size=5, overlap=0, separators=["\n\n", "\n"]
    ).chunk(words, "doc")
    # Should fall back to whitespace splitting handled at base level
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c.content.split()) <= 5


def test_recursive_overlap_preserved() -> None:
    # p1: 8 tokens (= chunk_size), p2: 4 tokens → overlap(3)+p2(4)=7 ≤ 8, so overlap kicks in
    p1 = " ".join(f"w{i}" for i in range(8))
    p2 = " ".join(f"x{i}" for i in range(4))
    text = f"{p1}\n\n{p2}"
    chunks = RecursiveChunker(chunk_size=8, overlap=3).chunk(text, "doc")
    assert len(chunks) == 2
    end_first = chunks[0].content.split()[-3:]
    start_second = chunks[1].content.split()[:3]
    assert end_first == start_second


def test_recursive_empty_string() -> None:
    assert RecursiveChunker().chunk("", "doc") == []


def test_recursive_single_word() -> None:
    chunks = RecursiveChunker(chunk_size=10, overlap=2).chunk("hello", "doc")
    assert len(chunks) == 1
    assert chunks[0].content == "hello"


def test_recursive_very_long_text() -> None:
    text = _word_tokens(1000)
    chunks = RecursiveChunker(chunk_size=50, overlap=5).chunk(text, "doc")
    assert len(chunks) > 1
    for c in chunks:
        assert isinstance(c, ChunkRecord)


def test_recursive_chunk_ids() -> None:
    text = _word_tokens(20)
    chunks = RecursiveChunker(chunk_size=8, overlap=0).chunk(text, "doc")
    for i, c in enumerate(chunks):
        assert c.id == f"doc_chunk_{i}"
        assert c.document_id == "doc"


# ── get_chunker factory ───────────────────────────────────────────────────────

def test_get_chunker_fixed() -> None:
    cfg = _make_config(CHUNKING_STRATEGY="fixed", CHUNKING_SIZE=256, CHUNKING_OVERLAP=32)
    chunker = get_chunker(cfg)
    assert isinstance(chunker, FixedSizeChunker)
    assert chunker.chunk_size == 256
    assert chunker.overlap == 32


def test_get_chunker_sentence() -> None:
    cfg = _make_config(CHUNKING_STRATEGY="sentence", CHUNKING_MAX_SENTENCES=3)
    chunker = get_chunker(cfg)
    assert isinstance(chunker, SentenceChunker)
    assert chunker.max_sentences == 3


def test_get_chunker_recursive() -> None:
    cfg = _make_config(CHUNKING_STRATEGY="recursive", CHUNKING_SIZE=128, CHUNKING_OVERLAP=16)
    chunker = get_chunker(cfg)
    assert isinstance(chunker, RecursiveChunker)
    assert chunker.chunk_size == 128
    assert chunker.overlap == 16


def test_get_chunker_semantic_with_embed_fn() -> None:
    cfg = _make_config(CHUNKING_STRATEGY="semantic", CHUNKING_SEMANTIC_THRESHOLD=0.9)
    embed_fn: Callable[[str], npt.NDArray[Any]] = lambda t: np.array([1.0, 0.0])
    chunker = get_chunker(cfg, embed_fn=embed_fn)
    assert isinstance(chunker, SemanticChunker)
    assert chunker.threshold == 0.9


def test_get_chunker_semantic_without_embed_fn_raises() -> None:
    cfg = _make_config(CHUNKING_STRATEGY="semantic")
    with pytest.raises(QuantaError, match="embed_fn"):
        get_chunker(cfg)
