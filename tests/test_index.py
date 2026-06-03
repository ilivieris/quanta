"""Tests for turborag.index.TurboIndex."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from turborag.exceptions import TurboRAGError

DIM = 768


# ── turbovec mock ─────────────────────────────────────────────────────────────

@pytest.fixture
def turbovec_mod():
    """Patch sys.modules so TurboIndex uses a fake turbovec."""
    tv = MagicMock()
    inner = MagicMock()
    inner.__len__ = MagicMock(return_value=0)
    tv.IdMapIndex.return_value = inner
    tv.IdMapIndex.load.return_value = inner
    with patch.dict(sys.modules, {"turbovec": tv}):
        yield tv, inner


@pytest.fixture
def turbo_index(turbovec_mod, tmp_index_dir):
    from turborag.index import TurboIndex

    tv, inner = turbovec_mod
    idx = TurboIndex(name="test", dim=DIM, bit_width=4, index_dir=tmp_index_dir)
    return idx, inner, tv


# ── add ───────────────────────────────────────────────────────────────────────

def test_add_calls_turbovec(turbo_index, sample_vectors, sample_ids):
    idx, inner, _ = turbo_index
    idx.add(sample_vectors, sample_ids)
    inner.add_with_ids.assert_called_once()
    args = inner.add_with_ids.call_args
    assert args[0][0].shape == (10, DIM)  # vectors
    assert args[0][1].dtype == np.uint64  # uint64 id array


def test_add_registers_id_mapping(turbo_index, sample_vectors, sample_ids):
    idx, _, _ = turbo_index
    idx.add(sample_vectors, sample_ids)
    for sid in sample_ids:
        assert sid in idx


def test_add_wrong_dim_raises(turbo_index):
    idx, _, _ = turbo_index
    bad = np.zeros((3, 100), dtype=np.float32)
    with pytest.raises(TurboRAGError, match="shape"):
        idx.add(bad, ["a", "b", "c"])


def test_add_mismatched_ids_raises(turbo_index, sample_vectors):
    idx, _, _ = turbo_index
    with pytest.raises(TurboRAGError, match=r"len\(ids\)"):
        idx.add(sample_vectors, ["only-one"])


def test_add_duplicate_id_is_idempotent(turbo_index, sample_vectors, sample_ids):
    """Re-adding the same IDs must not duplicate entries in the mapping."""
    idx, _, _ = turbo_index
    idx.add(sample_vectors, sample_ids)
    idx.add(sample_vectors, sample_ids)  # second time, no error
    assert len(idx._str_to_u64) == len(sample_ids)


def test_add_rolls_back_on_turbovec_failure(turbo_index, sample_vectors, sample_ids):
    idx, inner, _ = turbo_index
    inner.add_with_ids.side_effect = RuntimeError("turbovec boom")
    with pytest.raises(TurboRAGError):
        idx.add(sample_vectors, sample_ids)
    # Mappings must be clean after failure
    for sid in sample_ids:
        assert sid not in idx


# ── search ────────────────────────────────────────────────────────────────────

def _set_search_return(idx, inner, ids: list[str], scores: list[float]):
    u64s = np.array([idx._str_to_u64[sid] for sid in ids], dtype=np.uint64)
    scores_arr = np.array(scores, dtype=np.float32)
    # turbovec returns (scores, ids) each shape (1, k)
    result = (scores_arr[np.newaxis], u64s[np.newaxis])
    inner.search.return_value = result
    inner.search_with_allowlist.return_value = result


def test_search_returns_search_results(turbo_index, sample_vectors, sample_ids):
    idx, inner, _ = turbo_index
    idx.add(sample_vectors, sample_ids)
    _set_search_return(idx, inner, sample_ids[:3], [0.9, 0.7, 0.5])

    results = idx.search(sample_vectors[0], k=3)

    assert len(results) == 3
    assert results[0].id == sample_ids[0]
    assert results[0].score == pytest.approx(0.9)


def test_search_2d_query_squeezed(turbo_index, sample_vectors, sample_ids):
    idx, inner, _ = turbo_index
    idx.add(sample_vectors, sample_ids)
    _set_search_return(idx, inner, sample_ids[:2], [0.8, 0.6])

    # (1, DIM) query must work the same as (DIM,)
    query_2d = sample_vectors[0].reshape(1, DIM)
    results = idx.search(query_2d, k=2)
    assert len(results) == 2


def test_search_bad_query_shape_raises(turbo_index, sample_vectors):
    idx, _, _ = turbo_index
    with pytest.raises(TurboRAGError, match="shape"):
        idx.search(np.zeros((2, DIM), dtype=np.float32), k=1)


def test_search_with_allowed_ids(turbo_index, sample_vectors, sample_ids):
    idx, inner, _ = turbo_index
    idx.add(sample_vectors, sample_ids)
    allowed = sample_ids[:5]
    _set_search_return(idx, inner, allowed[:2], [0.9, 0.8])

    results = idx.search(sample_vectors[0], k=2, allowed_ids=allowed)

    inner.search_with_allowlist.assert_called_once()
    allowlist_arg = inner.search_with_allowlist.call_args.kwargs.get(
        "allowlist",
        inner.search_with_allowlist.call_args[1].get("allowlist"),
    )
    assert len(allowlist_arg) == 5
    assert len(results) == 2


def test_search_skips_unknown_u64(turbo_index, sample_vectors, sample_ids):
    """Unknown uint64 in turbovec results are silently dropped."""
    idx, inner, _ = turbo_index
    idx.add(sample_vectors, sample_ids)
    # Return one valid id and one unknown u64
    known_u64 = idx._str_to_u64[sample_ids[0]]
    inner.search.return_value = (
        np.array([[0.9, 0.8]], dtype=np.float32),
        np.array([[known_u64, 99999999999]], dtype=np.uint64),
    )
    results = idx.search(sample_vectors[0], k=2)
    assert len(results) == 1
    assert results[0].id == sample_ids[0]


# ── remove ────────────────────────────────────────────────────────────────────

def test_remove_known_id(turbo_index, sample_vectors, sample_ids):
    idx, _, _ = turbo_index
    idx.add(sample_vectors, sample_ids)
    removed = idx.remove(sample_ids[0])
    assert removed is True
    assert sample_ids[0] not in idx


def test_remove_unknown_id_returns_false(turbo_index):
    idx, _, _ = turbo_index
    assert idx.remove("nonexistent") is False


# ── contains / len ────────────────────────────────────────────────────────────

def test_contains_after_add(turbo_index, sample_vectors, sample_ids):
    idx, _, _ = turbo_index
    idx.add(sample_vectors, sample_ids)
    assert sample_ids[0] in idx
    assert "not-added" not in idx


# ── save / load ───────────────────────────────────────────────────────────────

def test_save_creates_both_files(turbo_index, sample_vectors, sample_ids, tmp_index_dir):
    idx, _, _ = turbo_index
    idx.add(sample_vectors, sample_ids)
    idx.save()

    assert Path(tmp_index_dir, "test.tvim").parent.exists()
    ids_path = Path(tmp_index_dir, "test.ids.json")
    assert ids_path.exists()
    payload = json.loads(ids_path.read_text())
    assert payload["dim"] == DIM
    assert payload["bit_width"] == 4
    assert set(payload["ids"].keys()) == set(sample_ids)


def test_load_restores_mapping(turbovec_mod, sample_vectors, sample_ids, tmp_index_dir):
    from turborag.index import TurboIndex

    tv, inner = turbovec_mod

    # Build, populate, and save an index
    idx = TurboIndex(name="myidx", dim=DIM, index_dir=tmp_index_dir)
    idx.add(sample_vectors, sample_ids)
    idx.save()
    # The mock doesn't write the .tvim file; create a placeholder so load() can find it.
    Path(tmp_index_dir, "myidx.tvim").touch()

    # Load a fresh instance
    loaded = TurboIndex.load("myidx", index_dir=tmp_index_dir)

    assert loaded._dim == DIM
    assert loaded._bit_width == 4
    assert set(loaded._str_to_u64.keys()) == set(sample_ids)
    # Round-trip: reverse mapping must be consistent
    for sid, u64 in loaded._str_to_u64.items():
        assert loaded._u64_to_str[u64] == sid


def test_load_missing_file_raises(turbovec_mod, tmp_index_dir):
    from turborag.index import TurboIndex

    with pytest.raises(TurboRAGError, match="not found"):
        TurboIndex.load("does_not_exist", index_dir=tmp_index_dir)


def test_repr(turbo_index):
    idx, _, _ = turbo_index
    r = repr(idx)
    assert "test" in r
    assert str(DIM) in r
