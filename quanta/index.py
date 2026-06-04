from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import xxhash

from quanta.exceptions import QuantaError
from quanta.types import SearchResult
from quanta.utils.logging import get_logger

logger = get_logger(__name__)


def _hash(str_id: str) -> int:
    """Deterministic str -> uint64 via xxhash (collision-resistant, fixed seed)."""
    return xxhash.xxh64(str_id).intdigest()


def _normalise_hits(raw: Any) -> list[tuple[int, float]]:
    """Normalise turbovec search output to a flat list of (uint64_id, score) pairs.

    Handles two common return shapes:
      • (ids_array, scores_array) — FAISS-style numpy tuple
      • list of (id, score) tuples or objects with .id / .score attributes
    """
    if isinstance(raw, tuple) and len(raw) == 2:
        ids_arr, scores_arr = raw
        return [(int(uid), float(s)) for uid, s in zip(ids_arr, scores_arr, strict=False)]
    pairs: list[tuple[int, float]] = []
    for hit in raw:
        if isinstance(hit, tuple):
            uid, score = hit
        else:
            uid, score = hit.id, hit.score
        pairs.append((int(uid), float(score)))
    return pairs


class QuantaIndex:
    """turbovec ``IdMapIndex`` wrapper with string-id management and persistence.

    String IDs are mapped to uint64 via xxhash-64.  The mapping is kept in
    memory and flushed to ``{index_dir}/{name}.ids.json`` on ``save()``.
    The quantised vectors are written to ``{index_dir}/{name}.tvim``.
    """

    def __init__(
        self,
        name: str,
        dim: int,
        bit_width: int = 4,
        index_dir: str = "./indexes",
    ) -> None:
        self._name = name
        self._dim = dim
        self._bit_width = bit_width
        self._index_dir = Path(index_dir)

        self._str_to_u64: dict[str, int] = {}
        self._u64_to_str: dict[int, str] = {}

        self._index: Any = self._create_index()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _create_index(self) -> Any:
        try:
            import turbovec
        except ImportError as exc:
            raise QuantaError(
                "turbovec is required. Install it with: pip install turbovec"
            ) from exc
        try:
            return turbovec.IdMapIndex(dim=self._dim, bit_width=self._bit_width)
        except Exception as exc:
            raise QuantaError(
                f"Failed to create turbovec.IdMapIndex(dim={self._dim}, "
                f"bit_width={self._bit_width}): {exc}"
            ) from exc

    def _resolve_ids(self, ids: list[str]) -> tuple[list[int], list[str]]:
        """Hash every id; detect collisions; return (u64_list, newly_registered).

        Raises ``QuantaError`` on collision, leaving internal state unchanged.
        """
        u64_ids: list[int] = []
        newly_registered: list[str] = []
        try:
            for str_id in ids:
                u64 = _hash(str_id)
                existing = self._u64_to_str.get(u64)
                if existing is not None and existing != str_id:
                    raise QuantaError(
                        f"xxhash-64 collision: {str_id!r} and {existing!r} "
                        f"both map to {u64}"
                    )
                if str_id not in self._str_to_u64:
                    self._str_to_u64[str_id] = u64
                    self._u64_to_str[u64] = str_id
                    newly_registered.append(str_id)
                u64_ids.append(u64)
        except QuantaError:
            for sid in newly_registered:
                uid = self._str_to_u64.pop(sid, None)
                if uid is not None:
                    self._u64_to_str.pop(uid, None)
            raise
        return u64_ids, newly_registered

    @property
    def _tvim_path(self) -> Path:
        return self._index_dir / f"{self._name}.tvim"

    @property
    def _ids_path(self) -> Path:
        return self._index_dir / f"{self._name}.ids.json"

    # ── Mutations ─────────────────────────────────────────────────────────────

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        """Add *vectors* to the index, keyed by *ids*.

        Vectors with an ID that already exists are upserted (behaviour
        delegated to the underlying turbovec index).

        Args:
            vectors: float array of shape ``(n, dim)``.
            ids:     n string identifiers; order must match ``vectors``.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise QuantaError(
                f"vectors must have shape (n, {self._dim}), got {vectors.shape}"
            )
        if vectors.shape[0] != len(ids):
            raise QuantaError(
                f"len(ids)={len(ids)} does not match vectors.shape[0]={vectors.shape[0]}"
            )

        u64_ids, newly_registered = self._resolve_ids(ids)
        u64_array = np.array(u64_ids, dtype=np.uint64)

        try:
            self._index.add_with_ids(vectors, u64_array)
        except Exception as exc:
            for sid in newly_registered:
                uid = self._str_to_u64.pop(sid, None)
                if uid is not None:
                    self._u64_to_str.pop(uid, None)
            raise QuantaError(f"QuantaIndex.add failed: {exc}") from exc

        logger.debug(
            "QuantaIndex[%s] added %d vector(s) (size=%d)", self._name, len(ids), len(self)
        )

    def remove(self, id: str) -> bool:
        """Remove *id* from the index.

        Returns:
            ``True`` if the id was present and removed, ``False`` if unknown.
        """
        if id not in self._str_to_u64:
            return False
        u64 = self._str_to_u64[id]
        try:
            self._index.remove(u64)
        except Exception as exc:
            raise QuantaError(f"QuantaIndex.remove failed for id={id!r}: {exc}") from exc
        del self._str_to_u64[id]
        del self._u64_to_str[u64]
        logger.debug("QuantaIndex[%s] removed id=%r (size=%d)", self._name, id, len(self))
        return True

    # ── Queries ───────────────────────────────────────────────────────────────

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        allowed_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        """Search for the *k* nearest neighbours of *query*.

        Args:
            query:       float array of shape ``(dim,)`` or ``(1, dim)``.
            k:           maximum number of results to return.
            allowed_ids: when given, only IDs in this list are considered;
                         unknown IDs are silently skipped.

        Returns:
            List of :class:`SearchResult` sorted by descending score.
            ``metadata`` is left empty here; the retriever layer populates it.
        """
        query = np.ascontiguousarray(query, dtype=np.float32)
        if query.ndim == 2:
            if query.shape[0] != 1:
                raise QuantaError(
                    f"2-D query must have shape (1, {self._dim}), got {query.shape}"
                )
            query = query[0]
        if query.shape != (self._dim,):
            raise QuantaError(
                f"query must have shape ({self._dim},), got {query.shape}"
            )

        try:
            queries = query[np.newaxis]  # (1, dim) — turbovec requires 2-D batch
            if allowed_ids is not None:
                u64_allowlist = [
                    self._str_to_u64[sid] for sid in allowed_ids if sid in self._str_to_u64
                ]
                if not u64_allowlist:
                    return []
                allowlist = np.array(u64_allowlist, dtype=np.uint64)
                scores_batch, ids_batch = self._index.search(queries, k=k, allowlist=allowlist)
            else:
                scores_batch, ids_batch = self._index.search(queries, k=k)
            # turbovec returns (scores, ids) each shape (1, k); unpack the single row
            raw = list(zip(ids_batch[0].tolist(), scores_batch[0].tolist()))
        except QuantaError:
            raise
        except Exception as exc:
            raise QuantaError(f"QuantaIndex.search failed: {exc}") from exc

        results: list[SearchResult] = []
        for u64_id, score in _normalise_hits(raw):
            str_id = self._u64_to_str.get(u64_id)
            if str_id is None:
                logger.warning(
                    "QuantaIndex[%s] search returned unknown uint64 id %d — skipping",
                    self._name,
                    u64_id,
                )
                continue
            results.append(SearchResult(id=str_id, score=score))

        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Flush index vectors to ``<index_dir>/<name>.tvim`` and the id mapping
        to ``<index_dir>/<name>.ids.json``."""
        self._index_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._index.write(str(self._tvim_path))
        except Exception as exc:
            raise QuantaError(
                f"QuantaIndex.save failed writing {self._tvim_path}: {exc}"
            ) from exc

        payload = {
            "dim": self._dim,
            "bit_width": self._bit_width,
            "ids": self._str_to_u64,
        }
        try:
            self._ids_path.write_text(
                json.dumps(payload, separators=(",", ":")), encoding="utf-8"
            )
        except OSError as exc:
            raise QuantaError(
                f"QuantaIndex.save failed writing {self._ids_path}: {exc}"
            ) from exc

        logger.info(
            "QuantaIndex[%s] saved %d vector(s) to %s",
            self._name, len(self), self._index_dir,
        )

    @classmethod
    def load(cls, name: str, index_dir: str = "./indexes") -> QuantaIndex:
        """Reconstruct a :class:`QuantaIndex` from disk.

        Reads ``{index_dir}/{name}.tvim`` (turbovec) and
        ``{index_dir}/{name}.ids.json`` (id mapping + metadata).
        """
        index_path = Path(index_dir)
        tvim_path = index_path / f"{name}.tvim"
        ids_path = index_path / f"{name}.ids.json"

        for path in (tvim_path, ids_path):
            if not path.exists():
                raise QuantaError(f"Index file not found: {path}")

        try:
            raw_json = ids_path.read_text(encoding="utf-8")
            payload: dict[str, Any] = json.loads(raw_json)
            dim: int = payload["dim"]
            bit_width: int = payload["bit_width"]
            mapping: dict[str, int] = payload["ids"]
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            raise QuantaError(
                f"QuantaIndex.load failed reading {ids_path}: {exc}"
            ) from exc

        try:
            import turbovec
            tv_index = turbovec.IdMapIndex.load(str(tvim_path))
        except ImportError as exc:
            raise QuantaError(
                "turbovec is required. Install it with: pip install turbovec"
            ) from exc
        except Exception as exc:
            raise QuantaError(
                f"QuantaIndex.load failed reading {tvim_path}: {exc}"
            ) from exc

        # Bypass __init__ — the turbovec index is already built from disk.
        instance: QuantaIndex = cls.__new__(cls)
        instance._name = name
        instance._dim = dim
        instance._bit_width = bit_width
        instance._index_dir = index_path
        instance._index = tv_index
        instance._str_to_u64 = dict(mapping)
        instance._u64_to_str = {v: k for k, v in mapping.items()}

        logger.info(
            "QuantaIndex[%s] loaded %d vector(s) from %s", name, len(instance), index_dir
        )
        return instance

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        try:
            return len(self._index)
        except Exception:
            return len(self._str_to_u64)

    def __contains__(self, id: object) -> bool:
        return id in self._str_to_u64

    def __repr__(self) -> str:
        return (
            f"QuantaIndex(name={self._name!r}, dim={self._dim}, "
            f"bit_width={self._bit_width}, size={len(self)})"
        )
