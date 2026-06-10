from __future__ import annotations

import json
import sys
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from quanta import QuantaIndex
from quanta.config import get_settings

# Allow importing neo4j_connection from examples/ when running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from examples.project.neo4j_connection import Neo4jConnection, neo4j_connection


_DEFAULT_INDEX_DIR = "./medical_indexes"
_INDEX_KEYS = ("patients", "doctors", "diagnoses", "procedures")

_QUERIES = {
    "patients": (
        "MATCH (p:Patient) RETURN p.patientId AS id, p.summary AS text",
        "Patient",
    ),
    "doctors": (
        "MATCH (d:Doctor) RETURN d.doctorId AS id, d.expertise AS text",
        "Doctor",
    ),
    "diagnoses": (
        "MATCH (d:Diagnosis) RETURN d.icdCode AS id, d.clinicalDescription AS text",
        "Diagnosis",
    ),
    "procedures": (
        "MATCH (p:Procedure) RETURN p.procCode AS id, p.procedureDescription AS text",
        "Procedure",
    ),
}


def _index_exists(index_dir: str) -> bool:
    p = Path(index_dir)
    return (
        all(
            (p / f"{key}.tvim").exists() and (p / f"{key}.ids.json").exists()
            for key in _INDEX_KEYS
        )
        and (p / "metadata.json").exists()
    )


class MedicalIndexer:
    def __init__(self, graph: Neo4jConnection, cfg, index_dir: str = _DEFAULT_INDEX_DIR) -> None:
        self._graph = graph
        self._cfg = cfg
        self._index_dir = index_dir
        self._model = SentenceTransformer(cfg.EMBED_MODEL)
        dim = cfg.EMBED_DIM

        self._indexes: dict[str, QuantaIndex] = {
            key: QuantaIndex(name=key, dim=dim, bit_width=4, index_dir=index_dir)
            for key in _INDEX_KEYS
        }
        # id → {node_type, display_name, raw_text}
        self.id_to_metadata: dict[str, dict] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def build(self) -> None:
        for key, (cypher, node_type) in _QUERIES.items():
            rows = self._graph.query(cypher) or []
            if not rows:
                print(f"[{node_type}] 0 nodes found — skipping")
                continue

            ids = [str(r["id"]) for r in rows]
            texts = [str(r["text"] or "") for r in rows]

            vectors = self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            ).astype(np.float32)

            self._indexes[key].add(vectors, ids)

            for nid, text in zip(ids, texts):
                self.id_to_metadata[nid] = {
                    "node_type":    node_type,
                    "display_name": nid,
                    "raw_text":     text,
                }

            print(f"[{node_type}] {len(ids)} nodes indexed")

    def save(self) -> None:
        Path(self._index_dir).mkdir(parents=True, exist_ok=True)
        for idx in self._indexes.values():
            idx.save()
        meta_path = Path(self._index_dir) / "metadata.json"
        meta_path.write_text(
            json.dumps(self.id_to_metadata, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[Indexer] Saved to {self._index_dir}/")

    @classmethod
    def load_from_disk(
        cls,
        graph: Neo4jConnection,
        cfg,
        index_dir: str = _DEFAULT_INDEX_DIR,
    ) -> MedicalIndexer:
        instance: MedicalIndexer = cls.__new__(cls)
        instance._graph = graph
        instance._cfg = cfg
        instance._index_dir = index_dir
        instance._model = SentenceTransformer(cfg.EMBED_MODEL)

        instance._indexes = {
            key: QuantaIndex.load(key, index_dir=index_dir)
            for key in _INDEX_KEYS
        }

        meta_path = Path(index_dir) / "metadata.json"
        instance.id_to_metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        total = sum(len(idx) for idx in instance._indexes.values())
        print(f"[Indexer] Loaded {total} vectors from {index_dir}/")
        return instance

    def search_patients(self, query_vec: np.ndarray, k: int = 5,
                        allowed_ids: list[str] | None = None) -> list[dict]:
        return self._search("patients", query_vec, k, allowed_ids)

    def search_doctors(self, query_vec: np.ndarray, k: int = 5,
                       allowed_ids: list[str] | None = None) -> list[dict]:
        return self._search("doctors", query_vec, k, allowed_ids)

    def search_diagnoses(self, query_vec: np.ndarray, k: int = 5,
                         allowed_ids: list[str] | None = None) -> list[dict]:
        return self._search("diagnoses", query_vec, k, allowed_ids)

    def search_procedures(self, query_vec: np.ndarray, k: int = 5,
                          allowed_ids: list[str] | None = None) -> list[dict]:
        return self._search("procedures", query_vec, k, allowed_ids)

    def embed(self, text: str) -> np.ndarray:
        return self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)[0]

    # ── Internals ──────────────────────────────────────────────────────────────

    def _search(self, index_key: str, query_vec: np.ndarray, k: int,
                allowed_ids: list[str] | None) -> list[dict]:
        idx = self._indexes[index_key]
        kwargs: dict = {"k": k}
        if allowed_ids is not None:
            kwargs["allowed_ids"] = allowed_ids

        hits = idx.search(query_vec, **kwargs)
        results = []
        for hit in hits:
            meta = self.id_to_metadata.get(hit.id, {})
            results.append({
                "id":           hit.id,
                "score":        hit.score,
                "node_type":    meta.get("node_type", ""),
                "display_name": meta.get("display_name", hit.id),
                "raw_text":     meta.get("raw_text", ""),
            })
        return results


# ── Module-level convenience ───────────────────────────────────────────────────

def build_indexer(
    uri: str,
    user: str,
    password: str,
    index_dir: str = _DEFAULT_INDEX_DIR,
    force_rebuild: bool = False,
) -> MedicalIndexer:
    graph = neo4j_connection(
        neo4j_settings={
            "connection_url": uri,
            "username": user,
            "password": password,
        }
    )
    cfg = get_settings()

    if not force_rebuild and _index_exists(index_dir):
        print(f"[Indexer] Found existing indexes in {index_dir}/ — loading from disk …")
        return MedicalIndexer.load_from_disk(graph=graph, cfg=cfg, index_dir=index_dir)

    print(f"[Indexer] Building indexes from Neo4j …")
    indexer = MedicalIndexer(graph=graph, cfg=cfg, index_dir=index_dir)
    indexer.build()
    indexer.save()
    return indexer
