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


# ── Image index ────────────────────────────────────────────────────────────────

_IMAGE_INDEX_KEY = "images"
_IMAGE_EMBED_DIM = 512  # BiomedCLIP output dimensionality


def _image_index_exists(index_dir: str) -> bool:
    p = Path(index_dir)
    return (
        (p / f"{_IMAGE_INDEX_KEY}.tvim").exists()
        and (p / f"{_IMAGE_INDEX_KEY}.ids.json").exists()
        and (p / "image_metadata.json").exists()
    )


class ImageIndexer:
    """BiomedCLIP-based index for medical images. Supports text→image and image→image search."""

    MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    def __init__(self, graph: Neo4jConnection, index_dir: str = _DEFAULT_INDEX_DIR) -> None:
        self._graph = graph
        self._index_dir = index_dir
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._index = QuantaIndex(
            name=_IMAGE_INDEX_KEY,
            dim=_IMAGE_EMBED_DIM,
            bit_width=4,
            index_dir=index_dir,
        )
        self.id_to_metadata: dict[str, dict] = {}

    # ── Model loading (lazy) ───────────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip_torch is not installed. Run: pip install open_clip_torch"
            )
        model, _preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            self.MODEL_NAME
        )
        tokenizer = open_clip.get_tokenizer(self.MODEL_NAME)
        model.eval()
        self._model = model
        self._preprocess = preprocess_val
        self._tokenizer = tokenizer
        print("[ImageIndexer] BiomedCLIP loaded")

    # ── Embedding ──────────────────────────────────────────────────────────────

    def embed_image(self, image_path: str) -> np.ndarray:
        """Embed an image file → 512-dim L2-normalised vector."""
        import torch
        from PIL import Image

        self._load_model()
        img = Image.open(image_path).convert("RGB")
        tensor = self._preprocess(img).unsqueeze(0)
        with torch.no_grad():
            features = self._model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text query for cross-modal image search → 512-dim L2-normalised vector."""
        import torch

        self._load_model()
        tokens = self._tokenizer([text])
        with torch.no_grad():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    # ── Build / Save / Load ────────────────────────────────────────────────────

    def build(self) -> None:
        """Query Neo4j for MedicalImage nodes, embed each image, add to index."""
        self._load_model()
        rows = self._graph.query(
            "MATCH (img:MedicalImage) "
            "RETURN img.imageId AS id, img.filePath AS path, "
            "       img.imageType AS img_type, img.description AS description, "
            "       img.modality AS modality, img.capturedDate AS captured_date, "
            "       img.patientId AS patient_id"
        ) or []

        if not rows:
            print("[ImageIndexer] No MedicalImage nodes found — skipping")
            return

        vectors: list[np.ndarray] = []
        ids: list[str] = []
        for i, row in enumerate(rows):
            path = row.get("path") or ""
            if not path or not Path(path).exists():
                continue
            try:
                vec = self.embed_image(path)
                vectors.append(vec)
                img_id = str(row["id"])
                ids.append(img_id)
                self.id_to_metadata[img_id] = {
                    "image_type":    row.get("img_type", ""),
                    "description":   row.get("description", ""),
                    "modality":      row.get("modality", ""),
                    "captured_date": str(row.get("captured_date") or ""),
                    "patient_id":    row.get("patient_id", ""),
                    "file_path":     path,
                }
            except Exception as exc:
                print(f"[WARN] Could not embed {path}: {exc}")

            if (i + 1) % 50 == 0:
                print(f"[ImageIndexer] Embedded {i + 1}/{len(rows)} images …")

        if vectors:
            self._index.add(np.stack(vectors), ids)
        print(f"[ImageIndexer] {len(ids)} images indexed")

    def save(self) -> None:
        Path(self._index_dir).mkdir(parents=True, exist_ok=True)
        self._index.save()
        meta_path = Path(self._index_dir) / "image_metadata.json"
        meta_path.write_text(
            json.dumps(self.id_to_metadata, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[ImageIndexer] Saved to {self._index_dir}/")

    @classmethod
    def load_from_disk(
        cls, graph: Neo4jConnection, index_dir: str = _DEFAULT_INDEX_DIR
    ) -> "ImageIndexer":
        instance: "ImageIndexer" = cls.__new__(cls)
        instance._graph = graph
        instance._index_dir = index_dir
        instance._model = None
        instance._preprocess = None
        instance._tokenizer = None
        instance._index = QuantaIndex.load(_IMAGE_INDEX_KEY, index_dir=index_dir)
        meta_path = Path(index_dir) / "image_metadata.json"
        instance.id_to_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        print(
            f"[ImageIndexer] Loaded {len(instance._index)} image vectors from {index_dir}/"
        )
        return instance

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 5,
        image_type: str | None = None,
    ) -> list[dict]:
        """Return top-k similar images. Optionally filter by image_type."""
        allowed_ids: list[str] | None = None
        if image_type:
            allowed_ids = [
                iid for iid, meta in self.id_to_metadata.items()
                if meta.get("image_type") == image_type
            ]

        kwargs: dict = {"k": k}
        if allowed_ids is not None:
            kwargs["allowed_ids"] = allowed_ids

        hits = self._index.search(query_vec, **kwargs)
        results = []
        for hit in hits:
            meta = self.id_to_metadata.get(hit.id, {})
            results.append({
                "image_id":     hit.id,
                "score":        hit.score,
                "image_type":   meta.get("image_type", ""),
                "modality":     meta.get("modality", ""),
                "description":  meta.get("description", ""),
                "patient_id":   meta.get("patient_id", ""),
                "captured_date": meta.get("captured_date", ""),
                "file_path":    meta.get("file_path", ""),
            })
        return results


def build_image_indexer(
    graph: Neo4jConnection,
    index_dir: str = _DEFAULT_INDEX_DIR,
    force_rebuild: bool = False,
) -> ImageIndexer:
    if not force_rebuild and _image_index_exists(index_dir):
        print(f"[ImageIndexer] Found existing index in {index_dir}/ — loading from disk …")
        return ImageIndexer.load_from_disk(graph=graph, index_dir=index_dir)

    print("[ImageIndexer] Building image index from Neo4j …")
    indexer = ImageIndexer(graph=graph, index_dir=index_dir)
    indexer.build()
    indexer.save()
    return indexer
