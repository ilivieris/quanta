from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class QuantaSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "quanta"
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_POOL_SIZE: int = 5

    # ── Neo4j (optional — graph is optional) ─────────────────────────────────
    NEO4J_URI: str | None = None
    NEO4J_USER: str | None = None
    NEO4J_PASSWORD: str | None = None
    NEO4J_DATABASE: str = "neo4j"

    # ── Embedding ─────────────────────────────────────────────────────────────
    EMBED_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    EMBED_DIM: int = 768

    # ── Index defaults ────────────────────────────────────────────────────────
    DEFAULT_BIT_WIDTH: int = 4
    DEFAULT_TOP_K: int = 10

    # ── Docstore backend ──────────────────────────────────────────────────────
    DOCSTORE_BACKEND: Literal["postgres", "duckdb"] = "postgres"
    DUCKDB_PATH: str = "./quanta.duckdb"

    # ── Redis embedding cache ─────────────────────────────────────────────────
    REDIS_HOST: str | None = None
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = None
    REDIS_DB: int = 0
    REDIS_TTL_SECONDS: int = 86400

    # ── Chunking ──────────────────────────────────────────────────────────────
    CHUNKING_STRATEGY: Literal["fixed", "sentence", "semantic", "recursive"] = "fixed"
    CHUNKING_SIZE: int = 512
    CHUNKING_OVERLAP: int = 64
    CHUNKING_MAX_SENTENCES: int = 5
    CHUNKING_SEMANTIC_THRESHOLD: float = 0.85

    # ── BM25 (optional — tantivy backend) ────────────────────────────────────
    BM25_BACKEND: Literal["tantivy"] | None = None
    TANTIVY_INDEX_PATH: str = "./quanta_tantivy"

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def graph_configured(self) -> bool:
        return bool(self.NEO4J_URI and self.NEO4J_USER and self.NEO4J_PASSWORD)


def get_settings() -> QuantaSettings:
    return QuantaSettings()  # type: ignore[call-arg]
