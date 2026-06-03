from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class TurboRAGSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "turborag"
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_POOL_SIZE: int = 5

    # ── Neo4j (optional — graph is optional) ─────────────────────────────────
    NEO4J_URI: str | None = None
    NEO4J_USER: str | None = None
    NEO4J_PASSWORD: str | None = None
    NEO4J_DATABASE: str = "neo4j"

    # ── Index defaults ────────────────────────────────────────────────────────
    DEFAULT_BIT_WIDTH: int = 4
    DEFAULT_TOP_K: int = 10

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
