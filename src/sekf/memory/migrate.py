"""Programmatic Alembic entry points for the memory store. Owner: Krishna."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Connection

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def alembic_config(url: str | None = None, connection: Connection | None = None) -> Config:
    """Build an Alembic config pointing at the memory migrations.

    Example:
        >>> cfg = alembic_config("sqlite:///./data/sekf.sqlite3")
    """
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    if url is not None:
        # ConfigParser interpolation treats '%' specially (URL-encoded passwords).
        cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    if connection is not None:
        cfg.attributes["connection"] = connection
    return cfg


def run_migrations(url: str | None = None, *, connection: Connection | None = None) -> None:
    """Upgrade the memory schema to the latest revision (idempotent).

    Pass either a database URL or an open connection (required for in-memory SQLite).
    """
    if url is None and connection is None:
        raise ValueError("pass a database url or a connection")
    command.upgrade(alembic_config(url, connection), "head")
