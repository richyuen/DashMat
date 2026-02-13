"""Database engine and configuration."""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


# AG Grid Enterprise License Key
# Replace with actual license key
AG_GRID_LICENSE_KEY = "your_license_key_here"


def _default_sqlite_url() -> str:
    db_path = Path(__file__).resolve().parent / "data" / "dashmat_local.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.as_posix()}"


def _default_mrd_sqlite_url() -> str:
    db_path = Path(__file__).resolve().parent / "data" / "MRD.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.as_posix()}"


def _build_database_url() -> str:
    """Resolve database URL with SQLite default and SQL Server support."""
    # Preferred: explicit SQLAlchemy URL, e.g.
    # mssql+pyodbc://user:pass@host:1433/db?driver=ODBC+Driver+18+for+SQL+Server
    explicit_url = os.getenv("DASHMAT_DATABASE_URL")
    if explicit_url:
        return explicit_url

    backend = os.getenv("DASHMAT_DB_BACKEND", "sqlite").strip().lower()
    if backend == "sqlserver":
        user = os.getenv("DASHMAT_DB_USER", "")
        password = os.getenv("DASHMAT_DB_PASSWORD", "")
        host = os.getenv("DASHMAT_DB_HOST", "localhost")
        port = os.getenv("DASHMAT_DB_PORT", "1433")
        database = os.getenv("DASHMAT_DB_NAME", "DashMat")
        driver = os.getenv("DASHMAT_DB_DRIVER", "ODBC Driver 18 for SQL Server")
        if user and password:
            return (
                f"mssql+pyodbc://{user}:{password}@{host}:{port}/{database}"
                f"?driver={driver.replace(' ', '+')}"
            )
    return _default_sqlite_url()


DATABASE_URL = _build_database_url()
engine: Engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)

MRD_DATABASE_URL = os.getenv("DASHMAT_MRD_DATABASE_URL", _default_mrd_sqlite_url())
engine_MRD: Engine = create_engine(MRD_DATABASE_URL, future=True, pool_pre_ping=True)
