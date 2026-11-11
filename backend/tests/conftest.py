import os
import tempfile
from pathlib import Path

_TEST_DB_DIR = tempfile.TemporaryDirectory()
os.environ["DATABASE_URL"] = f"sqlite:///{(Path(_TEST_DB_DIR.name) / 'test_bootstrap.db').as_posix()}"

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.db.base import Base
import backend.app.models  # noqa: F401


def create_test_session_factory():
    """Create an isolated temporary SQLite database and session factory for tests."""
    tmp_dir = tempfile.TemporaryDirectory()
    db_path = Path(tmp_dir.name) / "test_mlops.db"
    engine = create_engine(
        f"sqlite:///{db_path.as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return tmp_dir, engine, factory
