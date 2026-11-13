from backend.app.db.base import Base
from backend.app.db.session import engine
import backend.app.models  # noqa: F401


def init_db() -> None:
    print("Creating all MLOps tables in database...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")


if __name__ == "__main__":
    init_db()
