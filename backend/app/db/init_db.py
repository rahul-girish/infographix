from backend.app.db.session import engine
from backend.app.db.base import Base
import backend.app.models

def init_db() -> None:
    print("Creating all tables in PostgreSQL...")
    Base.metadata.create_all(bind=engine)
    print("All PostgreSQL tables created successfully!")

if __name__ == "__main__":
    init_db()

