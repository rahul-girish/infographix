import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.main import app
from backend.app.db.session import get_db, SessionLocal
from backend.app.models.user import User
from backend.app.core.security import get_password_hash, create_access_token

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture(scope="function")
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@pytest.fixture(scope="function")
def test_user(db_session: Session):
    email = f"test_{uuid.uuid4().hex[:8]}@university.edu"
    user = User(
        email=email,
        name="Dr. Test User",
        hashed_password=get_password_hash("TestPassword123!"),
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    
    token = create_access_token(user.id)
    headers = {"Authorization": f"Bearer {token}"}
    
    yield {"user": user, "token": token, "headers": headers}
    
    # Cleanup
    db_session.delete(user)
    db_session.commit()

