import uuid

def test_register_user(client):
    email = f"new_{uuid.uuid4().hex[:8]}@example.com"
    response = client.post("/api/v1/auth/register", json={
        "name": "Jane Doe",
        "email": email,
        "password": "SecurePassword123!"
    })
    assert response.status_code == 201
    data = response.json()
    assert "access_token" in data
    assert data["user"]["email"] == email
    assert data["user"]["name"] == "Jane Doe"

def test_register_duplicate_user(client, test_user):
    response = client.post("/api/v1/auth/register", json={
        "name": "Another User",
        "email": test_user["user"].email,
        "password": "AnotherPassword123!"
    })
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]

def test_login_success(client, test_user):
    response = client.post("/api/v1/auth/login", json={
        "email": test_user["user"].email,
        "password": "TestPassword123!"
    })
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["user"]["email"] == test_user["user"].email

def test_login_invalid_password(client, test_user):
    response = client.post("/api/v1/auth/login", json={
        "email": test_user["user"].email,
        "password": "WrongPassword999!"
    })
    assert response.status_code == 401

def test_get_me(client, test_user):
    response = client.get("/api/v1/auth/me", headers=test_user["headers"])
    assert response.status_code == 200
    assert response.json()["email"] == test_user["user"].email

