import io
from PIL import Image

def create_sample_png_bytes():
    img = Image.new("RGB", (100, 100), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

def test_asset_lifecycle(client, test_user):
    headers = test_user["headers"]
    png_bytes = create_sample_png_bytes()

    # Upload image asset
    upload_res = client.post(
        "/api/v1/assets",
        files={"file": ("diagram.png", png_bytes, "image/png")},
        headers=headers
    )
    assert upload_res.status_code == 201
    asset_data = upload_res.json()
    assert asset_data["fileName"] == "diagram.png"
    assert "/content" in asset_data["url"]
    asset_id = asset_data["id"]

    # Stream image
    content_res = client.get(f"/api/v1/assets/{asset_id}/content")
    assert content_res.status_code == 200
    assert content_res.headers["content-type"] == "image/png"
    assert len(content_res.content) == len(png_bytes)

    # Delete asset
    del_res = client.delete(f"/api/v1/assets/{asset_id}", headers=headers)
    assert del_res.status_code == 204

    # Verify deleted
    verify_res = client.get(f"/api/v1/assets/{asset_id}/content")
    assert verify_res.status_code == 404

def test_project_submission(client, test_user):
    headers = test_user["headers"]

    # Create project
    proj_res = client.post("/api/v1/projects", json={"title": "Poster for Submission"}, headers=headers)
    project_id = proj_res.json()["id"]

    # Submit project
    submit_res = client.post(f"/api/v1/projects/{project_id}/submit", headers=headers)
    assert submit_res.status_code == 201
    data = submit_res.json()
    assert data["status"] == "submitted"
    assert "submittedAt" in data

    # Verify project status in list and get
    get_res = client.get(f"/api/v1/projects/{project_id}", headers=headers)
    assert get_res.json()["status"] == "submitted"

    # Get submission details
    sub_detail_res = client.get(f"/api/v1/projects/{project_id}/submission", headers=headers)
    assert sub_detail_res.status_code == 200
    assert sub_detail_res.json()["status"] == "submitted"

