import uuid

def test_project_lifecycle(client, test_user):
    headers = test_user["headers"]

    # 1. Create project
    create_res = client.post("/api/v1/projects", json={
        "title": "Quantum Computing Architecture",
        "templateId": "research"
    }, headers=headers)
    assert create_res.status_code == 201
    project = create_res.json()
    project_id = project["id"]
    assert project["title"] == "Quantum Computing Architecture"
    assert project["status"] == "draft"
    assert len(project["sections"]) == 13
    assert project["poster"] is not None
    assert len(project["poster"]["elements"]) > 0

    # 2. Get project
    get_res = client.get(f"/api/v1/projects/{project_id}", headers=headers)
    assert get_res.status_code == 200
    assert get_res.json()["id"] == project_id

    # 3. Rename project
    rename_res = client.patch(f"/api/v1/projects/{project_id}/rename", json={
        "title": "Quantum Computing & Quantum Algorithms"
    }, headers=headers)
    assert rename_res.status_code == 200
    assert rename_res.json()["title"] == "Quantum Computing & Quantum Algorithms"

    # 4. Duplicate project
    dup_res = client.post(f"/api/v1/projects/{project_id}/duplicate", headers=headers)
    assert dup_res.status_code == 201
    dup_project = dup_res.json()
    assert dup_project["id"] != project_id
    assert "(copy)" in dup_project["title"]
    # Verify element IDs are unique in duplicate
    original_elem_ids = [e["id"] for e in project["poster"]["elements"]]
    dup_elem_ids = [e["id"] for e in dup_project["poster"]["elements"]]
    assert len(set(original_elem_ids).intersection(set(dup_elem_ids))) == 0

    # 5. Save poster canvas
    updated_poster = project["poster"]
    updated_poster["title"] = "Updated Canvas Title"
    updated_poster["elements"].append({
        "id": str(uuid.uuid4()),
        "type": "text",
        "name": "Custom Callout",
        "x": 100,
        "y": 100,
        "width": 200,
        "height": 50,
        "rotation": 0,
        "layer": 99,
        "visible": True,
        "locked": False,
        "text": "Critical Finding!"
    })
    save_res = client.put(f"/api/v1/projects/{project_id}/poster", json=updated_poster, headers=headers)
    assert save_res.status_code == 200

    # Verify canvas updated
    get_after_save = client.get(f"/api/v1/projects/{project_id}", headers=headers).json()
    assert len(get_after_save["poster"]["elements"]) == len(updated_poster["elements"])

    # 6. Delete project
    del_res = client.delete(f"/api/v1/projects/{project_id}", headers=headers)
    assert del_res.status_code == 204

    # 7. Verify deletion
    verify_del = client.get(f"/api/v1/projects/{project_id}", headers=headers)
    assert verify_del.status_code == 404

