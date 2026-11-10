import io
import fitz
from backend.app.services.document_parser import document_parser_service

def create_sample_pdf_bytes():
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((50, 50), "Deep Learning for Scientific Layouts", fontsize=24)
    page1.insert_text((50, 100), "Authors: Dr. Alice Smith, Dr. Bob Jones", fontsize=14)
    page1.insert_text((50, 150), "Abstract\nThis paper investigates automated poster layout generation.", fontsize=12)
    page1.insert_text((50, 220), "Introduction\nPresenting complex research requires structured layouts.", fontsize=12)
    
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Methodology\nWe apply anisotropic DBSCAN for spatial grouping of elements.", fontsize=12)
    page2.insert_text((50, 200), "Results\nOur approach achieved 0.88 silhouette score on benchmark posters.", fontsize=12)
    page2.insert_text((50, 350), "Conclusion\nAutomated poster generation is viable and scalable.", fontsize=12)

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    buf.seek(0)
    return buf.getvalue()

def test_document_upload_and_extraction(client, test_user):
    headers = test_user["headers"]
    
    # Create project
    proj_res = client.post("/api/v1/projects", json={"title": "Test Paper Project"}, headers=headers)
    project_id = proj_res.json()["id"]

    # Upload PDF
    pdf_bytes = create_sample_pdf_bytes()
    upload_res = client.post(
        f"/api/v1/projects/{project_id}/documents",
        files={"file": ("scientific_paper.pdf", pdf_bytes, "application/pdf")},
        headers=headers
    )
    assert upload_res.status_code == 201
    data = upload_res.json()
    assert data["name"] == "scientific_paper.pdf"
    assert data["document"]["pageCount"] == 2
    assert len(data["sections"]) == 13

    # Check that abstract was parsed with page 1
    abstract_sec = next(s for s in data["sections"] if s["id"] == "abstract")
    assert "automated poster layout" in abstract_sec["content"]
    assert abstract_sec["pageNumber"] == 1

    # Check methodology was parsed with page 2
    method_sec = next(s for s in data["sections"] if s["id"] == "methodology")
    assert "DBSCAN" in method_sec["content"]
    assert method_sec["pageNumber"] == 2

    # Check download endpoint
    doc_id = data["document"]["id"]
    download_res = client.get(f"/api/v1/documents/{doc_id}/download", headers=headers)
    assert download_res.status_code == 200
    assert len(download_res.content) == len(pdf_bytes)

