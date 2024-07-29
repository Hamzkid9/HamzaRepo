import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_upload_pdf():
    # Test uploading a PDF file
    with open("Hamza.Resume-2.pdf", "rb") as file:
        response = client.post("/upload_pdf/", files={"file": ("Hamza.Resume-2.pdf", file, "application/pdf")})
    assert response.status_code == 200
    assert "filename" in response.json()
    assert "content" in response.json()

def test_query_pdf():
    # Test querying the uploaded PDF content
    payload = {
        "query": "example",
        "filename": "Hamza.Resume-2.pdf"  # Ensure this matches the uploaded file name
    }
    response = client.post("/query_pdf/", json=payload)
    assert response.status_code == 200
    assert "response" in response.json()

def test_query_pdf_file_not_found():
    # Test querying a non-existent file
    payload = {
        "query": "example",
        "filename": "nonexistent.pdf"
    }
    response = client.post("/query_pdf/", json=payload)
    assert response.status_code == 404
    assert response.json() == {"detail": "File not found"}

def test_query_pdf_no_relevant_content():
    # Test querying a file with no relevant content
    payload = {
        "query": "nonexistent",
        "filename": "Hamza.Resume-2.pdf"  # Ensure this matches the uploaded file name
    }
    response = client.post("/query_pdf/", json=payload)
    assert response.status_code == 200
    assert response.json()["response"] == "No relevant content found."

def test_query_pdf_with_openai():
    # Test querying the uploaded PDF content using OpenAI API
    payload = {
        "query": "example",
        "filename": "Hamza.Resume-2.pdf"  # Ensure this matches the uploaded file name
    }
    response = client.post("/query_openai/", json=payload)
    assert response.status_code == 200
    assert "response" in response.json()
