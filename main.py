from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import pdfplumber
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import openai
import os

app = FastAPI()

# OpenAI API key configuration
openai.api_key = os.getenv("OPENAI_API_KEY")

# RAG model
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")

# Document store
document_store = {}

class Query(BaseModel):
    query: str
    filename: str

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    content = ""
    with pdfplumber.open(file.file) as pdf:
        for page in pdf.pages:
            content += page.extract_text()
    
    document_store[file.filename] = content
    return {"filename": file.filename, "content": content[:1000]}  # Return first 1000 chars for brevity

@app.post("/query_pdf/")
async def query_pdf(query: Query):
    filename = query.filename
    if filename not in document_store:
        raise HTTPException(status_code=404, detail="File not found")
    
    content = document_store[filename]
    
    # Enhanced keyword matching
    relevant_section = ""
    paragraphs = content.split("\n\n")
    for paragraph in paragraphs:
        if query.query.lower() in paragraph.lower():
            relevant_section += paragraph + "\n\n"
    
    if not relevant_section:
        return {"response": "No relevant content found."}
    
    # Generate response using RAG model
    try:
        inputs = tokenizer(relevant_section, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return {"response": summary}
    except Exception as e:
        return {"error": str(e)}

@app.post("/query_openai/")
async def query_openai(query: Query):
    filename = query.filename
    if filename not in document_store:
        raise HTTPException(status_code=404, detail="File not found")

    content = document_store[filename]

    # Retrieve relevant section using simple keyword matching
    relevant_section = ""
    for line in content.split("\n"):
        if query.query.lower() in line.lower():
            relevant_section += line + "\n"
    
    if not relevant_section:
        return {"response": "No relevant content found."}

    # Generate response using OpenAI LLM
    try:
        response = openai.Completion.create(
            engine="davinci",
            prompt=relevant_section + "\n\n" + query.query,
            max_tokens=150
        )
        return {"response": response.choices[0].text.strip()}
    except Exception as e:
        return {"error": str(e)}
