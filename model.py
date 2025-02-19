from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI
app = FastAPI()

# Load the Hugging Face summarization model
summarizer = pipeline("summarization")

# Define request model
class TextRequest(BaseModel):
    text: str

# Summarization endpoint
@app.post("/summarize")
def summarize_text(request: TextRequest):
    summary = summarizer(request.text, max_length=150, min_length=30, do_sample=False)
    return {"summary": summary[0]["summary_text"]}

# Root endpoint
@app.get("/")
def home():
    return {"message": "FastAPI Summarization API is running!"}
