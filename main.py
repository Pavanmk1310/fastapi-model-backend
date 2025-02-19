from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI
app = FastAPI()

# Load a smaller Hugging Face summarization model (T5-small)
summarizer = pipeline("summarization", model="t5-small")

# Define request model
class TextRequest(BaseModel):
    text: str

# Summarization endpoint
@app.post("/summarize")
def summarize_text(request: TextRequest):
    summary = summarizer(request.text, max_length=100, min_length=20, do_sample=False)
    return {"summary": summary[0]["summary_text"]}

# Root endpoint
@app.get("/")
def home():
    return {"message": "FastAPI Summarization API is running with T5-small!"}
