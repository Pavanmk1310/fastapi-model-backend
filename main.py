from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL if needed, e.g., ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load model
summarizer = pipeline("summarization")

class TextRequest(BaseModel):
    text: str

@app.post("/summarize")
def summarize_text(request: TextRequest):
    summary = summarizer(request.text, max_length=150, min_length=30, do_sample=False)
    return {"summary": summary[0]["summary_text"]}

@app.get("/")
def home():
    return {"message": "FastAPI Summarization API is running!"}
