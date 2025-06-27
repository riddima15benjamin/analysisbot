from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fitz  # PyMuPDF
import requests
import json
import re
from dotenv import load_dotenv
import os
load_dotenv()

app = FastAPI()

# Allow all origins (for development; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY environment variable")

class ResumeText(BaseModel):
    text: str


@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    """
    Extract text from uploaded PDF resume.
    """
    try:
        content = await file.read()
        doc = fitz.open(stream=content, filetype="pdf")
        extracted_text = "\n".join([page.get_text() for page in doc])
        return {"text": extracted_text}
    except Exception as e:
        return {"error": "Failed to extract text", "details": str(e)}


@app.post("/analyze-resume")
async def analyze_resume(data: ResumeText):
    prompt = f"""
You are an expert resume reviewer. Analyze the resume below and return a JSON response with the following keys:

- "score" (0-100)
- "suggestions": [list of things to improve]
- "strengths": [list of strong points]
- "weaknesses": [list of weak points]

Resume:
{data.text}

Respond only in valid JSON format, without explanation.
"""

    try:
        response = requests.post(
            url="https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5
            }
        )
        response.raise_for_status()
        result_json = response.json()
        content = result_json["choices"][0]["message"]["content"]
        
        # Strip any Markdown code block formatting
        cleaned = re.sub(r"```json|```", "", content).strip()


        return json.loads(cleaned)

    except requests.RequestException as req_err:
        return {"error": "Groq API request failed", "details": str(req_err)}
    except json.JSONDecodeError:
        return {"error": "Failed to decode Groq response as JSON", "raw": content}
    except Exception as e:
        return {"error": "Unexpected error occurred", "details": str(e)}