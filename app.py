"""
app.py

FastAPI entrypoint.
Does NOT reinterpret predictor output.
"""

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from predict import ResIQPredictor
from utils import extract_text_from_pdf

app = FastAPI()
templates = Jinja2Templates(directory="templates")

predictor = ResIQPredictor()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request}
    )


@app.post("/analyze")
async def analyze(
    resume: UploadFile,
    job_description: str = Form(...)
):
    resume_text = extract_text_from_pdf(resume)

    # 🚨 PASS THROUGH EXACTLY
    return predictor.predict(
        resume_text=resume_text,
        job_text=job_description
    )
