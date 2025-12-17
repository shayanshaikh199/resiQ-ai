"""
app.py

FastAPI backend for ResIQ AI using templates/.
"""

from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from predict import ResIQPredictor
from utils import extract_text_from_pdf

app = FastAPI()
predictor = ResIQPredictor()

# Tell FastAPI where templates live
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/analyze")
async def analyze(
    resume: UploadFile,
    job_description: str = Form(...)
):
    resume_text = extract_text_from_pdf(resume)
    return predictor.predict(resume_text, job_description)
