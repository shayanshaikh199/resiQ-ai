from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from predict import ResIQPredictor
from pdf_utils import extract_text_from_pdf

app = FastAPI(title="ResIQ AI")

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model once
predictor = ResIQPredictor()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": None}
    )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    resume_pdf: UploadFile = File(...),
    job_text: str = Form(...)
):
    resume_text = extract_text_from_pdf(resume_pdf.file)

    result = predictor.predict(resume_text, job_text)

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result}
    )
