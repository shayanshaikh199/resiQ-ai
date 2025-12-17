from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from predict import ResIQPredictor
from utils import extract_text_from_pdf

app = FastAPI()

# Static assets + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Predictor (now embedding-based)
predictor = ResIQPredictor()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": None}
    )


@app.post("/analyze", response_class=HTMLResponse)
def analyze(
    request: Request,
    resume_pdf: UploadFile = File(...),
    job_text: str = Form(...)
):
    print("DEBUG: analyze route hit")

    pdf_bytes = resume_pdf.file.read()
    resume_text = extract_text_from_pdf(pdf_bytes)

    result = predictor.predict(resume_text, job_text)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result
        }
    )
