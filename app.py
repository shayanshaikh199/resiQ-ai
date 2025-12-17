from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from predict import ResIQPredictor
from utils import extract_text_from_pdf

app = FastAPI()

# Static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Predictor (lazy-load model)
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
    print("DEBUG: PDF size =", len(pdf_bytes))

    resume_text = extract_text_from_pdf(pdf_bytes)
    print("DEBUG: extracted text length =", len(resume_text))

    result = predictor.predict(resume_text, job_text)
    print("DEBUG: prediction complete")

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result
        }
    )
