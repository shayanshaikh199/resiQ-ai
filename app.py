from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from predict import ResIQPredictor

app = FastAPI(title="ResIQ AI")

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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
    resume_text: str = Form(...),
    job_text: str = Form(...)
):
    result = predictor.predict(resume_text, job_text)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result}
    )
