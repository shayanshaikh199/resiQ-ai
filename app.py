from fastapi import FastAPI
from pydantic import BaseModel

from predict import ResIQPredictor

app = FastAPI(title="ResIQ AI")

# Load model once at startup
predictor = ResIQPredictor()


class MatchRequest(BaseModel):
    resume_text: str
    job_text: str


class MatchResponse(BaseModel):
    prediction: int
    confidence: float
    uncertain: bool


@app.get("/")
def health_check():
    return {"status": "ResIQ AI running"}


@app.post("/predict", response_model=MatchResponse)
def predict_match(data: MatchRequest):
    result = predictor.predict(
        resume_text=data.resume_text,
        job_text=data.job_text
    )
    return result
