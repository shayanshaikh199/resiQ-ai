import joblib
from sanity import looks_like_resume

MODEL_PATH = "models/resiq_model.joblib"
UNCERTAINTY_THRESHOLD = 0.15


class ResIQPredictor:
    def __init__(self):
        self.pipeline = None

    def load_model(self):
        if self.pipeline is None:
            print("DEBUG: Loading ML model...")
            self.pipeline = joblib.load(MODEL_PATH)
            print("DEBUG: Model loaded")

    def predict(self, resume_text: str, job_text: str):
        self.load_model()

        # Sanity check: reject non-resume text
        if not looks_like_resume(resume_text):
            return {
                "prediction": 0,
                "confidence": 0.0,
                "uncertain": False,
                "reason": "Uploaded document does not appear to be a resume"
            }

        combined_text = resume_text + " " + job_text
        prob = self.pipeline.predict_proba([combined_text])[0][1]

        return {
            "prediction": int(prob >= 0.5),
            "confidence": round(prob, 3),
            "uncertain": abs(prob - 0.5) < UNCERTAINTY_THRESHOLD
        }
