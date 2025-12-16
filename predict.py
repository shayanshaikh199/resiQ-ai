import joblib

MODEL_PATH = "models/resiq_model.joblib"
UNCERTAINTY_THRESHOLD = 0.15  # how close to 0.5 counts as uncertain


class ResIQPredictor:
    def __init__(self):
        # Load trained pipeline
        self.pipeline = joblib.load(MODEL_PATH)

    def predict(self, resume_text: str, job_text: str):
        # Combine texts
        text = resume_text + " " + job_text

        # Probability of a good match (class 1)
        prob = self.pipeline.predict_proba([text])[0][1]

        prediction = int(prob >= 0.5)

        # Uncertainty = closeness to decision boundary
        uncertainty = abs(prob - 0.5)

        return {
            "prediction": prediction,
            "confidence": round(prob, 3),
            "uncertain": uncertainty < UNCERTAINTY_THRESHOLD
        }
