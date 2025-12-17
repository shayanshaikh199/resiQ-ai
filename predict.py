"""
predict.py

Handles similarity scoring and recommendation generation.
"""

import joblib
from sklearn.metrics.pairwise import cosine_similarity
from recommendations import generate_recommendations


class ResIQPredictor:
    def __init__(self):
        self.vectorizer = joblib.load("models/vectorizer.joblib")
        self.model = joblib.load("models/resiq_model.joblib")

    def predict(self, resume_text: str, job_text: str) -> dict:
        vectors = self.vectorizer.transform([resume_text, job_text])
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

        prediction = int(similarity >= 0.6)

        insights = generate_recommendations(
            resume_text=resume_text,
            job_text=job_text,
            similarity_score=similarity,
            is_match=bool(prediction)
        )

        return {
            "prediction": prediction,
            "confidence": round(float(similarity), 3),
            "explanation": insights["explanation"],
            "recommendations": insights["recommendations"]
        }
