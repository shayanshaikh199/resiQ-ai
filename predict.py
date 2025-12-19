"""
predict.py

FINAL locked prediction interface.
Always returns the same response schema.
"""

import joblib
from sklearn.metrics.pairwise import cosine_similarity
from recommendations import generate_recommendations


class ResIQPredictor:
    def __init__(self):
        self.vectorizer = joblib.load("models/vectorizer.joblib")

    def predict(self, resume_text: str, job_text: str) -> dict:
        # Defensive defaults
        response = {
            "prediction": 0,
            "confidence": 0.0,
            "explanation": [],
            "issues": [],
            "recommendations": []
        }

        try:
            if not resume_text.strip() or not job_text.strip():
                response["issues"].append("Missing resume or job description text.")
                response["recommendations"].append(
                    "Ensure both inputs contain readable text."
                )
                return response

            vectors = self.vectorizer.transform([resume_text, job_text])
            similarity = float(
                cosine_similarity(vectors[0], vectors[1])[0][0]
            )

            prediction = int(similarity >= 0.6)

            insights = generate_recommendations(
                resume_text=resume_text,
                job_text=job_text,
                similarity_score=similarity,
                is_match=bool(prediction),
                vectorizer=self.vectorizer
            )


            response.update({
                "prediction": prediction,
                "confidence": round(similarity, 3),
                "explanation": insights.get("explanation", []),
                "issues": insights.get("issues", []),
                "recommendations": insights.get("recommendations", [])
            })

            return response

        except Exception as e:
            response["issues"].append(
                "Internal error occurred during analysis."
            )
            response["recommendations"].append(
                "Try uploading a different resume file."
            )
            return response
