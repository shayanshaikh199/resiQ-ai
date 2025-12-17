"""
predict.py

Semantic resume–job matcher with explainability.
"""

from sanity import looks_like_resume
from embedding_matcher import EmbeddingMatcher
from explainer import MatchExplainer

MATCH_THRESHOLD = 0.55
UNCERTAIN_MARGIN = 0.05


class ResIQPredictor:
    def __init__(self):
        self.matcher = EmbeddingMatcher()
        self.explainer = MatchExplainer()

    def predict(self, resume_text: str, job_text: str):
        # Step 1: Sanity check — is this even a resume?
        if not looks_like_resume(resume_text):
            return {
                "prediction": 0,
                "confidence": 0.0,
                "uncertain": False,
                "reason": "Uploaded document does not appear to be a resume",
                "keywords": []
            }

        # Step 2: Semantic similarity
        similarity = self.matcher.similarity(resume_text, job_text)

        prediction = int(similarity >= MATCH_THRESHOLD)
        uncertain = abs(similarity - MATCH_THRESHOLD) <= UNCERTAIN_MARGIN

        # Step 3: Explain WHY it matched (or not)
        keywords = self.explainer.explain(resume_text, job_text)

        return {
            "prediction": prediction,
            "confidence": round(similarity, 3),
            "uncertain": uncertain,
            "keywords": keywords
        }
